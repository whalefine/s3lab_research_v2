// =============================================================================
// cal_bbox.v
//
// Bounding box extraction，對齊 run_backbone_numpy.cal_bbox()。
//
// 輸入：
//   score_map  [1, FEAT_SZ²]  Q8.8（sigmoid 後，ctr branch 輸出）
//   size_map   [2, FEAT_SZ²]  Q8.8（sigmoid 後，size branch 輸出）
//   offset_map [2, FEAT_SZ²]  Q8.8（offset branch 輸出）
//
// 輸出（Q8.8，對應 pred_boxes）：
//   cx_o = (idx_x × 256 + offset_x_int) >> 4      / 256.0 = cx/FEAT_SZ
//   cy_o = (idx_y × 256 + offset_y_int) >> 4      / 256.0 = cy/FEAT_SZ
//   w_o  = size_map[0][idx]
//   h_o  = size_map[1][idx]
//
// FEAT_SZ = 16（2^4），因此 >> 4 對應 /FEAT_SZ。
//
// 演算法：
//   1. Argmax 遍歷 score_map（FEAT_SZ² = 256 values）
//   2. idx → (idx_y, idx_x)
//   3. 查 size/offset_map at idx
//   4. 計算 cx, cy（RTL integer shift）
//
// 所有輸入在 start 後以 stream 方式送入（FEAT_LEN cycles）；
// done 拉高時輸出 bbox 有效。
// =============================================================================

module cal_bbox #(
    parameter FEAT_SZ  = 16,        // spatial size (must be power of 2)
    parameter FEAT_LEN = 256,       // FEAT_SZ × FEAT_SZ
    parameter SHIFT    = 4          // log2(FEAT_SZ) = 4
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // Score map input stream (FEAT_LEN values, Q8.8)
    input  wire [15:0] score_i,
    input  wire        score_valid,

    // Size map [0] stream then [1] stream (2 × FEAT_LEN values)
    input  wire [15:0] size_i,
    input  wire        size_valid,

    // Offset map [0] stream then [1] stream (2 × FEAT_LEN values)
    input  wire [15:0] offset_i,
    input  wire        offset_valid,

    // Status
    output wire        busy,
    output reg         done,

    // Result (Q8.8 normalized)
    output reg [15:0] cx_o,   // cx / FEAT_SZ in Q8.8
    output reg [15:0] cy_o,   // cy / FEAT_SZ in Q8.8
    output reg [15:0] w_o,    // size_map[0] at argmax
    output reg [15:0] h_o     // size_map[1] at argmax
);

// FSM state encoding
parameter S_IDLE    = 3'd0;
parameter S_ARGMAX  = 3'd1;   // find argmax of score_map
parameter S_SIZE    = 3'd2;   // read size_map[0][idx] and [1][idx]
parameter S_OFFSET  = 3'd3;   // read offset_map[0][idx] and [1][idx]
parameter S_CALC    = 3'd4;   // compute cx, cy
parameter S_DONE    = 3'd5;

reg [2:0] state, next_state;
reg [8:0] cnt;       // general counter [0..FEAT_LEN-1]

// Buffers for maps
reg [15:0] score_buf  [0:FEAT_LEN-1];
reg [15:0] size0_buf  [0:FEAT_LEN-1];
reg [15:0] size1_buf  [0:FEAT_LEN-1];
reg [15:0] off0_buf   [0:FEAT_LEN-1];
reg [15:0] off1_buf   [0:FEAT_LEN-1];

// Argmax result
reg [7:0]  max_score;
reg [7:0]  argmax_idx;

// S_CALC temporaries (blocking assigns; must be module-level for Verilog-2001)
reg [3:0]  calc_idx_x;
reg [3:0]  calc_idx_y;
reg [15:0] calc_offset_x;
reg [15:0] calc_offset_y;
reg [19:0] calc_cx_raw;
reg [19:0] calc_cy_raw;

// Address load stage
reg [1:0] load_phase;  // 0=score, 1=size0, 2=size1, 3=offset(0+1)

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:   next_state = start  ? S_ARGMAX : S_IDLE;
        S_ARGMAX: next_state = (cnt == FEAT_LEN-1 && score_valid) ? S_SIZE   : S_ARGMAX;
        S_SIZE:   next_state = (cnt == 2*FEAT_LEN-1 && size_valid) ? S_OFFSET : S_SIZE;
        S_OFFSET: next_state = (cnt == 2*FEAT_LEN-1 && offset_valid) ? S_CALC : S_OFFSET;
        S_CALC:   next_state = S_DONE;
        S_DONE:   next_state = S_IDLE;
        default:  next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done <= 1'b0;

    if (reset) begin
        cnt       <= 9'd0;
        max_score <= 8'd0;
        argmax_idx<= 8'd0;
    end else begin
        case (state)
            S_IDLE: begin
                cnt       <= 9'd0;
                max_score <= 8'd0;
                argmax_idx<= 8'd0;
            end

            S_ARGMAX: begin
                // Load score_map and find argmax simultaneously
                if (score_valid) begin
                    score_buf[cnt] <= score_i;
                    if (score_i[7:0] > max_score) begin
                        max_score  <= score_i[7:0];
                        argmax_idx <= cnt[7:0];
                    end
                    cnt <= cnt + 9'd1;
                end
            end

            S_SIZE: begin
                // First FEAT_LEN values → size0_buf, next FEAT_LEN → size1_buf
                if (size_valid) begin
                    if (cnt < FEAT_LEN)
                        size0_buf[cnt]            <= size_i;
                    else
                        size1_buf[cnt-FEAT_LEN]   <= size_i;
                    cnt <= cnt + 9'd1;
                end
            end

            S_OFFSET: begin
                if (offset_valid) begin
                    if (cnt < FEAT_LEN)
                        off0_buf[cnt]          <= offset_i;
                    else
                        off1_buf[cnt-FEAT_LEN] <= offset_i;
                    cnt <= cnt + 9'd1;
                end
            end

            S_CALC: begin
                // Compute bbox coordinates using RTL integer shifts
                // idx_x = argmax_idx % FEAT_SZ = argmax_idx[3:0]
                // idx_y = argmax_idx / FEAT_SZ = argmax_idx[7:4]
                // cx = (idx_x × 256 + offset_x_int) >> SHIFT (Q8.8 integer)
                calc_idx_x    = argmax_idx[3:0];
                calc_idx_y    = argmax_idx[7:4];
                calc_offset_x = off0_buf[argmax_idx];
                calc_offset_y = off1_buf[argmax_idx];

                calc_cx_raw = ({calc_idx_x, 8'd0} + calc_offset_x);
                calc_cy_raw = ({calc_idx_y, 8'd0} + calc_offset_y);

                cx_o <= calc_cx_raw[19:SHIFT];
                cy_o <= calc_cy_raw[19:SHIFT];
                w_o  <= size0_buf[argmax_idx];
                h_o  <= size1_buf[argmax_idx];
            end

            S_DONE: begin
                done <= 1'b1;
                cnt  <= 9'd0;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
