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
// offset 為 Q8.8 有號；與 numpy 一致須做 $signed 相加再算術右移 >>>SHIFT。
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
// 必須能數到 2*FEAT_LEN（預設 512）；9-bit 只到 511，S_SIZE/S_OFFSET 永遠無法 cnt==512 → 卡 state=2/3
reg [9:0] cnt;       // general counter: S_ARGMAX 至 FEAT_LEN；S_SIZE/S_OFFSET 至 2*FEAT_LEN
integer buf_i;

// Buffers for maps
reg [15:0] score_buf  [0:FEAT_LEN-1];
reg [15:0] size0_buf  [0:FEAT_LEN-1];
reg [15:0] size1_buf  [0:FEAT_LEN-1];
reg [15:0] off0_buf   [0:FEAT_LEN-1];
reg [15:0] off1_buf   [0:FEAT_LEN-1];

// Argmax result（與 numpy argmax 對齊：比較完整 Q8.8，而非僅小數 8-bit）
reg signed [15:0] max_score_q;
reg [7:0]  argmax_idx;

// S_CALC temporaries (blocking assigns; must be module-level for Verilog-2001)
reg [3:0]  calc_idx_x;
reg [3:0]  calc_idx_y;
reg [15:0] calc_offset_x;
reg [15:0] calc_offset_y;
reg signed [19:0] calc_sum_cx, calc_sum_cy;
reg signed [19:0] calc_sh_cx, calc_sh_cy;

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
// Exit streaming states when cnt reaches total samples — do not require
// *_valid on that cycle (TB / bbox_dv may drop the last beat same as
// layer_norm S_LOAD).  cnt is cleared on stage handoff in segment 3 below.
always @(*) begin
    case (state)
        S_IDLE:   next_state = start  ? S_ARGMAX : S_IDLE;
        S_ARGMAX: next_state = (cnt == FEAT_LEN)       ? S_SIZE   : S_ARGMAX;
        S_SIZE:   next_state = (cnt == 2*FEAT_LEN)     ? S_OFFSET : S_SIZE;
        S_OFFSET: next_state = (cnt == 2*FEAT_LEN)     ? S_CALC   : S_OFFSET;
        S_CALC:   next_state = S_DONE;
        S_DONE:   next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done <= 1'b0;

    if (reset) begin
        cnt             <= 10'd0;
        max_score_q     <= 16'sh8000;
        argmax_idx      <= 8'd0;
        for (buf_i = 0; buf_i < FEAT_LEN; buf_i = buf_i + 1) begin
            score_buf[buf_i] <= 16'sd0;
            size0_buf[buf_i] <= 16'sd0;
            size1_buf[buf_i] <= 16'sd0;
            off0_buf[buf_i]  <= 16'sd0;
            off1_buf[buf_i]  <= 16'sd0;
        end
    end else begin
        case (state)
            S_IDLE: begin
                cnt       <= 10'd0;
                max_score_q <= 16'sh8000;
                argmax_idx<= 8'd0;
            end

            S_ARGMAX: begin
                // Load score_map and find argmax simultaneously
                if (score_valid) begin
                    score_buf[cnt[7:0]] <= score_i;
                    if ($signed(score_i) > max_score_q) begin
                        max_score_q <= score_i;
                        argmax_idx <= cnt[7:0];
                    end
                    cnt <= cnt + 10'd1;
                end
            end

            S_SIZE: begin
                // First FEAT_LEN values → size0_buf, next FEAT_LEN → size1_buf
                if (size_valid) begin
                    if (cnt < FEAT_LEN)
                        size0_buf[cnt[7:0]]          <= size_i;
                    else
                        size1_buf[cnt - FEAT_LEN]    <= size_i;
                    cnt <= cnt + 10'd1;
                end
            end

            S_OFFSET: begin
                if (offset_valid) begin
                    if (cnt < FEAT_LEN)
                        off0_buf[cnt[7:0]]       <= offset_i;
                    else
                        off1_buf[cnt - FEAT_LEN] <= offset_i;
                    cnt <= cnt + 10'd1;
                end
            end

            S_CALC: begin
                // Compute bbox coordinates using RTL integer shifts
                // idx_x = argmax_idx % FEAT_SZ = argmax_idx[3:0]
                // idx_y = argmax_idx / FEAT_SZ = argmax_idx[7:4]
                // 對齊 numpy：((idx*256 + offset_q88_signed) >> SHIFT)，>> 為算術右移
                calc_idx_x    = argmax_idx[3:0];
                calc_idx_y    = argmax_idx[7:4];
                calc_offset_x = off0_buf[argmax_idx];
                calc_offset_y = off1_buf[argmax_idx];

                calc_sum_cx = $signed({6'b0, calc_idx_x, 8'b0}) +
                              $signed(calc_offset_x);
                calc_sum_cy = $signed({6'b0, calc_idx_y, 8'b0}) +
                              $signed(calc_offset_y);
                calc_sh_cx  = calc_sum_cx >>> SHIFT;
                calc_sh_cy  = calc_sum_cy >>> SHIFT;

                cx_o <= calc_sh_cx[15:0];
                cy_o <= calc_sh_cy[15:0];
                w_o  <= size0_buf[argmax_idx];
                h_o  <= size1_buf[argmax_idx];
            end

            S_DONE: begin
                done <= 1'b1;
                cnt  <= 10'd0;
            end

            default: ;
        endcase

        // Handoff: after last streamed sample cnt==FEAT_LEN / 2*FEAT_LEN, clear
        // counter for the next stage (avoids cnt=256 at S_SIZE entry).
        if (next_state == S_SIZE && state == S_ARGMAX)
            cnt <= 10'd0;
        else if (next_state == S_OFFSET && state == S_SIZE)
            cnt <= 10'd0;
        else if (next_state == S_CALC && state == S_OFFSET)
            cnt <= 10'd0;
    end
end

assign busy = (state != S_IDLE);

endmodule
