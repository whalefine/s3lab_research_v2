// =============================================================================
// head_top.v
//
// CenterPredictor Head Top，對齊 run_backbone_numpy head_branch × 3 + cal_bbox。
//
// 架構：
//   opt_feat: [1, IN_CH=768, 16, 16]  (backbone 輸出 reshape 後)
//
//   score_map_ctr = head_branch(opt_feat, "ctr",    DO_SIGMOID=1)  [1,1,16,16]
//   size_map      = head_branch(opt_feat, "size",   DO_SIGMOID=1)  [1,2,16,16]
//   offset_map    = head_branch(opt_feat, "offset", DO_SIGMOID=0)  [1,2,16,16]
//
//   bbox = cal_bbox(score_map_ctr, size_map, offset_map)
//          → [cx_o, cy_o, w_o, h_o] (Q8.8)
//
// 三個 branch 可共用 opt_feat 輸入（依序執行）或同時執行（面積 ×3）。
// 此實作採用「依序執行 ctr → size → offset → cal_bbox」策略（面積最小）。
//
// 外部 controller 提供 (a_i, w_i) 串流；head_top 依 state 選擇對應 branch weights。
// =============================================================================

module head_top #(
    parameter IN_CH    = 768,
    parameter C1       = 256,
    parameter C2       = 128,
    parameter C3       = 64,
    parameter C4       = 32,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // opt_feat input (replayed 3 times by external controller)
    input  wire signed [15:0] a_i,
    input  wire signed [15:0] w_i,
    input  wire signed [15:0] b_i,
    input  wire        a_valid,

    // Branch weight bank select (driven by wgt_bank_o; external ROM mux)
    output reg  [1:0]  wgt_bank_o,  // 0=ctr, 1=size, 2=offset

    // Status
    output wire        busy,
    output reg         done,

    // bbox output (Q8.8)
    output wire [15:0] cx_o,
    output wire [15:0] cy_o,
    output wire [15:0] w_o,
    output wire [15:0] h_o
);

localparam FEAT_SZ = FEAT_H * FEAT_W;

// FSM state encoding
parameter S_IDLE    = 3'd0;
parameter S_CTR     = 3'd1;   // ctr branch
parameter S_SIZE    = 3'd2;   // size branch
parameter S_OFFSET  = 3'd3;   // offset branch
parameter S_BBOX    = 3'd4;   // cal_bbox
parameter S_DONE    = 3'd5;

reg [2:0] state, next_state;

// Output buffers from branches
reg [15:0] score_buf [0:FEAT_SZ-1];    // ctr: 1×FEAT_SZ
reg [15:0] size0_buf [0:FEAT_SZ-1];    // size[0]: 1×FEAT_SZ
reg [15:0] size1_buf [0:FEAT_SZ-1];    // size[1]: 1×FEAT_SZ
reg [15:0] off0_buf  [0:FEAT_SZ-1];    // offset[0]
reg [15:0] off1_buf  [0:FEAT_SZ-1];    // offset[1]

// Branch instance (shared, muxed by state)
wire br_busy, br_done;
wire signed [15:0] br_y;
wire br_y_valid;
reg  br_start;

// ctr branch (DO_SIGMOID=1, OUT_CH=1)
head_branch #(
    .IN_CH(IN_CH), .C1(C1), .C2(C2), .C3(C3), .C4(C4),
    .OUT_CH(1), .FEAT_H(FEAT_H), .FEAT_W(FEAT_W), .DO_SIGMOID(1)
) u_ctr (
    .clk(clk), .reset(reset), .start(br_start && state==S_CTR),
    .a_i(a_i), .w_i(w_i), .b_i(b_i), .a_valid(a_valid),
    .busy(br_busy), .done(br_done), .y_o(br_y), .y_valid(br_y_valid)
);

// Store branch outputs and track pixel count
reg [8:0] pix_cnt;

// cal_bbox control
wire bbox_busy, bbox_done;
reg  bbox_start;
reg  [8:0] bbox_feed_cnt;
reg  bbox_score_v, bbox_size_v, bbox_off_v;

cal_bbox #(.FEAT_SZ(FEAT_H), .FEAT_LEN(FEAT_SZ)) u_bbox (
    .clk(clk), .reset(reset), .start(bbox_start),
    .score_i(score_buf[bbox_feed_cnt]),  .score_valid(bbox_score_v),
    .size_i (size0_buf[bbox_feed_cnt]),  .size_valid (bbox_size_v),
    .offset_i(off0_buf[bbox_feed_cnt]),  .offset_valid(bbox_off_v),
    .busy(bbox_busy), .done(bbox_done),
    .cx_o(cx_o), .cy_o(cy_o), .w_o(w_o), .h_o(h_o)
);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:   next_state = start    ? S_CTR   : S_IDLE;
        S_CTR:    next_state = br_done  ? S_SIZE   : S_CTR;
        S_SIZE:   next_state = br_done  ? S_OFFSET : S_SIZE;
        S_OFFSET: next_state = br_done  ? S_BBOX   : S_OFFSET;
        S_BBOX:   next_state = bbox_done ? S_DONE  : S_BBOX;
        S_DONE:   next_state = S_IDLE;
        default:  next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done       <= 1'b0;
    br_start   <= 1'b0;
    bbox_start <= 1'b0;
    bbox_score_v <= 1'b0;
    bbox_size_v  <= 1'b0;
    bbox_off_v   <= 1'b0;

    if (reset) begin
        pix_cnt      <= 9'd0;
        bbox_feed_cnt<= 9'd0;
        wgt_bank_o   <= 2'd0;
    end else begin
        case (state)
            S_IDLE: begin
                pix_cnt <= 9'd0;
            end

            S_CTR: begin
                wgt_bank_o <= 2'd0;
                if (!br_busy) br_start <= 1'b1;
                if (br_y_valid) begin
                    score_buf[pix_cnt] <= br_y;
                    pix_cnt <= pix_cnt + 9'd1;
                end
                if (br_done) pix_cnt <= 9'd0;
            end

            S_SIZE: begin
                wgt_bank_o <= 2'd1;
                if (!br_busy) br_start <= 1'b1;
                if (br_y_valid) begin
                    // First FEAT_SZ → size0_buf, next FEAT_SZ → size1_buf
                    if (pix_cnt < FEAT_SZ)
                        size0_buf[pix_cnt]          <= br_y;
                    else
                        size1_buf[pix_cnt - FEAT_SZ] <= br_y;
                    pix_cnt <= pix_cnt + 9'd1;
                end
                if (br_done) pix_cnt <= 9'd0;
            end

            S_OFFSET: begin
                wgt_bank_o <= 2'd2;
                if (!br_busy) br_start <= 1'b1;
                if (br_y_valid) begin
                    if (pix_cnt < FEAT_SZ)
                        off0_buf[pix_cnt]          <= br_y;
                    else
                        off1_buf[pix_cnt - FEAT_SZ] <= br_y;
                    pix_cnt <= pix_cnt + 9'd1;
                end
                if (br_done) pix_cnt <= 9'd0;
            end

            S_BBOX: begin
                // Feed all 5 buffers into cal_bbox serially
                if (!bbox_busy) begin
                    bbox_start    <= 1'b1;
                    bbox_feed_cnt <= 9'd0;
                end else begin
                    bbox_score_v <= 1'b1;
                    bbox_size_v  <= 1'b1;
                    bbox_off_v   <= 1'b1;
                    if (bbox_feed_cnt < FEAT_SZ-1)
                        bbox_feed_cnt <= bbox_feed_cnt + 9'd1;
                end
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
