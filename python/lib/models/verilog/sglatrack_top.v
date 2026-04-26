// =============================================================================
// sglatrack_top.v
//
// SGLATrack 頂層模組，對齊 run_backbone_numpy.main() 完整推論流程。
//
// 範圍：post-embedding input → backbone → head → cal_bbox → pred_boxes。
// Tracker 後處理（/resize_factor、map_box_back、clip_box）留在軟體端。
//
// 介面：
//   clk, reset
//   start       : 開始一次推論
//   busy, done  : 狀態輸出
//
//   data_in [15:0]  : 串流輸入，Q8.8（post-embedding tokens）
//   data_valid      : data_in 有效
//   data_addr [19:0]: 外部 RAM 位址（由硬體輸出，外部控制器依此提供 data_in）
//
//   wgt_in  [15:0]  : 串流權重輸入，Q8.8（from testbench ROM）
//   wgt_bias[15:0]  : 串流 bias 輸入，Q8.8
//   wgt_addr[19:0]  : 權重 ROM 位址（backbone block × feature）
//   wgt_bank [1:0]  : head branch 選擇（0=ctr,1=size,2=offset）
//
//   cx_o, cy_o, w_o, h_o [15:0]: 輸出 bbox（Q8.8 normalized）
//
// 架構：
//   backbone_top → head_top（opt_feat = backbone_out[-256:] reshape 16×16）
// =============================================================================

module sglatrack_top #(
    parameter EMBED_DIM   = 768,
    parameter N_TOKENS    = 320,
    parameter SEARCH_TOKS = 256,   // search token count (last N_TOKENS-64)
    parameter FEAT_H      = 16,
    parameter FEAT_W      = 16
) (
    input  wire        clk,
    input  wire        reset,

    // Top-level control
    input  wire        start,
    output wire        busy,
    output reg         done,

    // Input token stream
    input  wire signed [15:0] data_in,
    input  wire        data_valid,

    // Weight / bias ROM interface
    input  wire signed [15:0] wgt_in,
    input  wire signed [15:0] wgt_bias,
    output wire [19:0] wgt_addr,
    output wire [1:0]  wgt_bank,    // head branch select

    // Output bbox (Q8.8)
    output wire [15:0] cx_o,
    output wire [15:0] cy_o,
    output wire [15:0] w_o,
    output wire [15:0] h_o
);

// FSM state encoding
parameter S_IDLE      = 2'd0;
parameter S_BACKBONE  = 2'd1;
parameter S_HEAD      = 2'd2;
parameter S_DONE      = 2'd3;

reg [1:0] state, next_state;

// backbone_top signals
wire bb_busy, bb_done;
wire signed [15:0] bb_y;
wire bb_y_valid;
wire [19:0] bb_wgt_addr;
wire [3:0]  bb_block_idx;
reg  bb_start;

backbone_top #(
    .EMBED_DIM  (EMBED_DIM),
    .N_TOKENS   (N_TOKENS)
) u_backbone (
    .clk(clk), .reset(reset), .start(bb_start),
    .x_i(data_in), .x_valid(data_valid),
    .wgt_i(wgt_in), .bias_i(wgt_bias),
    .wgt_addr_o(bb_wgt_addr),
    .block_idx_o(bb_block_idx),
    .busy(bb_busy), .done(bb_done),
    .y_o(bb_y), .y_valid(bb_y_valid)
);

// head_top signals
wire hd_busy, hd_done;
wire [1:0] hd_wgt_bank;
reg  hd_start;
// opt_feat = last SEARCH_TOKS × EMBED_DIM tokens of backbone_out, reshaped to [EMBED_DIM, 16, 16]
// External controller reshapes; we stream backbone_out directly to head after reshape.

head_top #(
    .IN_CH  (EMBED_DIM),
    .FEAT_H (FEAT_H),
    .FEAT_W (FEAT_W)
) u_head (
    .clk(clk), .reset(reset), .start(hd_start),
    .a_i(bb_y), .w_i(wgt_in), .b_i(wgt_bias), .a_valid(bb_y_valid),
    .wgt_bank_o(hd_wgt_bank),
    .busy(hd_busy), .done(hd_done),
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
        S_IDLE:     next_state = start   ? S_BACKBONE : S_IDLE;
        S_BACKBONE: next_state = bb_done ? S_HEAD     : S_BACKBONE;
        S_HEAD:     next_state = hd_done ? S_DONE     : S_HEAD;
        S_DONE:     next_state = S_IDLE;
        default:    next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done     <= 1'b0;
    bb_start <= 1'b0;
    hd_start <= 1'b0;

    if (reset) begin
        ;
    end else begin
        case (state)
            S_IDLE: begin
                if (start) bb_start <= 1'b1;
            end

            S_BACKBONE: begin
                // backbone_top streams out when done; head receives via bb_y/bb_y_valid
            end

            S_HEAD: begin
                if (!hd_busy) hd_start <= 1'b1;
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

assign busy     = (state != S_IDLE);
assign wgt_addr = (state == S_BACKBONE) ? bb_wgt_addr : 20'd0;
assign wgt_bank = hd_wgt_bank;

endmodule
