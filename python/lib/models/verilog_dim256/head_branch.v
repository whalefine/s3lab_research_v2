// =============================================================================
// head_branch.v  (verilog_dim256_head)
// -----------------------------------------------------------------------------
// CenterPredictor one branch (numpy head_branch):
//   conv1 256->256 (3x3 pad1 ReLU)  read OPT write B
//   conv2 256->128                  read B write A
//   conv3 128->64                   read A write B
//   conv4  64->32                   read B write A
//   conv5  32->OUT5 (1x1)           read A write B
//   if DO_SIGMOID: stream B through sigmoid_lut -> MAP
//   else:          copy B -> MAP
//
// OPT is read-only (shared across branches). A/B scratch depth >= 65536.
// MAP depth = OUT5*256.
// Golden: box_head_conv{1..5}_<branch>_out_bi.txt
// =============================================================================

module head_branch #(
    parameter OUT5       = 1,
    parameter DO_SIGMOID = 1,
    parameter DATA_W     = 16,
    parameter X_AW       = 17,
    parameter W_AW       = 20,
    parameter B_AW       = 9
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        busy,
    output reg         done,

    // Opt feat (read-only NCHW) — never overwritten by this module
    output reg         opt_ceb_o, opt_web_o,
    output reg  [X_AW-1:0] opt_addr_o,
    input  wire [DATA_W-1:0] opt_q_i,

    // Scratch bank A
    output reg         a_ceb_o, a_web_o,
    output reg  [X_AW-1:0] a_addr_o,
    output reg  [DATA_W-1:0] a_din_o,
    input  wire [DATA_W-1:0] a_q_i,

    // Scratch bank B
    output reg         b_ceb_o, b_web_o,
    output reg  [X_AW-1:0] b_addr_o,
    output reg  [DATA_W-1:0] b_din_o,
    input  wire [DATA_W-1:0] b_q_i,

    // Output map buffer
    output reg         map_ceb_o, map_web_o,
    output reg  [X_AW-1:0] map_addr_o,
    output reg  [DATA_W-1:0] map_din_o,

    input  wire signed [DATA_W-1:0] wgt_i,
    input  wire signed [DATA_W-1:0] bias_i,
    output reg  [2:0]  wgt_layer_o,
    output reg  [W_AW-1:0] wgt_addr_o,
    output reg  [B_AW-1:0] bias_addr_o
);

localparam S_IDLE = 4'd0;
localparam S_C1   = 4'd1;
localparam S_C2   = 4'd2;
localparam S_C3   = 4'd3;
localparam S_C4   = 4'd4;
localparam S_C5   = 4'd5;
localparam S_POST = 4'd6;  // sigmoid or copy
localparam S_DONE = 4'd7;

reg [3:0] state, next_state;
reg [3:0] prev_state;

wire c1_busy, c1_done, c2_busy, c2_done, c3_busy, c3_done;
wire c4_busy, c4_done, c5_busy, c5_done;

wire c1_xceb, c1_xweb; wire [X_AW-1:0] c1_xa; wire [DATA_W-1:0] c1_xq;
wire c1_yceb, c1_yweb; wire [X_AW-1:0] c1_ya; wire [DATA_W-1:0] c1_yd;
wire [W_AW-1:0] c1_wa; wire [B_AW-1:0] c1_ba;

wire c2_xceb, c2_xweb; wire [X_AW-1:0] c2_xa; wire [DATA_W-1:0] c2_xq;
wire c2_yceb, c2_yweb; wire [X_AW-1:0] c2_ya; wire [DATA_W-1:0] c2_yd;
wire [W_AW-1:0] c2_wa; wire [B_AW-1:0] c2_ba;

wire c3_xceb, c3_xweb; wire [X_AW-1:0] c3_xa; wire [DATA_W-1:0] c3_xq;
wire c3_yceb, c3_yweb; wire [X_AW-1:0] c3_ya; wire [DATA_W-1:0] c3_yd;
wire [W_AW-1:0] c3_wa; wire [B_AW-1:0] c3_ba;

wire c4_xceb, c4_xweb; wire [X_AW-1:0] c4_xa; wire [DATA_W-1:0] c4_xq;
wire c4_yceb, c4_yweb; wire [X_AW-1:0] c4_ya; wire [DATA_W-1:0] c4_yd;
wire [W_AW-1:0] c4_wa; wire [B_AW-1:0] c4_ba;

wire c5_xceb, c5_xweb; wire [X_AW-1:0] c5_xa; wire [DATA_W-1:0] c5_xq;
wire c5_yceb, c5_yweb; wire [X_AW-1:0] c5_ya; wire [DATA_W-1:0] c5_yd;
wire [W_AW-1:0] c5_wa; wire [B_AW-1:0] c5_ba;

assign c1_xq = opt_q_i;
assign c2_xq = b_q_i;
assign c3_xq = a_q_i;
assign c4_xq = b_q_i;
assign c5_xq = a_q_i;

wire enter = (state != prev_state);
wire pulse_c1 = (state == S_C1) && enter;
wire pulse_c2 = (state == S_C2) && enter;
wire pulse_c3 = (state == S_C3) && enter;
wire pulse_c4 = (state == S_C4) && enter;
wire pulse_c5 = (state == S_C5) && enter;

conv2d_ws #(.IN_CH(256), .OUT_CH(256), .K(3), .PAD(1), .HAS_RELU(1)) u_c1 (
    .clk(clk), .reset(reset), .start(pulse_c1),
    .busy(c1_busy), .done(c1_done),
    .x_ceb_o(c1_xceb), .x_web_o(c1_xweb), .x_addr_o(c1_xa), .x_q_i(c1_xq),
    .y_ceb_o(c1_yceb), .y_web_o(c1_yweb), .y_addr_o(c1_ya), .y_din_o(c1_yd),
    .wgt_i(wgt_i), .bias_i(bias_i), .wgt_addr_o(c1_wa), .bias_addr_o(c1_ba)
);
conv2d_ws #(.IN_CH(256), .OUT_CH(128), .K(3), .PAD(1), .HAS_RELU(1)) u_c2 (
    .clk(clk), .reset(reset), .start(pulse_c2),
    .busy(c2_busy), .done(c2_done),
    .x_ceb_o(c2_xceb), .x_web_o(c2_xweb), .x_addr_o(c2_xa), .x_q_i(c2_xq),
    .y_ceb_o(c2_yceb), .y_web_o(c2_yweb), .y_addr_o(c2_ya), .y_din_o(c2_yd),
    .wgt_i(wgt_i), .bias_i(bias_i), .wgt_addr_o(c2_wa), .bias_addr_o(c2_ba)
);
conv2d_ws #(.IN_CH(128), .OUT_CH(64), .K(3), .PAD(1), .HAS_RELU(1)) u_c3 (
    .clk(clk), .reset(reset), .start(pulse_c3),
    .busy(c3_busy), .done(c3_done),
    .x_ceb_o(c3_xceb), .x_web_o(c3_xweb), .x_addr_o(c3_xa), .x_q_i(c3_xq),
    .y_ceb_o(c3_yceb), .y_web_o(c3_yweb), .y_addr_o(c3_ya), .y_din_o(c3_yd),
    .wgt_i(wgt_i), .bias_i(bias_i), .wgt_addr_o(c3_wa), .bias_addr_o(c3_ba)
);
conv2d_ws #(.IN_CH(64), .OUT_CH(32), .K(3), .PAD(1), .HAS_RELU(1)) u_c4 (
    .clk(clk), .reset(reset), .start(pulse_c4),
    .busy(c4_busy), .done(c4_done),
    .x_ceb_o(c4_xceb), .x_web_o(c4_xweb), .x_addr_o(c4_xa), .x_q_i(c4_xq),
    .y_ceb_o(c4_yceb), .y_web_o(c4_yweb), .y_addr_o(c4_ya), .y_din_o(c4_yd),
    .wgt_i(wgt_i), .bias_i(bias_i), .wgt_addr_o(c4_wa), .bias_addr_o(c4_ba)
);
conv2d_ws #(.IN_CH(32), .OUT_CH(OUT5), .FEAT_H(16), .FEAT_W(16),
            .K(1), .PAD(0), .HAS_RELU(0)) u_c5 (
    .clk(clk), .reset(reset), .start(pulse_c5),
    .busy(c5_busy), .done(c5_done),
    .x_ceb_o(c5_xceb), .x_web_o(c5_xweb), .x_addr_o(c5_xa), .x_q_i(c5_xq),
    .y_ceb_o(c5_yceb), .y_web_o(c5_yweb), .y_addr_o(c5_ya), .y_din_o(c5_yd),
    .wgt_i(wgt_i), .bias_i(bias_i), .wgt_addr_o(c5_wa), .bias_addr_o(c5_ba)
);

// Post: copy/sigmoid B -> MAP
// Timing: rd@sub0 -> cap@sub1 -> (sig pulse@sub2 / wait@sub3) -> wr@sub4
localparam MAP_LEN = OUT5 * 256;
reg [9:0]  post_idx;
reg [2:0]  post_sub;
reg        post_done_r;
wire       sig_out_v;
wire signed [DATA_W-1:0] sig_out;
reg signed [DATA_W-1:0] post_data_r;
reg        sig_in_v;

sigmoid_lut u_sig (
    .clk(clk), .rst_n(~reset),
    .in_valid(sig_in_v),
    .in_q88(post_data_r),
    .out_valid(sig_out_v),
    .out_q88(sig_out)
);

assign busy = (state != S_IDLE);

always @(posedge clk) begin
    if (reset) begin
        state <= S_IDLE;
        prev_state <= S_IDLE;
    end else begin
        prev_state <= state;
        state <= next_state;
    end
end

always @(*) begin
    next_state = state;
    case (state)
        S_IDLE: if (start) next_state = S_C1;
        S_C1:   if (c1_done) next_state = S_C2;
        S_C2:   if (c2_done) next_state = S_C3;
        S_C3:   if (c3_done) next_state = S_C4;
        S_C4:   if (c4_done) next_state = S_C5;
        S_C5:   if (c5_done) next_state = S_POST;
        S_POST: if (post_done_r) next_state = S_DONE;
        S_DONE: next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

always @(posedge clk) begin
    if (reset) done <= 1'b0;
    else if (state == S_DONE) done <= 1'b1;
    else done <= 1'b0;
end

// Post FSM
always @(posedge clk) begin
    if (reset) begin
        post_idx <= 10'd0;
        post_sub <= 3'd0;
        post_done_r <= 1'b0;
        sig_in_v <= 1'b0;
        post_data_r <= 16'sd0;
    end else if (state != S_POST) begin
        post_idx <= 10'd0;
        post_sub <= 3'd0;
        post_done_r <= 1'b0;
        sig_in_v <= 1'b0;
    end else begin
        sig_in_v <= 1'b0;
        case (post_sub)
            3'd0: begin
                // issue read B[post_idx]
                post_sub <= 3'd1;
            end
            3'd1: begin
                // capture (addr issued previous cycle)
                post_data_r <= $signed(b_q_i);
                if (DO_SIGMOID != 0)
                    post_sub <= 3'd2;
                else
                    post_sub <= 3'd4;
            end
            3'd2: begin
                // post_data_r stable; pulse sigmoid
                sig_in_v <= 1'b1;
                post_sub <= 3'd3;
            end
            3'd3: begin
                if (sig_out_v) begin
                    post_data_r <= sig_out;
                    post_sub <= 3'd4;
                end
            end
            3'd4: begin
                // write MAP (combo uses post_sub==4)
                if (post_idx == MAP_LEN[9:0] - 10'd1)
                    post_done_r <= 1'b1;
                else begin
                    post_idx <= post_idx + 10'd1;
                    post_sub <= 3'd0;
                end
            end
            default: post_sub <= 3'd0;
        endcase
    end
end

// SRAM / weight port mux
// c1: R opt W B; c2: R B W A; c3: R A W B; c4: R B W A; c5: R A W B
always @(*) begin
    opt_ceb_o = 1'b1; opt_web_o = 1'b1; opt_addr_o = {X_AW{1'b0}};
    a_ceb_o = 1'b1; a_web_o = 1'b1; a_addr_o = {X_AW{1'b0}}; a_din_o = {DATA_W{1'b0}};
    b_ceb_o = 1'b1; b_web_o = 1'b1; b_addr_o = {X_AW{1'b0}}; b_din_o = {DATA_W{1'b0}};
    map_ceb_o = 1'b1; map_web_o = 1'b1; map_addr_o = {X_AW{1'b0}}; map_din_o = {DATA_W{1'b0}};
    wgt_layer_o = 3'd0;
    wgt_addr_o = {W_AW{1'b0}};
    bias_addr_o = {B_AW{1'b0}};

    case (state)
        S_C1: begin
            opt_ceb_o = c1_xceb; opt_web_o = c1_xweb; opt_addr_o = c1_xa;
            b_ceb_o = c1_yceb; b_web_o = c1_yweb; b_addr_o = c1_ya; b_din_o = c1_yd;
            wgt_layer_o = 3'd1; wgt_addr_o = c1_wa; bias_addr_o = c1_ba;
        end
        S_C2: begin
            b_ceb_o = c2_xceb; b_web_o = c2_xweb; b_addr_o = c2_xa;
            a_ceb_o = c2_yceb; a_web_o = c2_yweb; a_addr_o = c2_ya; a_din_o = c2_yd;
            wgt_layer_o = 3'd2; wgt_addr_o = c2_wa; bias_addr_o = c2_ba;
        end
        S_C3: begin
            a_ceb_o = c3_xceb; a_web_o = c3_xweb; a_addr_o = c3_xa;
            b_ceb_o = c3_yceb; b_web_o = c3_yweb; b_addr_o = c3_ya; b_din_o = c3_yd;
            wgt_layer_o = 3'd3; wgt_addr_o = c3_wa; bias_addr_o = c3_ba;
        end
        S_C4: begin
            b_ceb_o = c4_xceb; b_web_o = c4_xweb; b_addr_o = c4_xa;
            a_ceb_o = c4_yceb; a_web_o = c4_yweb; a_addr_o = c4_ya; a_din_o = c4_yd;
            wgt_layer_o = 3'd4; wgt_addr_o = c4_wa; bias_addr_o = c4_ba;
        end
        S_C5: begin
            a_ceb_o = c5_xceb; a_web_o = c5_xweb; a_addr_o = c5_xa;
            b_ceb_o = c5_yceb; b_web_o = c5_yweb; b_addr_o = c5_ya; b_din_o = c5_yd;
            wgt_layer_o = 3'd5; wgt_addr_o = c5_wa; bias_addr_o = c5_ba;
        end
        S_POST: begin
            if (post_sub == 3'd0) begin
                b_ceb_o = 1'b0; b_web_o = 1'b1;
                b_addr_o = {{(X_AW-10){1'b0}}, post_idx};
            end
            if (post_sub == 3'd4) begin
                map_ceb_o = 1'b0; map_web_o = 1'b0;
                map_addr_o = {{(X_AW-10){1'b0}}, post_idx};
                map_din_o = post_data_r;
            end
        end
        default: ;
    endcase
end

endmodule
