// =============================================================================
// conv2d_ws.v  (verilog_dim256)
// -----------------------------------------------------------------------------
// Q8.8 integer MAC Conv2D (RTL approx of run_backbone_numpy_dim256_q88.conv2d):
//   Numpy: float conv + fp()=np.round. RTL: integer MAC + rnd_shr8 (+128).
//   acc = sum(x_q88 * w_q88)
//   y   = sat16(rnd_shr8(acc) + bias_q88)
//   if HAS_RELU: y = max(y, 0)
//
// Timing-friendly: NO combinational / or % for feat decode.
 // Nested counters (oc,oh,ow,ic,kh,kw) + running wgt_addr.
// For FEAT_H=FEAT_W=16: NCHW addr = {c[7:0], h[3:0], w[3:0]} (shift only).
//
// No internal weight ROM. TB ROM / act SRAM contract (CLK=clk):
//   posedge T  : DUT presents x_addr / wgt_addr / bias_addr
//   posedge T+1: x_q_i / wgt_i / bias_i valid for addr@T
//
// Golden: Activation/box_head_conv*_out_bi.txt
// Golden-Weight: Weight/box_head_conv*_folded_weight_bi.txt
// =============================================================================

module conv2d_ws #(
    parameter IN_CH    = 256,
    parameter OUT_CH   = 256,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter K        = 3,
    parameter PAD      = 1,
    parameter HAS_RELU = 1,
    parameter DATA_W   = 16,
    parameter ACC_W    = 48,
    parameter X_AW     = 17,
    parameter W_AW     = 20,
    parameter B_AW     = 9
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output wire        busy,
    output reg         done,

    output reg         x_ceb_o,
    output reg         x_web_o,
    output reg  [X_AW-1:0] x_addr_o,
    input  wire [DATA_W-1:0] x_q_i,

    output reg         y_ceb_o,
    output reg         y_web_o,
    output reg  [X_AW-1:0] y_addr_o,
    output reg  [DATA_W-1:0] y_din_o,

    input  wire signed [DATA_W-1:0] wgt_i,
    input  wire signed [DATA_W-1:0] bias_i,
    output reg  [W_AW-1:0] wgt_addr_o,
    output reg  [B_AW-1:0] bias_addr_o
);

localparam KK          = K * K;
localparam FEAT_PER_OC = IN_CH * KK;
localparam OUT_H       = FEAT_H + 2 * PAD - K + 1;
localparam OUT_W       = FEAT_W + 2 * PAD - K + 1;

// Require 16x16 for shift-only NCHW (dim256 head). Other sizes need revisit.
// synthesis translate_off
initial begin
    if ((FEAT_H != 16) || (FEAT_W != 16) || (OUT_H != 16) || (OUT_W != 16))
        $display("[conv2d_ws] WARN: FEAT/OUT not 16x16; addr uses <<8/<<4 assumptions");
end
// synthesis translate_on

localparam S_IDLE  = 3'd0;
localparam S_BLOAD = 3'd1;
localparam S_MAC   = 3'd2;
localparam S_SAT   = 3'd3;
localparam S_DONE  = 3'd4;

reg [2:0] state, next_state;

reg [8:0]  oc_r;
reg [4:0]  oh_r, ow_r;
reg [8:0]  ic_r;       // 0..IN_CH-1
reg [1:0]  kh_r, kw_r; // 0..K-1 (K<=3)
reg        mac_fill;
reg        bload_phase;
reg [W_AW-1:0] wgt_run_r;  // current weight addr within oc
reg [W_AW-1:0] oc_base_r;  // oc * FEAT_PER_OC (updated by +FEAT_PER_OC only)

reg signed [ACC_W-1:0]  acc_r;
reg signed [DATA_W-1:0] bias_r;
reg                     issue_pad_r;

wire ic_last = (ic_r == IN_CH[8:0] - 9'd1);
wire kh_last = (kh_r == K[1:0] - 2'd1);
wire kw_last = (kw_r == K[1:0] - 2'd1);
wire feat_last = ic_last && kh_last && kw_last;
wire ow_last   = (ow_r == OUT_W[4:0] - 5'd1);
wire oh_last   = (oh_r == OUT_H[4:0] - 5'd1);
wire oc_last   = (oc_r == OUT_CH[8:0] - 9'd1);

// Current spatial with pad (combinational add only; no div)
wire signed [6:0] cur_ih = $signed({2'b0, oh_r}) + $signed({5'b0, kh_r}) - PAD[6:0];
wire signed [6:0] cur_iw = $signed({2'b0, ow_r}) + $signed({5'b0, kw_r}) - PAD[6:0];
wire cur_pad = (cur_ih < 0) || (cur_iw < 0)
            || (cur_ih >= FEAT_H[6:0]) || (cur_iw >= FEAT_W[6:0]);
// NCHW flat for 16x16: {c[7:0], h[3:0], w[3:0]}
wire [X_AW-1:0] cur_xaddr = {1'b0, ic_r[7:0], cur_ih[3:0], cur_iw[3:0]};

// Next counters (peek for pipeline issue of feat+1)
reg [8:0]  n_ic;
reg [1:0]  n_kh, n_kw;
always @(*) begin
    n_ic = ic_r;
    n_kh = kh_r;
    n_kw = kw_r;
    if (!kw_last)
        n_kw = kw_r + 2'd1;
    else if (!kh_last) begin
        n_kw = 2'd0;
        n_kh = kh_r + 2'd1;
    end else if (!ic_last) begin
        n_kw = 2'd0;
        n_kh = 2'd0;
        n_ic = ic_r + 9'd1;
    end
end

wire signed [6:0] nxt_ih = $signed({2'b0, oh_r}) + $signed({5'b0, n_kh}) - PAD[6:0];
wire signed [6:0] nxt_iw = $signed({2'b0, ow_r}) + $signed({5'b0, n_kw}) - PAD[6:0];
wire nxt_pad = (nxt_ih < 0) || (nxt_iw < 0)
            || (nxt_ih >= FEAT_H[6:0]) || (nxt_iw >= FEAT_W[6:0]);
wire [X_AW-1:0] nxt_xaddr = {1'b0, n_ic[7:0], nxt_ih[3:0], nxt_iw[3:0]};

wire [X_AW-1:0] y_addr_c = {1'b0, oc_r[7:0], oh_r[3:0], ow_r[3:0]};

// SAT: rnd_shr8 (align numpy fp / backbone MAC)
wire signed [ACC_W-1:0] sat_shr = (acc_r + 48'sd128) >>> 8;
wire signed [ACC_W-1:0] sat_add = sat_shr
    + {{(ACC_W-DATA_W){bias_r[DATA_W-1]}}, bias_r};
wire signed [ACC_W-1:0] lim_pos = 48'sd32767;
wire signed [ACC_W-1:0] lim_neg = -48'sd32768;
wire signed [DATA_W-1:0] sat16 =
    (sat_add > lim_pos) ? 16'sh7FFF :
    (sat_add < lim_neg) ? 16'sh8000 :
    sat_add[DATA_W-1:0];
wire signed [DATA_W-1:0] sat_relu =
    ((HAS_RELU != 0) && sat16[DATA_W-1]) ? 16'sd0 : sat16;

assign busy = (state != S_IDLE);

always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:  if (start) next_state = S_BLOAD;
        S_BLOAD: if (bload_phase) next_state = S_MAC;
        S_MAC:   if (!mac_fill && feat_last) next_state = S_SAT;
        S_SAT:   if (ow_last && oh_last && oc_last) next_state = S_DONE;
                 else next_state = S_BLOAD;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

always @(posedge clk) begin
    if (reset) done <= 1'b0;
    else if (state == S_DONE) done <= 1'b1;
    else done <= 1'b0;
end

always @(posedge clk) begin
    if (reset) begin
        oc_r <= 9'd0; oh_r <= 5'd0; ow_r <= 5'd0;
        ic_r <= 9'd0; kh_r <= 2'd0; kw_r <= 2'd0;
        mac_fill <= 1'b0; bload_phase <= 1'b0;
        wgt_run_r <= {W_AW{1'b0}};
        oc_base_r <= {W_AW{1'b0}};
    end else begin
        case (state)
            S_IDLE: if (start) begin
                oc_r <= 9'd0; oh_r <= 5'd0; ow_r <= 5'd0;
                ic_r <= 9'd0; kh_r <= 2'd0; kw_r <= 2'd0;
                mac_fill <= 1'b1; bload_phase <= 1'b0;
                wgt_run_r <= {W_AW{1'b0}};
                oc_base_r <= {W_AW{1'b0}};
            end
            S_BLOAD: begin
                if (!bload_phase) bload_phase <= 1'b1;
                else begin
                    bload_phase <= 1'b0;
                    mac_fill <= 1'b1;
                    ic_r <= 9'd0; kh_r <= 2'd0; kw_r <= 2'd0;
                    wgt_run_r <= oc_base_r;
                end
            end
            S_MAC: begin
                if (mac_fill) begin
                    mac_fill <= 1'b0;
                end else if (!feat_last) begin
                    if (!kw_last)
                        kw_r <= kw_r + 2'd1;
                    else if (!kh_last) begin
                        kw_r <= 2'd0;
                        kh_r <= kh_r + 2'd1;
                    end else begin
                        kw_r <= 2'd0;
                        kh_r <= 2'd0;
                        ic_r <= ic_r + 9'd1;
                    end
                    wgt_run_r <= wgt_run_r + {{(W_AW-1){1'b0}}, 1'b1};
                end
            end
            S_SAT: begin
                if (!(ow_last && oh_last && oc_last)) begin
                    if (!ow_last) begin
                        ow_r <= ow_r + 5'd1;
                    end else if (!oh_last) begin
                        ow_r <= 5'd0; oh_r <= oh_r + 5'd1;
                    end else begin
                        ow_r <= 5'd0; oh_r <= 5'd0; oc_r <= oc_r + 9'd1;
                        // constant add (no variable multiply)
                        oc_base_r <= oc_base_r + FEAT_PER_OC[W_AW-1:0];
                    end
                    bload_phase <= 1'b0;
                end
            end
            default: begin
            end
        endcase
    end
end

always @(posedge clk) begin
    if (reset) bias_r <= 16'sd0;
    else if (state == S_BLOAD && bload_phase) bias_r <= bias_i;
end

always @(posedge clk) begin
    if (reset) issue_pad_r <= 1'b0;
    else if (state == S_MAC) begin
        if (mac_fill) issue_pad_r <= cur_pad;
        else if (!feat_last) issue_pad_r <= nxt_pad;
    end
end

wire signed [31:0] mac_prod_w = $signed(x_q_i) * $signed(wgt_i);

always @(posedge clk) begin
    if (reset) acc_r <= {ACC_W{1'b0}};
    else if (state == S_BLOAD && bload_phase) acc_r <= {ACC_W{1'b0}};
    else if (state == S_MAC && !mac_fill && !issue_pad_r)
        acc_r <= acc_r + {{(ACC_W-32){mac_prod_w[31]}}, mac_prod_w};
end

always @(*) begin
    x_ceb_o = 1'b1; x_web_o = 1'b1; x_addr_o = {X_AW{1'b0}};
    y_ceb_o = 1'b1; y_web_o = 1'b1; y_addr_o = {X_AW{1'b0}};
    y_din_o = {DATA_W{1'b0}};
    wgt_addr_o = {W_AW{1'b0}};
    bias_addr_o = {B_AW{1'b0}};

    if (state == S_BLOAD && !bload_phase) begin
        bias_addr_o = oc_r[B_AW-1:0];
    end else if (state == S_MAC && mac_fill) begin
        if (!cur_pad) begin
            x_ceb_o = 1'b0; x_web_o = 1'b1; x_addr_o = cur_xaddr;
        end
        wgt_addr_o = wgt_run_r;
    end else if (state == S_MAC && !mac_fill && !feat_last) begin
        if (!nxt_pad) begin
            x_ceb_o = 1'b0; x_web_o = 1'b1; x_addr_o = nxt_xaddr;
        end
        wgt_addr_o = wgt_run_r + {{(W_AW-1){1'b0}}, 1'b1};
    end else if (state == S_SAT) begin
        y_ceb_o = 1'b0; y_web_o = 1'b0;
        y_addr_o = y_addr_c;
        y_din_o = sat_relu;
    end
end

endmodule
