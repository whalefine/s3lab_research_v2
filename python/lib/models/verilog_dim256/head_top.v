// =============================================================================
// head_top.v  (verilog_dim256_head)
// -----------------------------------------------------------------------------
// Serial CenterPredictor: BRANCH_CTR -> BRANCH_SIZE -> BRANCH_OFFSET -> BBOX
// Opt bank = read-only NCHW opt_feat (TB preload). Scratch A/B for intermediates.
// Three head_branch instances; TB muxes weights by branch_sel_o + wgt_layer_o.
//
// Golden: box_head_after_forward_head_pred_boxes_bi.txt
// =============================================================================

module head_top #(
    parameter DATA_W = 16,
    parameter X_AW   = 17,
    parameter W_AW   = 20,
    parameter B_AW   = 9
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        busy,
    output reg         done,

    // Read-only opt_feat NCHW
    output wire        opt_ceb_o, opt_web_o,
    output wire [X_AW-1:0] opt_addr_o,
    input  wire [DATA_W-1:0] opt_q_i,

    // Scratch banks
    output wire        a_ceb_o, a_web_o,
    output wire [X_AW-1:0] a_addr_o,
    output wire [DATA_W-1:0] a_din_o,
    input  wire [DATA_W-1:0] a_q_i,

    output wire        b_ceb_o, b_web_o,
    output wire [X_AW-1:0] b_addr_o,
    output wire [DATA_W-1:0] b_din_o,
    input  wire [DATA_W-1:0] b_q_i,

    // Per-branch map buffers
    output wire        ctr_map_ceb_o, ctr_map_web_o,
    output wire [X_AW-1:0] ctr_map_addr_o,
    output wire [DATA_W-1:0] ctr_map_din_o,
    input  wire [DATA_W-1:0] ctr_map_q_i,

    output wire        sz_map_ceb_o, sz_map_web_o,
    output wire [X_AW-1:0] sz_map_addr_o,
    output wire [DATA_W-1:0] sz_map_din_o,
    input  wire [DATA_W-1:0] sz_map_q_i,

    output wire        off_map_ceb_o, off_map_web_o,
    output wire [X_AW-1:0] off_map_addr_o,
    output wire [DATA_W-1:0] off_map_din_o,
    input  wire [DATA_W-1:0] off_map_q_i,

    // Weight mux hints for TB
    output wire [1:0]  branch_sel_o,  // 0=ctr 1=size 2=offset
    output wire [2:0]  wgt_layer_o,
    output wire [W_AW-1:0] wgt_addr_o,
    output wire [B_AW-1:0] bias_addr_o,
    input  wire signed [DATA_W-1:0] wgt_i,
    input  wire signed [DATA_W-1:0] bias_i,

    output wire        bbox_valid,
    output wire signed [DATA_W-1:0] bbox_data,
    output wire [1:0]  bbox_idx
);

localparam S_IDLE = 3'd0;
localparam S_CTR  = 3'd1;
localparam S_SIZE = 3'd2;
localparam S_OFF  = 3'd3;
localparam S_BBOX = 3'd4;
localparam S_DONE = 3'd5;

reg [2:0] state, next_state, prev_state;

wire ctr_busy, ctr_done, sz_busy, sz_done, off_busy, off_done;
wire bb_busy, bb_done;

wire        co_ceb, co_web; wire [X_AW-1:0] co_a;
wire        ca_ceb, ca_web; wire [X_AW-1:0] ca_a; wire [DATA_W-1:0] ca_d;
wire        cb_ceb, cb_web; wire [X_AW-1:0] cb_a; wire [DATA_W-1:0] cb_d;
wire        cm_ceb, cm_web; wire [X_AW-1:0] cm_a; wire [DATA_W-1:0] cm_d;
wire [2:0]  cl; wire [W_AW-1:0] cwa; wire [B_AW-1:0] cba;

wire        so_ceb, so_web; wire [X_AW-1:0] so_a;
wire        sa_ceb, sa_web; wire [X_AW-1:0] sa_a; wire [DATA_W-1:0] sa_d;
wire        sb_ceb, sb_web; wire [X_AW-1:0] sb_a; wire [DATA_W-1:0] sb_d;
wire        sm_ceb, sm_web; wire [X_AW-1:0] sm_a; wire [DATA_W-1:0] sm_d;
wire [2:0]  sl; wire [W_AW-1:0] swa; wire [B_AW-1:0] sba;

wire        oo_ceb, oo_web; wire [X_AW-1:0] oo_a;
wire        oa_ceb, oa_web; wire [X_AW-1:0] oa_a; wire [DATA_W-1:0] oa_d;
wire        ob_ceb, ob_web; wire [X_AW-1:0] ob_a; wire [DATA_W-1:0] ob_d;
wire        om_ceb, om_web; wire [X_AW-1:0] om_a; wire [DATA_W-1:0] om_d;
wire [2:0]  ol; wire [W_AW-1:0] owa; wire [B_AW-1:0] oba;

wire start_ctr = (state == S_CTR)  && (prev_state != S_CTR);
wire start_sz  = (state == S_SIZE) && (prev_state != S_SIZE);
wire start_off = (state == S_OFF)  && (prev_state != S_OFF);
wire start_bb  = (state == S_BBOX) && (prev_state != S_BBOX);

head_branch #(.OUT5(1), .DO_SIGMOID(1)) u_ctr (
    .clk(clk), .reset(reset), .start(start_ctr),
    .busy(ctr_busy), .done(ctr_done),
    .opt_ceb_o(co_ceb), .opt_web_o(co_web), .opt_addr_o(co_a), .opt_q_i(opt_q_i),
    .a_ceb_o(ca_ceb), .a_web_o(ca_web), .a_addr_o(ca_a), .a_din_o(ca_d), .a_q_i(a_q_i),
    .b_ceb_o(cb_ceb), .b_web_o(cb_web), .b_addr_o(cb_a), .b_din_o(cb_d), .b_q_i(b_q_i),
    .map_ceb_o(cm_ceb), .map_web_o(cm_web), .map_addr_o(cm_a), .map_din_o(cm_d),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_layer_o(cl), .wgt_addr_o(cwa), .bias_addr_o(cba)
);

head_branch #(.OUT5(2), .DO_SIGMOID(1)) u_sz (
    .clk(clk), .reset(reset), .start(start_sz),
    .busy(sz_busy), .done(sz_done),
    .opt_ceb_o(so_ceb), .opt_web_o(so_web), .opt_addr_o(so_a), .opt_q_i(opt_q_i),
    .a_ceb_o(sa_ceb), .a_web_o(sa_web), .a_addr_o(sa_a), .a_din_o(sa_d), .a_q_i(a_q_i),
    .b_ceb_o(sb_ceb), .b_web_o(sb_web), .b_addr_o(sb_a), .b_din_o(sb_d), .b_q_i(b_q_i),
    .map_ceb_o(sm_ceb), .map_web_o(sm_web), .map_addr_o(sm_a), .map_din_o(sm_d),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_layer_o(sl), .wgt_addr_o(swa), .bias_addr_o(sba)
);

head_branch #(.OUT5(2), .DO_SIGMOID(0)) u_off (
    .clk(clk), .reset(reset), .start(start_off),
    .busy(off_busy), .done(off_done),
    .opt_ceb_o(oo_ceb), .opt_web_o(oo_web), .opt_addr_o(oo_a), .opt_q_i(opt_q_i),
    .a_ceb_o(oa_ceb), .a_web_o(oa_web), .a_addr_o(oa_a), .a_din_o(oa_d), .a_q_i(a_q_i),
    .b_ceb_o(ob_ceb), .b_web_o(ob_web), .b_addr_o(ob_a), .b_din_o(ob_d), .b_q_i(b_q_i),
    .map_ceb_o(om_ceb), .map_web_o(om_web), .map_addr_o(om_a), .map_din_o(om_d),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_layer_o(ol), .wgt_addr_o(owa), .bias_addr_o(oba)
);

wire        bcc, bcw; wire [9:0] bca;
wire        bsc, bsw; wire [9:0] bsa;
wire        boc, bow; wire [9:0] boa;

cal_bbox #(.X_AW(10)) u_bbox (
    .clk(clk), .reset(reset), .start(start_bb),
    .busy(bb_busy), .done(bb_done),
    .ctr_ceb_o(bcc), .ctr_web_o(bcw), .ctr_addr_o(bca), .ctr_q_i(ctr_map_q_i),
    .sz_ceb_o(bsc), .sz_web_o(bsw), .sz_addr_o(bsa), .sz_q_i(sz_map_q_i),
    .off_ceb_o(boc), .off_web_o(bow), .off_addr_o(boa), .off_q_i(off_map_q_i),
    .bbox_valid(bbox_valid), .bbox_data(bbox_data), .bbox_idx(bbox_idx)
);

assign busy = (state != S_IDLE);

assign opt_ceb_o = (state==S_CTR) ? co_ceb : (state==S_SIZE) ? so_ceb : (state==S_OFF) ? oo_ceb : 1'b1;
assign opt_web_o = (state==S_CTR) ? co_web : (state==S_SIZE) ? so_web : (state==S_OFF) ? oo_web : 1'b1;
assign opt_addr_o= (state==S_CTR) ? co_a   : (state==S_SIZE) ? so_a   : (state==S_OFF) ? oo_a   : {X_AW{1'b0}};

assign a_ceb_o = (state==S_CTR) ? ca_ceb : (state==S_SIZE) ? sa_ceb : (state==S_OFF) ? oa_ceb : 1'b1;
assign a_web_o = (state==S_CTR) ? ca_web : (state==S_SIZE) ? sa_web : (state==S_OFF) ? oa_web : 1'b1;
assign a_addr_o= (state==S_CTR) ? ca_a   : (state==S_SIZE) ? sa_a   : (state==S_OFF) ? oa_a   : {X_AW{1'b0}};
assign a_din_o = (state==S_CTR) ? ca_d   : (state==S_SIZE) ? sa_d   : (state==S_OFF) ? oa_d   : {DATA_W{1'b0}};

assign b_ceb_o = (state==S_CTR) ? cb_ceb : (state==S_SIZE) ? sb_ceb : (state==S_OFF) ? ob_ceb : 1'b1;
assign b_web_o = (state==S_CTR) ? cb_web : (state==S_SIZE) ? sb_web : (state==S_OFF) ? ob_web : 1'b1;
assign b_addr_o= (state==S_CTR) ? cb_a   : (state==S_SIZE) ? sb_a   : (state==S_OFF) ? ob_a   : {X_AW{1'b0}};
assign b_din_o = (state==S_CTR) ? cb_d   : (state==S_SIZE) ? sb_d   : (state==S_OFF) ? ob_d   : {DATA_W{1'b0}};

assign ctr_map_ceb_o = (state==S_CTR) ? cm_ceb : (state==S_BBOX) ? bcc : 1'b1;
assign ctr_map_web_o = (state==S_CTR) ? cm_web : (state==S_BBOX) ? bcw : 1'b1;
assign ctr_map_addr_o= (state==S_CTR) ? cm_a   : (state==S_BBOX) ? {{(X_AW-10){1'b0}}, bca} : {X_AW{1'b0}};
assign ctr_map_din_o = (state==S_CTR) ? cm_d   : {DATA_W{1'b0}};

assign sz_map_ceb_o = (state==S_SIZE) ? sm_ceb : (state==S_BBOX) ? bsc : 1'b1;
assign sz_map_web_o = (state==S_SIZE) ? sm_web : (state==S_BBOX) ? bsw : 1'b1;
assign sz_map_addr_o= (state==S_SIZE) ? sm_a   : (state==S_BBOX) ? {{(X_AW-10){1'b0}}, bsa} : {X_AW{1'b0}};
assign sz_map_din_o = (state==S_SIZE) ? sm_d   : {DATA_W{1'b0}};

assign off_map_ceb_o = (state==S_OFF) ? om_ceb : (state==S_BBOX) ? boc : 1'b1;
assign off_map_web_o = (state==S_OFF) ? om_web : (state==S_BBOX) ? bow : 1'b1;
assign off_map_addr_o= (state==S_OFF) ? om_a   : (state==S_BBOX) ? {{(X_AW-10){1'b0}}, boa} : {X_AW{1'b0}};
assign off_map_din_o = (state==S_OFF) ? om_d   : {DATA_W{1'b0}};

assign wgt_layer_o = (state==S_CTR) ? cl : (state==S_SIZE) ? sl : (state==S_OFF) ? ol : 3'd0;
assign wgt_addr_o  = (state==S_CTR) ? cwa: (state==S_SIZE) ? swa: (state==S_OFF) ? owa: {W_AW{1'b0}};
assign bias_addr_o = (state==S_CTR) ? cba: (state==S_SIZE) ? sba: (state==S_OFF) ? oba: {B_AW{1'b0}};
// Combinational: must match state same cycle as wgt_layer (avoid 1-cycle lag)
assign branch_sel_o = (state==S_CTR) ? 2'd0 : (state==S_SIZE) ? 2'd1 : (state==S_OFF) ? 2'd2 : 2'd0;

always @(posedge clk) begin
    if (reset) begin
        state <= S_IDLE; prev_state <= S_IDLE;
    end else begin
        prev_state <= state;
        state <= next_state;
    end
end

always @(*) begin
    next_state = state;
    case (state)
        S_IDLE: if (start) next_state = S_CTR;
        S_CTR:  if (ctr_done) next_state = S_SIZE;
        S_SIZE: if (sz_done)  next_state = S_OFF;
        S_OFF:  if (off_done) next_state = S_BBOX;
        S_BBOX: if (bb_done)  next_state = S_DONE;
        S_DONE: next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

always @(posedge clk) begin
    if (reset) done <= 1'b0;
    else if (state == S_DONE) done <= 1'b1;
    else done <= 1'b0;
end

endmodule
