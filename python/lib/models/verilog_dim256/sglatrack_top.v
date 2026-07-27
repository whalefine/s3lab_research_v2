// =============================================================================
// sglatrack_top.v  (verilog_dim256)
// -----------------------------------------------------------------------------
// Merged dim256 backbone + CenterPredictor head (no ROM/SRAM macros).
//
// Top FSM: TS_IDLE -> TS_BACKBONE -> TS_RESHAPE -> TS_HEAD -> TS_DONE
// RESHAPE: tok1 token-flat search -> OPT NCHW
//   opt[c*256+h*16+w] = tok1[(64+h*16+w)*256+c]
//
// Golden: box_head_after_forward_head_pred_boxes_bi.txt
// =============================================================================

`include "inv_sqrt_lut_seed.v"
`include "inv_sqrt_nr.v"
`include "recip_lut_seed.v"
`include "recip_nr.v"
`include "residual.v"
`include "layer_norm_pip.v"
`include "care_attention.v"
`include "mlp_ws.v"
`include "transformer_block.v"
`include "backbone_top.v"
`include "sigmoid_lut.v"
`include "cal_bbox.v"
`include "conv2d_ws.v"
`include "head_branch.v"
`include "head_top.v"

module sglatrack_top #(
    parameter EMBED_DIM = 256,
    parameter N_TOKENS  = 320,
    parameter DATA_W    = 16,
    parameter X_AW      = 17
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire [3:0]  sel_block_i,
    input  wire signed [DATA_W-1:0] data_in,
    input  wire        data_valid,
    output wire        x_ready,
    output wire        busy,
    output wire        done,

    // Shared / phase-muxed SRAM: tok1 (BB + reshape read)
    output wire        sram_tok1_ceb_o,
    output wire        sram_tok1_web_o,
    output wire [X_AW-1:0] sram_tok1_addr_o,
    output wire [DATA_W-1:0] sram_tok1_din_o,
    input  wire [DATA_W-1:0] sram_tok1_q_i,

    output wire        sram_tok2_ceb_o,
    output wire        sram_tok2_web_o,
    output wire [X_AW-1:0] sram_tok2_addr_o,
    output wire [DATA_W-1:0] sram_tok2_din_o,
    input  wire [DATA_W-1:0] sram_tok2_q_i,

    output wire        sram_q_ceb_o,
    output wire        sram_q_web_o,
    output wire [X_AW-1:0] sram_q_addr_o,
    output wire [DATA_W-1:0] sram_q_din_o,
    input  wire [DATA_W-1:0] sram_q_q_i,

    output wire        sram_k_ceb_o,
    output wire        sram_k_web_o,
    output wire [X_AW-1:0] sram_k_addr_o,
    output wire [DATA_W-1:0] sram_k_din_o,
    input  wire [DATA_W-1:0] sram_k_q_i,

    output wire        sram_v_ceb_o,
    output wire        sram_v_web_o,
    output wire [X_AW-1:0] sram_v_addr_o,
    output wire [DATA_W-1:0] sram_v_din_o,
    input  wire [DATA_W-1:0] sram_v_q_i,

    output wire        sram_qkm_ceb_o,
    output wire        sram_qkm_web_o,
    output wire [X_AW-1:0] sram_qkm_addr_o,
    output wire [DATA_W-1:0] sram_qkm_din_o,
    input  wire [DATA_W-1:0] sram_qkm_q_i,

    // Head OPT / scratch / maps
    output wire        opt_ceb_o, opt_web_o,
    output wire [X_AW-1:0] opt_addr_o,
    output wire [DATA_W-1:0] opt_din_o,
    input  wire [DATA_W-1:0] opt_q_i,

    output wire        a_ceb_o, a_web_o,
    output wire [X_AW-1:0] a_addr_o,
    output wire [DATA_W-1:0] a_din_o,
    input  wire [DATA_W-1:0] a_q_i,

    output wire        b_ceb_o, b_web_o,
    output wire [X_AW-1:0] b_addr_o,
    output wire [DATA_W-1:0] b_din_o,
    input  wire [DATA_W-1:0] b_q_i,

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

    // Backbone weight ports (TB mux)
    output wire [3:0]  bb_block_idx_o,
    output wire [19:0] bb_wgt_addr_o,
    output wire [9:0]  bb_bias_addr_o,
    output wire [9:0]  bb_norm_feat_addr_o,
    output wire [9:0]  bb_bn_feat_addr_o,
    input  wire signed [DATA_W-1:0] bb_wgt_i,
    input  wire signed [DATA_W-1:0] bb_bias_i,
    input  wire signed [DATA_W-1:0] bb_norm_wgt_i,
    input  wire signed [DATA_W-1:0] bb_norm_bias_i,
    input  wire signed [DATA_W-1:0] bb_bn_wgt_i,
    input  wire signed [DATA_W-1:0] bb_bn_bias_i,

    // Head weight ports (TB mux)
    output wire [1:0]  hd_branch_sel_o,
    output wire [2:0]  hd_wgt_layer_o,
    output wire [19:0] hd_wgt_addr_o,
    output wire [8:0]  hd_bias_addr_o,
    input  wire signed [DATA_W-1:0] hd_wgt_i,
    input  wire signed [DATA_W-1:0] hd_bias_i,

    output wire        bbox_valid,
    output wire signed [DATA_W-1:0] bbox_data,
    output wire [1:0]  bbox_idx,

    // Debug / TB phase
    output wire [2:0]  top_state_o
);

localparam OPT_LEN = 65536; // 256*16*16

parameter TS_IDLE     = 3'd0;
parameter TS_BACKBONE = 3'd1;
parameter TS_RESHAPE  = 3'd2;
parameter TS_HEAD     = 3'd3;
parameter TS_DONE     = 3'd4;

reg [2:0] top_state, top_next;
reg       bb_start_r, hd_start_r;
reg       rs_done_r;

wire bb_busy, bb_done;
wire hd_busy, hd_done;

assign top_state_o = top_state;
assign busy = (top_state != TS_IDLE);
assign done = (top_state == TS_DONE);

// -------------------------------------------------------------------------
// Top FSM
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        top_state <= TS_IDLE;
    else
        top_state <= top_next;
end

always @(*) begin
    top_next = top_state;
    case (top_state)
        TS_IDLE:     top_next = start ? TS_BACKBONE : TS_IDLE;
        TS_BACKBONE: top_next = bb_done ? TS_RESHAPE : TS_BACKBONE;
        TS_RESHAPE:  top_next = rs_done_r ? TS_HEAD : TS_RESHAPE;
        TS_HEAD:     top_next = hd_done ? TS_DONE : TS_HEAD;
        TS_DONE:     top_next = TS_IDLE;
        default:     top_next = TS_IDLE;
    endcase
end

always @(posedge clk) begin
    bb_start_r <= 1'b0;
    hd_start_r <= 1'b0;
    if (!reset) begin
        case (top_state)
            TS_IDLE: begin
                if (start)
                    bb_start_r <= 1'b1;
            end
            TS_RESHAPE: begin
                if (rs_done_r)
                    hd_start_r <= 1'b1;
            end
            default: begin
            end
        endcase
    end
end

// -------------------------------------------------------------------------
// Backbone SRAM wires
// -------------------------------------------------------------------------
wire        bb_tok1_ceb, bb_tok1_web;
wire [X_AW-1:0] bb_tok1_addr;
wire [DATA_W-1:0] bb_tok1_din;
wire        bb_tok2_ceb, bb_tok2_web;
wire [X_AW-1:0] bb_tok2_addr;
wire [DATA_W-1:0] bb_tok2_din;
wire        bb_q_ceb, bb_q_web;
wire [X_AW-1:0] bb_q_addr;
wire [DATA_W-1:0] bb_q_din;
wire        bb_k_ceb, bb_k_web;
wire [X_AW-1:0] bb_k_addr;
wire [DATA_W-1:0] bb_k_din;
wire        bb_v_ceb, bb_v_web;
wire [X_AW-1:0] bb_v_addr;
wire [DATA_W-1:0] bb_v_din;
wire        bb_qkm_ceb, bb_qkm_web;
wire [X_AW-1:0] bb_qkm_addr;
wire [DATA_W-1:0] bb_qkm_din;

// Reshape ports
reg         rs_tok1_ceb, rs_tok1_web;
reg  [X_AW-1:0] rs_tok1_addr;
reg         rs_opt_ceb, rs_opt_web;
reg  [X_AW-1:0] rs_opt_addr;
reg  [DATA_W-1:0] rs_opt_din;

// Head ports
wire        hd_opt_ceb, hd_opt_web;
wire [X_AW-1:0] hd_opt_addr;
wire        hd_a_ceb, hd_a_web;
wire [X_AW-1:0] hd_a_addr;
wire [DATA_W-1:0] hd_a_din;
wire        hd_b_ceb, hd_b_web;
wire [X_AW-1:0] hd_b_addr;
wire [DATA_W-1:0] hd_b_din;
wire        hd_ctr_ceb, hd_ctr_web;
wire [X_AW-1:0] hd_ctr_addr;
wire [DATA_W-1:0] hd_ctr_din;
wire        hd_sz_ceb, hd_sz_web;
wire [X_AW-1:0] hd_sz_addr;
wire [DATA_W-1:0] hd_sz_din;
wire        hd_off_ceb, hd_off_web;
wire [X_AW-1:0] hd_off_addr;
wire [DATA_W-1:0] hd_off_din;

// -------------------------------------------------------------------------
// Phase mux
// -------------------------------------------------------------------------
wire phase_bb = (top_state == TS_BACKBONE) || (top_state == TS_IDLE);
wire phase_rs = (top_state == TS_RESHAPE);
wire phase_hd = (top_state == TS_HEAD);

assign sram_tok1_ceb_o  = phase_rs ? rs_tok1_ceb  : (phase_bb ? bb_tok1_ceb  : 1'b1);
assign sram_tok1_web_o  = phase_rs ? rs_tok1_web  : (phase_bb ? bb_tok1_web  : 1'b1);
assign sram_tok1_addr_o = phase_rs ? rs_tok1_addr : (phase_bb ? bb_tok1_addr : {X_AW{1'b0}});
assign sram_tok1_din_o  = phase_bb ? bb_tok1_din  : {DATA_W{1'b0}};

assign sram_tok2_ceb_o  = phase_bb ? bb_tok2_ceb  : 1'b1;
assign sram_tok2_web_o  = phase_bb ? bb_tok2_web  : 1'b1;
assign sram_tok2_addr_o = phase_bb ? bb_tok2_addr : {X_AW{1'b0}};
assign sram_tok2_din_o  = phase_bb ? bb_tok2_din  : {DATA_W{1'b0}};

assign sram_q_ceb_o  = phase_bb ? bb_q_ceb  : 1'b1;
assign sram_q_web_o  = phase_bb ? bb_q_web  : 1'b1;
assign sram_q_addr_o = phase_bb ? bb_q_addr : {X_AW{1'b0}};
assign sram_q_din_o  = phase_bb ? bb_q_din  : {DATA_W{1'b0}};

assign sram_k_ceb_o  = phase_bb ? bb_k_ceb  : 1'b1;
assign sram_k_web_o  = phase_bb ? bb_k_web  : 1'b1;
assign sram_k_addr_o = phase_bb ? bb_k_addr : {X_AW{1'b0}};
assign sram_k_din_o  = phase_bb ? bb_k_din  : {DATA_W{1'b0}};

assign sram_v_ceb_o  = phase_bb ? bb_v_ceb  : 1'b1;
assign sram_v_web_o  = phase_bb ? bb_v_web  : 1'b1;
assign sram_v_addr_o = phase_bb ? bb_v_addr : {X_AW{1'b0}};
assign sram_v_din_o  = phase_bb ? bb_v_din  : {DATA_W{1'b0}};

assign sram_qkm_ceb_o  = phase_bb ? bb_qkm_ceb  : 1'b1;
assign sram_qkm_web_o  = phase_bb ? bb_qkm_web  : 1'b1;
assign sram_qkm_addr_o = phase_bb ? bb_qkm_addr : {X_AW{1'b0}};
assign sram_qkm_din_o  = phase_bb ? bb_qkm_din  : {DATA_W{1'b0}};

assign opt_ceb_o  = phase_rs ? rs_opt_ceb  : (phase_hd ? hd_opt_ceb : 1'b1);
assign opt_web_o  = phase_rs ? rs_opt_web  : (phase_hd ? hd_opt_web : 1'b1);
assign opt_addr_o = phase_rs ? rs_opt_addr : (phase_hd ? hd_opt_addr : {X_AW{1'b0}});
assign opt_din_o  = phase_rs ? rs_opt_din  : {DATA_W{1'b0}};

assign a_ceb_o  = phase_hd ? hd_a_ceb  : 1'b1;
assign a_web_o  = phase_hd ? hd_a_web  : 1'b1;
assign a_addr_o = phase_hd ? hd_a_addr : {X_AW{1'b0}};
assign a_din_o  = phase_hd ? hd_a_din  : {DATA_W{1'b0}};

assign b_ceb_o  = phase_hd ? hd_b_ceb  : 1'b1;
assign b_web_o  = phase_hd ? hd_b_web  : 1'b1;
assign b_addr_o = phase_hd ? hd_b_addr : {X_AW{1'b0}};
assign b_din_o  = phase_hd ? hd_b_din  : {DATA_W{1'b0}};

assign ctr_map_ceb_o  = phase_hd ? hd_ctr_ceb  : 1'b1;
assign ctr_map_web_o  = phase_hd ? hd_ctr_web  : 1'b1;
assign ctr_map_addr_o = phase_hd ? hd_ctr_addr : {X_AW{1'b0}};
assign ctr_map_din_o  = phase_hd ? hd_ctr_din  : {DATA_W{1'b0}};

assign sz_map_ceb_o  = phase_hd ? hd_sz_ceb  : 1'b1;
assign sz_map_web_o  = phase_hd ? hd_sz_web  : 1'b1;
assign sz_map_addr_o = phase_hd ? hd_sz_addr : {X_AW{1'b0}};
assign sz_map_din_o  = phase_hd ? hd_sz_din  : {DATA_W{1'b0}};

assign off_map_ceb_o  = phase_hd ? hd_off_ceb  : 1'b1;
assign off_map_web_o  = phase_hd ? hd_off_web  : 1'b1;
assign off_map_addr_o = phase_hd ? hd_off_addr : {X_AW{1'b0}};
assign off_map_din_o  = phase_hd ? hd_off_din  : {DATA_W{1'b0}};

// -------------------------------------------------------------------------
// Backbone instance
// -------------------------------------------------------------------------
backbone_top #(
    .EMBED_DIM(EMBED_DIM),
    .N_TOKENS(N_TOKENS),
    .START_LAYER(5),
    .N_BLOCKS(7)
) u_backbone (
    .clk(clk), .reset(reset), .start(bb_start_r), .sel_block_i(sel_block_i),
    .x_i(data_in), .x_valid(data_valid), .x_ready(x_ready),
    .busy(bb_busy), .done(bb_done),
    .y_o(), .y_valid(),
    .sram_tok1_ceb_o(bb_tok1_ceb), .sram_tok1_web_o(bb_tok1_web),
    .sram_tok1_addr_o(bb_tok1_addr), .sram_tok1_din_o(bb_tok1_din),
    .sram_tok1_q_i(sram_tok1_q_i),
    .sram_tok2_ceb_o(bb_tok2_ceb), .sram_tok2_web_o(bb_tok2_web),
    .sram_tok2_addr_o(bb_tok2_addr), .sram_tok2_din_o(bb_tok2_din),
    .sram_tok2_q_i(sram_tok2_q_i),
    .sram_q_ceb_o(bb_q_ceb), .sram_q_web_o(bb_q_web),
    .sram_q_addr_o(bb_q_addr), .sram_q_din_o(bb_q_din), .sram_q_q_i(sram_q_q_i),
    .sram_k_ceb_o(bb_k_ceb), .sram_k_web_o(bb_k_web),
    .sram_k_addr_o(bb_k_addr), .sram_k_din_o(bb_k_din), .sram_k_q_i(sram_k_q_i),
    .sram_v_ceb_o(bb_v_ceb), .sram_v_web_o(bb_v_web),
    .sram_v_addr_o(bb_v_addr), .sram_v_din_o(bb_v_din), .sram_v_q_i(sram_v_q_i),
    .sram_qkm_ceb_o(bb_qkm_ceb), .sram_qkm_web_o(bb_qkm_web),
    .sram_qkm_addr_o(bb_qkm_addr), .sram_qkm_din_o(bb_qkm_din),
    .sram_qkm_q_i(sram_qkm_q_i),
    .block_idx_o(bb_block_idx_o), .wgt_addr_o(bb_wgt_addr_o),
    .bias_addr_o(bb_bias_addr_o), .norm_feat_addr_o(bb_norm_feat_addr_o),
    .wgt_i(bb_wgt_i), .bias_i(bb_bias_i),
    .norm_wgt_i(bb_norm_wgt_i), .norm_bias_i(bb_norm_bias_i),
    .bn_feat_addr_o(bb_bn_feat_addr_o),
    .bn_wgt_i(bb_bn_wgt_i), .bn_bias_i(bb_bn_bias_i)
);

// -------------------------------------------------------------------------
// Head instance
// -------------------------------------------------------------------------
head_top u_head (
    .clk(clk), .reset(reset), .start(hd_start_r),
    .busy(hd_busy), .done(hd_done),
    .opt_ceb_o(hd_opt_ceb), .opt_web_o(hd_opt_web),
    .opt_addr_o(hd_opt_addr), .opt_q_i(opt_q_i),
    .a_ceb_o(hd_a_ceb), .a_web_o(hd_a_web),
    .a_addr_o(hd_a_addr), .a_din_o(hd_a_din), .a_q_i(a_q_i),
    .b_ceb_o(hd_b_ceb), .b_web_o(hd_b_web),
    .b_addr_o(hd_b_addr), .b_din_o(hd_b_din), .b_q_i(b_q_i),
    .ctr_map_ceb_o(hd_ctr_ceb), .ctr_map_web_o(hd_ctr_web),
    .ctr_map_addr_o(hd_ctr_addr), .ctr_map_din_o(hd_ctr_din),
    .ctr_map_q_i(ctr_map_q_i),
    .sz_map_ceb_o(hd_sz_ceb), .sz_map_web_o(hd_sz_web),
    .sz_map_addr_o(hd_sz_addr), .sz_map_din_o(hd_sz_din),
    .sz_map_q_i(sz_map_q_i),
    .off_map_ceb_o(hd_off_ceb), .off_map_web_o(hd_off_web),
    .off_map_addr_o(hd_off_addr), .off_map_din_o(hd_off_din),
    .off_map_q_i(off_map_q_i),
    .branch_sel_o(hd_branch_sel_o), .wgt_layer_o(hd_wgt_layer_o),
    .wgt_addr_o(hd_wgt_addr_o), .bias_addr_o(hd_bias_addr_o),
    .wgt_i(hd_wgt_i), .bias_i(hd_bias_i),
    .bbox_valid(bbox_valid), .bbox_data(bbox_data), .bbox_idx(bbox_idx)
);

// -------------------------------------------------------------------------
// RESHAPE FSM: flat NCHW index i = c*256 + h*16 + w
// tok_addr = (64 + h*16 + w)*256 + c
// -------------------------------------------------------------------------
reg [16:0] rs_idx;
reg [1:0]  rs_sub; // 0=issue rd, 1=cap+wr
reg [DATA_W-1:0] rs_data_r;
reg        rs_active;

wire [7:0] rs_c = rs_idx[15:8];
wire [7:0] rs_sp = rs_idx[7:0];
wire [3:0] rs_h = rs_sp[7:4];
wire [3:0] rs_w = rs_sp[3:0];
// tok in [64,319]: 9-bit; addr = tok*256 + c
wire [8:0] rs_tok_i = 9'd64 + ({5'd0, rs_h} << 4) + {5'd0, rs_w};
wire [16:0] rs_tok_addr = {rs_tok_i, rs_c};

always @(posedge clk) begin
    if (reset) begin
        rs_idx <= 17'd0;
        rs_sub <= 2'd0;
        rs_done_r <= 1'b0;
        rs_active <= 1'b0;
        rs_data_r <= {DATA_W{1'b0}};
    end else if (top_state != TS_RESHAPE) begin
        rs_idx <= 17'd0;
        rs_sub <= 2'd0;
        rs_done_r <= 1'b0;
        rs_active <= 1'b0;
    end else begin
        rs_active <= 1'b1;
        rs_done_r <= 1'b0;
        case (rs_sub)
            2'd0: begin
                rs_sub <= 2'd1;
            end
            2'd1: begin
                rs_data_r <= sram_tok1_q_i;
                rs_sub <= 2'd2;
            end
            2'd2: begin
                // write happens combinationally this cycle
                if (rs_idx == OPT_LEN[16:0] - 17'd1)
                    rs_done_r <= 1'b1;
                else begin
                    rs_idx <= rs_idx + 17'd1;
                    rs_sub <= 2'd0;
                end
            end
            default: rs_sub <= 2'd0;
        endcase
    end
end

always @(*) begin
    rs_tok1_ceb = 1'b1;
    rs_tok1_web = 1'b1;
    rs_tok1_addr = {X_AW{1'b0}};
    rs_opt_ceb = 1'b1;
    rs_opt_web = 1'b1;
    rs_opt_addr = {X_AW{1'b0}};
    rs_opt_din = {DATA_W{1'b0}};
    if (top_state == TS_RESHAPE) begin
        if (rs_sub == 2'd0) begin
            rs_tok1_ceb = 1'b0;
            rs_tok1_web = 1'b1;
            rs_tok1_addr = rs_tok_addr;
        end
        if (rs_sub == 2'd2) begin
            rs_opt_ceb = 1'b0;
            rs_opt_web = 1'b0;
            rs_opt_addr = rs_idx;
            rs_opt_din = rs_data_r;
        end
    end
end

endmodule
