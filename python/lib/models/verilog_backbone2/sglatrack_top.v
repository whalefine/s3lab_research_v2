// =============================================================================
// sglatrack_top.v -- verilog_backbone2 wrapper (backbone-only, Plan B)
//
// RTL below: `include dependency order (leaf -> top). Same-directory paths.
// IDE: .vscode/settings.json adds this folder to verilog.includePaths.
// VCS: also pass memory/rom_*.v, Sram_*.v, TEST_backbone.v (do not duplicate RTL
//      .v on cmd line if using only this file + includes).
//
// Plan B: backbone norm stays in Sram_tok1 (no S_OUT stream); TB readback via
// tok1_readback_* after done. data_o/data_o_valid tied off in backbone_top.
//
// SRAM macros (port mux in child):
//   Sram_tok1  u_sram_tok1    inter-block / norm1 / backbone norm
//   Sram_tok2  u_sram_tok2    transformer_block x_buf
//   Sram_q     u_sram_q       care q + tmp-on-q
//   Sram_k/v/qkm             care_attention
// =============================================================================

`include "inv_sqrt_lut_seed.v"
`include "inv_sqrt_nr.v"
`include "recip_lut_seed.v"
`include "recip_nr.v"
`include "linear.v"
`include "linear_wide.v"
`include "residual.v"
`include "layer_norm.v"
`include "care_attention.v"
`include "mlp.v"
`include "transformer_block.v"
`include "backbone_top.v"


module sglatrack_top #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320,
    parameter DATA_W    = 16
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    start,
    output wire                    busy,
    output wire                    x_ready,
    output wire                    done,
    input  wire [3:0]              sel_block_i,
    input  wire signed [DATA_W-1:0] data_in,
    input  wire                    data_valid,
    output wire signed [DATA_W-1:0] data_o,
    output wire                    data_o_valid,
    // TB readback Sram_tok1 after done (sim only; tie readback=0 in synthesis)
    input  wire                    tok1_readback,
    input  wire [13:0]             tok1_readback_addr,
    output wire [DATA_W-1:0]       tok1_readback_q
);

// ---- backbone_top Sram_tok1 (inter-block + backbone norm) ----
wire              sram_tok1_ceb_bb;
wire              sram_tok1_web_bb;
wire [13:0]       sram_tok1_addr_bb;
wire [DATA_W-1:0] sram_tok1_din_bb;
wire [DATA_W-1:0] sram_tok1_q;

// ---- transformer_block Sram_tok2 (x_buf) ----
wire              sram_tok2_ceb;
wire              sram_tok2_web;
wire [13:0]       sram_tok2_addr;
wire [DATA_W-1:0] sram_tok2_din;
wire [DATA_W-1:0] sram_tok2_q;

// ---- care_attention + tmp-on-q ----
wire              sram_q_ceb;
wire              sram_q_web;
wire [13:0]       sram_q_addr;
wire [DATA_W-1:0] sram_q_din;
wire [DATA_W-1:0] sram_q_q;

wire              sram_k_ceb;
wire              sram_k_web;
wire [13:0]       sram_k_addr;
wire [DATA_W-1:0] sram_k_din;
wire [DATA_W-1:0] sram_k_q;

wire              sram_v_ceb;
wire              sram_v_web;
wire [13:0]       sram_v_addr;
wire [DATA_W-1:0] sram_v_din;
wire [DATA_W-1:0] sram_v_q;

wire              sram_qkm_ceb;
wire              sram_qkm_web;
wire [13:0]       sram_qkm_addr;
wire [DATA_W-1:0] sram_qkm_din;
wire [DATA_W-1:0] sram_qkm_q;

// Macro pins: no skew between A and CEB/WEB/D (see verilog_rule.mdc SS8)
wire              sram_tok1_ceb_mac;
wire              sram_tok1_web_mac;
wire [13:0]       sram_tok1_addr_mac;
wire [DATA_W-1:0] sram_tok1_din_mac;

wire              sram_tok2_ceb_mac;
wire              sram_tok2_web_mac;
wire [13:0]       sram_tok2_addr_mac;
wire [DATA_W-1:0] sram_tok2_din_mac;

wire              sram_q_ceb_mac;
wire              sram_q_web_mac;
wire [13:0]       sram_q_addr_mac;
wire [DATA_W-1:0] sram_q_din_mac;

wire              sram_k_ceb_mac;
wire              sram_k_web_mac;
wire [13:0]       sram_k_addr_mac;
wire [DATA_W-1:0] sram_k_din_mac;

wire              sram_v_ceb_mac;
wire              sram_v_web_mac;
wire [13:0]       sram_v_addr_mac;
wire [DATA_W-1:0] sram_v_din_mac;

wire              sram_qkm_ceb_mac;
wire              sram_qkm_web_mac;
wire [13:0]       sram_qkm_addr_mac;
wire [DATA_W-1:0] sram_qkm_din_mac;

// TB readback mux on Sram_tok1 (priority when tok1_readback=1; backbone must be done)
assign sram_tok1_ceb_mac  = tok1_readback ? 1'b0               : sram_tok1_ceb_bb;
assign sram_tok1_web_mac  = tok1_readback ? 1'b1               : sram_tok1_web_bb;
assign sram_tok1_addr_mac = tok1_readback ? tok1_readback_addr : sram_tok1_addr_bb;
assign sram_tok1_din_mac  = tok1_readback ? {DATA_W{1'b0}}     : sram_tok1_din_bb;
assign tok1_readback_q    = sram_tok1_q;

assign sram_tok2_ceb_mac       = sram_tok2_ceb;
assign sram_tok2_web_mac       = sram_tok2_web;
assign sram_tok2_addr_mac      = sram_tok2_addr;
assign sram_tok2_din_mac       = sram_tok2_din;

assign sram_q_ceb_mac       = sram_q_ceb;
assign sram_q_web_mac       = sram_q_web;
assign sram_q_addr_mac      = sram_q_addr;
assign sram_q_din_mac       = sram_q_din;

assign sram_k_ceb_mac       = sram_k_ceb;
assign sram_k_web_mac       = sram_k_web;
assign sram_k_addr_mac      = sram_k_addr;
assign sram_k_din_mac       = sram_k_din;

assign sram_v_ceb_mac       = sram_v_ceb;
assign sram_v_web_mac       = sram_v_web;
assign sram_v_addr_mac      = sram_v_addr;
assign sram_v_din_mac       = sram_v_din;

assign sram_qkm_ceb_mac     = sram_qkm_ceb;
assign sram_qkm_web_mac     = sram_qkm_web;
assign sram_qkm_addr_mac    = sram_qkm_addr;
assign sram_qkm_din_mac     = sram_qkm_din;

backbone_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_backbone (
    .clk            (clk),
    .reset          (reset),
    .start          (start),
    .sel_block_i    (sel_block_i),
    .x_i            (data_in),
    .x_valid        (data_valid),
    .busy           (busy),
    .x_ready        (x_ready),
    .done           (done),
    .y_o            (data_o),
    .y_valid        (data_o_valid),
    .sram_tok1_ceb_o    (sram_tok1_ceb_bb),
    .sram_tok1_web_o    (sram_tok1_web_bb),
    .sram_tok1_addr_o   (sram_tok1_addr_bb),
    .sram_tok1_din_o    (sram_tok1_din_bb),
    .sram_tok1_q_i      (sram_tok1_q),
    .sram_tok2_ceb_o   (sram_tok2_ceb),
    .sram_tok2_web_o   (sram_tok2_web),
    .sram_tok2_addr_o  (sram_tok2_addr),
    .sram_tok2_din_o   (sram_tok2_din),
    .sram_tok2_q_i     (sram_tok2_q),
    .sram_q_ceb_o       (sram_q_ceb),
    .sram_q_web_o       (sram_q_web),
    .sram_q_addr_o      (sram_q_addr),
    .sram_q_din_o       (sram_q_din),
    .sram_q_q_i         (sram_q_q),
    .sram_k_ceb_o       (sram_k_ceb),
    .sram_k_web_o       (sram_k_web),
    .sram_k_addr_o      (sram_k_addr),
    .sram_k_din_o       (sram_k_din),
    .sram_k_q_i         (sram_k_q),
    .sram_v_ceb_o       (sram_v_ceb),
    .sram_v_web_o       (sram_v_web),
    .sram_v_addr_o      (sram_v_addr),
    .sram_v_din_o       (sram_v_din),
    .sram_v_q_i         (sram_v_q),
    .sram_qkm_ceb_o     (sram_qkm_ceb),
    .sram_qkm_web_o     (sram_qkm_web),
    .sram_qkm_addr_o    (sram_qkm_addr),
    .sram_qkm_din_o     (sram_qkm_din),
    .sram_qkm_q_i       (sram_qkm_q)
);

Sram_tok1 u_sram_tok1 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tok1_ceb_mac),
    .WEB   (sram_tok1_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tok1_addr_mac),
    .D     (sram_tok1_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tok1_q)
);

Sram_tok2 u_sram_tok2 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tok2_ceb_mac),
    .WEB   (sram_tok2_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tok2_addr_mac),
    .D     (sram_tok2_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tok2_q)
);

Sram_q u_sram_q (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_q_ceb_mac),
    .WEB   (sram_q_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_q_addr_mac),
    .D     (sram_q_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_q_q)
);

Sram_k u_sram_k (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_k_ceb_mac),
    .WEB   (sram_k_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_k_addr_mac),
    .D     (sram_k_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_k_q)
);

Sram_v u_sram_v (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_v_ceb_mac),
    .WEB   (sram_v_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_v_addr_mac),
    .D     (sram_v_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_v_q)
);

Sram_qkm u_sram_qkm (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_qkm_ceb_mac),
    .WEB   (sram_qkm_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_qkm_addr_mac[10:0]),   // macro 11b; care pads {3'b0,s6_addr}
    .D     (sram_qkm_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_qkm_q)
);

endmodule
