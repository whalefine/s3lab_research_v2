// =============================================================================
// sglatrack_top.v -- verilog_head2 wrapper (head-only, Plan B)
//
// Plan A: TB preloads backbone into Sram_tok1; conv1 reads tok1 directly (no S_FILL).
// SRAM macros: port mux in head_top; macro pins direct connect (no skew A/CEB/WEB/D)
//
//   Sram_v    -> x_buf
//   Sram_q/k  -> sh1 lo/hi
//   Sram_tok2 -> sh2
//   Sram_tok1 -> backbone norm preload (TB); conv1 reads activations directly
//   Sram_qkm  -> cal_bbox size/off
//
// tok1_preload=1: TB drives Sram_tok1 write (simulation preload only; tie 0 in synthesis).
//
// Sources below: `include` dependency order (leaf -> head_top).
// Paths under memory/ are relative to VCS/Genus compile root (same as TEST_head.v).
// =============================================================================

// ---- verilog_head2 RTL (this directory) ----
`include "sigmoid_lut.v"
`include "cal_bbox.v"
`include "conv.v"
`include "tail.v"
`include "head_top.v"

module sglatrack_top #(
    parameter DATA_W   = 16,
    parameter IN_CH    = 32,
    parameter C_SH1    = 96,
    parameter C_SH2    = 48,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter N_TOKENS = 320,
    parameter LENS_Z   = 64
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    start,
    output wire                    busy,
    output wire                    done,
    // TB preload into Sram_tok1 (backbone norm, token-major); sim only
    input  wire                    tok1_preload,
    input  wire [13:0]             tok1_preload_addr,
    input  wire [DATA_W-1:0]       tok1_preload_din,
    output wire [DATA_W-1:0]       cx_o,
    output wire [DATA_W-1:0]       cy_o,
    output wire [DATA_W-1:0]       w_o,
    output wire [DATA_W-1:0]       h_o
);

wire              sram_x_ceb;
wire              sram_x_web;
wire [13:0]       sram_x_addr;
wire [DATA_W-1:0] sram_x_din;
wire [DATA_W-1:0] sram_x_q;

wire              sram_sh1_lo_ceb;
wire              sram_sh1_lo_web;
wire [13:0]       sram_sh1_lo_addr;
wire [DATA_W-1:0] sram_sh1_lo_din;
wire [DATA_W-1:0] sram_sh1_lo_q;

wire              sram_sh1_hi_ceb;
wire              sram_sh1_hi_web;
wire [13:0]       sram_sh1_hi_addr;
wire [DATA_W-1:0] sram_sh1_hi_din;
wire [DATA_W-1:0] sram_sh1_hi_q;

wire              sram_tok2_ceb;
wire              sram_tok2_web;
wire [13:0]       sram_tok2_addr;
wire [DATA_W-1:0] sram_tok2_din;
wire [DATA_W-1:0] sram_tok2_q;

wire              sram_tok1_ceb_hd;
wire              sram_tok1_web_hd;
wire [13:0]       sram_tok1_addr_hd;
wire [DATA_W-1:0] sram_tok1_din_hd;

wire              sram_tok1_ceb;
wire              sram_tok1_web;
wire [13:0]       sram_tok1_addr;
wire [DATA_W-1:0] sram_tok1_din;
wire [DATA_W-1:0] sram_tok1_q;

wire              sram_bbox_ceb;
wire              sram_bbox_web;
wire [10:0]       sram_bbox_addr;
wire [DATA_W-1:0] sram_bbox_din;
wire [DATA_W-1:0] sram_bbox_q;

// TB preload mux on Sram_tok1 (priority over head_top when tok1_preload=1)
assign sram_tok1_ceb  = tok1_preload ? 1'b0              : sram_tok1_ceb_hd;
assign sram_tok1_web  = tok1_preload ? 1'b0              : sram_tok1_web_hd;
assign sram_tok1_addr = tok1_preload ? tok1_preload_addr : sram_tok1_addr_hd;
assign sram_tok1_din  = tok1_preload ? tok1_preload_din  : sram_tok1_din_hd;

head_top #(
    .IN_CH    (IN_CH   ),
    .C_SH1    (C_SH1   ),
    .C_SH2    (C_SH2   ),
    .FEAT_H   (FEAT_H  ),
    .FEAT_W   (FEAT_W  ),
    .N_TOKENS (N_TOKENS),
    .LENS_Z   (LENS_Z  ),
    .DATA_W   (DATA_W  )
) u_head (
    .clk     (clk      ),
    .reset   (reset    ),
    .start   (start    ),
    .busy    (busy     ),
    .done    (done     ),
    .cx_o    (cx_o     ),
    .cy_o    (cy_o     ),
    .w_o     (w_o      ),
    .h_o     (h_o      ),
    .sram_x_ceb_o      (sram_x_ceb),
    .sram_x_web_o      (sram_x_web),
    .sram_x_addr_o     (sram_x_addr),
    .sram_x_din_o      (sram_x_din),
    .sram_x_q_i        (sram_x_q),
    .sram_sh1_lo_ceb_o (sram_sh1_lo_ceb),
    .sram_sh1_lo_web_o (sram_sh1_lo_web),
    .sram_sh1_lo_addr_o(sram_sh1_lo_addr),
    .sram_sh1_lo_din_o (sram_sh1_lo_din),
    .sram_sh1_lo_q_i   (sram_sh1_lo_q),
    .sram_sh1_hi_ceb_o (sram_sh1_hi_ceb),
    .sram_sh1_hi_web_o (sram_sh1_hi_web),
    .sram_sh1_hi_addr_o(sram_sh1_hi_addr),
    .sram_sh1_hi_din_o (sram_sh1_hi_din),
    .sram_sh1_hi_q_i   (sram_sh1_hi_q),
    .sram_tok2_ceb_o   (sram_tok2_ceb),
    .sram_tok2_web_o   (sram_tok2_web),
    .sram_tok2_addr_o  (sram_tok2_addr),
    .sram_tok2_din_o   (sram_tok2_din),
    .sram_tok2_q_i     (sram_tok2_q),
    .sram_tok1_ceb_o   (sram_tok1_ceb_hd),
    .sram_tok1_web_o   (sram_tok1_web_hd),
    .sram_tok1_addr_o  (sram_tok1_addr_hd),
    .sram_tok1_din_o   (sram_tok1_din_hd),
    .sram_tok1_q_i     (sram_tok1_q),
    .sram_bbox_ceb_o   (sram_bbox_ceb),
    .sram_bbox_web_o   (sram_bbox_web),
    .sram_bbox_addr_o  (sram_bbox_addr),
    .sram_bbox_din_o   (sram_bbox_din),
    .sram_bbox_q_i     (sram_bbox_q)
);

Sram_v u_sram_x_buf (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_x_ceb),
    .WEB   (sram_x_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_x_addr),
    .D     (sram_x_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_x_q)
);

Sram_q u_sram_sh1_lo (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_sh1_lo_ceb),
    .WEB   (sram_sh1_lo_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_sh1_lo_addr),
    .D     (sram_sh1_lo_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_sh1_lo_q)
);

Sram_k u_sram_sh1_hi (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_sh1_hi_ceb),
    .WEB   (sram_sh1_hi_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_sh1_hi_addr),
    .D     (sram_sh1_hi_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_sh1_hi_q)
);

Sram_tok2 u_sram_tok2 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tok2_ceb),
    .WEB   (sram_tok2_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tok2_addr),
    .D     (sram_tok2_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tok2_q)
);

Sram_tok1 u_sram_tok1 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tok1_ceb),
    .WEB   (sram_tok1_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tok1_addr),
    .D     (sram_tok1_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tok1_q)
);

Sram_qkm u_sram_size_off (
    .SLP    (1'b0),
    .DSLP   (1'b0),
    .SD     (1'b0),
    .PUDELAY(),
    .CLK    (~clk),
    .CEB    (sram_bbox_ceb),
    .WEB    (sram_bbox_web),
    .BIST   (1'b0),
    .CEBM   (),
    .WEBM   (),
    .A      (sram_bbox_addr),
    .D      (sram_bbox_din),
    .BWEB   (16'b0),
    .AM     (),
    .DM     (),
    .BWEBM  (16'b0),
    .RTSEL  (2'b01),
    .WTSEL  (2'b00),
    .Q      (sram_bbox_q)
);

endmodule
