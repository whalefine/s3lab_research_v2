// =============================================================================
// head_top.v -- verilog_head2 head (conv1 + conv2 + tail + cal_bbox)
// cal_bbox: size/off maps in u_bbox.u_sram_size_off (Sram_qkm), not head_top SRAM pool
// -----------------------------------------------------------------------------
// numpy: head_shared_trunk() + cal_bbox() in run_backbone_numpy_shared_trunk.py
// Input: backbone token stream -> x_buf (Sram_tok1 8192x16) ; Golden: bbox + activations
// Weights in conv.v / tail.v ROM (memory/ at compile)
//
// x_buf (search NCHW in S_FILL, conv1 in S_CONV1): Sram_tok1 12288x16 macro, use depth 8192
//   write: S_FILL fill_search -> fill_dst, a_i (posedge port reg)
//   read:  S_CONV1 conv MAC phase0 issue read; phase1 comb x_q -> c1_x_i_mac
//
// sh1_buf (conv1 out / conv2 in): Sram_q (lo) + Sram_k (hi), 12288x16 each (C1_LEN=24576, flat>=12288 -> hi)
//   SRAM read contract (CLK = ~clk, same as verilog_backbone2):
//     posedge T:   drive A, CEB=0, WEB=1
//     posedge T+1: Q valid for A@T; conv2 MAC phase1 uses comb sh1_rd_q (read @ phase0 posedge)
//   SRAM write: c1_y_valid same cycle comb mux (c1_wr_idx, c1_y_data) -> posedge port reg -> macro
//   Port regs: comb _n -> posedge reg -> macro (sim #1 delay on macro pins avoids $hold vs ~clk)
// Golden: Activation/box_head_shared_conv1_* (conv1 out map)
// sh2_buf (conv2 out / tail in): Sram_tok2 12288x16 (C2_LEN=12288)
// wgt_buf (conv1/conv2 weight prefetch): Sram_v 12288x16 (use addr 0..287 conv1, 0..863 conv2)
//   WPRE phase1 write w_i; MAC phase0 read mac_feat, phase1 comb wgt_q -> wgt_rd_i
// Compile (5 macros, 5 .v files — do not substitute one file for all):
//   Sram_tok1 12288 16 16 s (x_buf)
//   Sram_q / Sram_k / Sram_tok2 (sh1/sh2)
//   Sram_v 12288 16 16 s (wgt_buf)
// =============================================================================

module head_top #(
    parameter IN_CH    = 32,
    parameter C_SH1    = 96,
    parameter C_SH2    = 48,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter N_TOKENS = 320,
    parameter LENS_Z   = 64,
    parameter DATA_W   = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    input  wire signed [DATA_W-1:0] a_i,
    input  wire                     a_valid,

    output wire        busy,
    output reg         done,

    output wire [DATA_W-1:0] cx_o,
    output wire [DATA_W-1:0] cy_o,
    output wire [DATA_W-1:0] w_o,
    output wire [DATA_W-1:0] h_o
);

localparam FEAT_SZ     = FEAT_H * FEAT_W;
localparam SKIP_VALS   = LENS_Z * IN_CH;
localparam TOT_VALS    = N_TOKENS * IN_CH;
localparam TOT_VALS_M1 = TOT_VALS - 1;
localparam IN_LEN_HEAD = FEAT_SZ * IN_CH;
localparam C1_LEN      = C_SH1 * FEAT_SZ;
localparam C2_LEN      = C_SH2 * FEAT_SZ;
localparam C1_HALF     = C1_LEN >> 1;  // 12288: bank boundary (not a power-of-2 bit slice)
localparam X_BUF_AW    = 13;           // IN_LEN_HEAD=8192 -> addr 0..8191
localparam SH1_BANK_AW = 14;           // per-bank index 0..12287; Sram_q / Sram_k .A width
localparam SH2_AW      = 14;           // sh2 flat 0..C2_LEN-1 (12288)
localparam WGT_AW     = 14;           // Sram_v depth 12288; wgt uses low 10b from conv

wire rst_n = ~reset;

parameter S_IDLE  = 3'd0;
parameter S_FILL  = 3'd1;
parameter S_CONV1 = 3'd2;
parameter S_CONV2 = 3'd3;
parameter S_TAIL  = 3'd4;
parameter S_BBOX  = 3'd5;
parameter S_DONE  = 3'd6;

reg [2:0] CS, NS;

reg [13:0] fill_cnt;
wire [13:0] fill_off    = fill_cnt - SKIP_VALS;
wire [7:0]  fill_n      = fill_off[12:5];
wire [4:0]  fill_c      = fill_off[4:0];
wire        fill_search = (fill_cnt >= SKIP_VALS) && (fill_cnt < TOT_VALS);
wire [13:0] fill_dst    = {fill_c, fill_n};

reg [DATA_W-1:0] bbox_reg [0:3];

reg c1_start, c2_start, t_start, b_start;
reg c1_started, c2_started, t_started, b_started;

wire              c1_busy, c1_done, c1_y_valid;
wire [DATA_W-1:0] c1_y_data;
wire [13:0]       c1_x_addr;
wire [13:0]       c1_x_addr_mac;
wire              c1_mac_phase;
wire [DATA_W-1:0] c1_x_i_mac;

wire              c2_busy, c2_done, c2_y_valid;
wire [DATA_W-1:0] c2_y_data;
wire [14:0]       c2_x_addr;
wire [14:0]       c2_x_addr_mac;
wire              c2_mac_phase;
wire [DATA_W-1:0] sh1_rd_q;
wire [DATA_W-1:0] c2_x_i_mac;

wire              c1_wgt_wr_en;
wire [9:0]        c1_wgt_wr_addr;
wire [DATA_W-1:0] c1_wgt_wr_data;
wire              c1_wgt_rd_req;
wire [9:0]        c1_wgt_rd_addr;
wire              c2_wgt_wr_en;
wire [9:0]        c2_wgt_wr_addr;
wire [DATA_W-1:0] c2_wgt_wr_data;
wire              c2_wgt_rd_req;
wire [9:0]        c2_wgt_rd_addr;
wire [DATA_W-1:0] wgt_q;

reg [31:0] c1_wr_idx;
reg [31:0] c2_wr_idx;

// x_buf SRAM (S_FILL write -> conv1 read); Golden: backbone search tokens in x_buf order
reg        x_ceb;
reg        x_web;
reg [X_BUF_AW-1:0] x_addr;
reg [DATA_W-1:0] x_din;
wire [DATA_W-1:0] x_q;

reg        x_ceb_n;
reg        x_web_n;
reg [X_BUF_AW-1:0] x_addr_n;
reg [DATA_W-1:0] x_din_n;

wire [X_BUF_AW-1:0] x_wr_addr   = fill_dst[X_BUF_AW-1:0];
wire [X_BUF_AW-1:0] x_rd_addr   = c1_x_addr_mac[X_BUF_AW-1:0];
wire        x_fill_wr    = (CS == S_FILL) && a_valid && fill_search;
wire        x_c1_rd_req  = (CS == S_CONV1) && c1_busy && !c1_mac_phase;

wire        x_ceb_mac;
wire        x_web_mac;
wire [X_BUF_AW-1:0] x_addr_mac;
wire [SH1_BANK_AW-1:0] x_addr_mac_p;  // Sram_tok1 .A is 14b (12288 depth); x_buf uses 8192
wire [DATA_W-1:0] x_din_mac;

`ifdef SYNTHESIS
assign x_ceb_mac  = x_ceb;
assign x_web_mac  = x_web;
assign x_addr_mac = x_addr;
assign x_din_mac  = x_din;
`else
assign #1 x_ceb_mac  = x_ceb;
assign #1 x_web_mac  = x_web;
assign x_addr_mac    = x_addr;
assign #1 x_din_mac  = x_din;
`endif

assign x_addr_mac_p = {{(SH1_BANK_AW-X_BUF_AW){1'b0}}, x_addr_mac};

Sram_tok1 u_sram_x_buf (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (x_ceb_mac),
    .WEB   (x_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (x_addr_mac_p),
    .D     (x_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (x_q)
);

// sh1 SRAM (conv1 capture -> conv2 read); flat[14:0] -> bank + 14b local (split at C1_HALF)
// Macro ports are posedge-registered (_n comb -> reg) so A is stable at macro negedge sample.
reg        sh1_lo_ceb;
reg        sh1_lo_web;
reg [SH1_BANK_AW-1:0] sh1_lo_addr;
reg [DATA_W-1:0]      sh1_lo_din;
wire [DATA_W-1:0]     sh1_lo_q;

reg        sh1_hi_ceb;
reg        sh1_hi_web;
reg [SH1_BANK_AW-1:0] sh1_hi_addr;
reg [DATA_W-1:0]      sh1_hi_din;
wire [DATA_W-1:0]     sh1_hi_q;

reg        sh1_lo_ceb_n;
reg        sh1_lo_web_n;
reg [SH1_BANK_AW-1:0] sh1_lo_addr_n;
reg [DATA_W-1:0]      sh1_lo_din_n;

reg        sh1_hi_ceb_n;
reg        sh1_hi_web_n;
reg [SH1_BANK_AW-1:0] sh1_hi_addr_n;
reg [DATA_W-1:0]      sh1_hi_din_n;

wire [14:0]       sh1_wr_flat15  = c1_wr_idx[14:0];
wire [14:0]       sh1_rd_flat15 = c2_x_addr_mac[14:0];
wire              sh1_wr_bank    = (sh1_wr_flat15 >= C1_HALF);
wire              sh1_rd_bank    = (sh1_rd_flat15 >= C1_HALF);
wire [SH1_BANK_AW-1:0] sh1_wr_laddr = sh1_wr_bank ?
    (sh1_wr_flat15 - C1_HALF) : sh1_wr_flat15[SH1_BANK_AW-1:0];
wire [SH1_BANK_AW-1:0] sh1_rd_laddr = sh1_rd_bank ?
    (sh1_rd_flat15 - C1_HALF) : sh1_rd_flat15[SH1_BANK_AW-1:0];
// Issue SRAM read on conv MAC phase0 (x_addr_mac_rd = x_addr_nxt; Q valid next posedge)
wire        sh1_c2_rd_req  = (CS == S_CONV2) && c2_busy && !c2_mac_phase;

// conv1 capture: same-cycle write as legacy sh1_buf[c1_wr_idx]<=c1_y_data (posedge port reg)

// Macro-facing wires (posedge-reg outputs; optional sim delay for TSMC $hold on A vs CLK)
wire        sh1_lo_ceb_mac;
wire        sh1_lo_web_mac;
wire [SH1_BANK_AW-1:0] sh1_lo_addr_mac;
wire [DATA_W-1:0]      sh1_lo_din_mac;
wire        sh1_hi_ceb_mac;
wire        sh1_hi_web_mac;
wire [SH1_BANK_AW-1:0] sh1_hi_addr_mac;
wire [DATA_W-1:0]      sh1_hi_din_mac;

`ifdef SYNTHESIS
assign sh1_lo_ceb_mac  = sh1_lo_ceb;
assign sh1_lo_web_mac  = sh1_lo_web;
assign sh1_lo_addr_mac = sh1_lo_addr;
assign sh1_lo_din_mac  = sh1_lo_din;
assign sh1_hi_ceb_mac  = sh1_hi_ceb;
assign sh1_hi_web_mac  = sh1_hi_web;
assign sh1_hi_addr_mac = sh1_hi_addr;
assign sh1_hi_din_mac  = sh1_hi_din;
`else
assign #1 sh1_lo_ceb_mac  = sh1_lo_ceb;
assign #1 sh1_lo_web_mac  = sh1_lo_web;
assign sh1_lo_addr_mac    = sh1_lo_addr;
assign #1 sh1_lo_din_mac  = sh1_lo_din;
assign #1 sh1_hi_ceb_mac  = sh1_hi_ceb;
assign #1 sh1_hi_web_mac  = sh1_hi_web;
assign sh1_hi_addr_mac    = sh1_hi_addr;
assign #1 sh1_hi_din_mac  = sh1_hi_din;
`endif

Sram_q u_sram_sh1_lo (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sh1_lo_ceb_mac),
    .WEB   (sh1_lo_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sh1_lo_addr_mac),
    .D     (sh1_lo_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sh1_lo_q)
);

Sram_k u_sram_sh1_hi (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sh1_hi_ceb_mac),
    .WEB   (sh1_hi_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sh1_hi_addr_mac),
    .D     (sh1_hi_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sh1_hi_q)
);

wire              t_busy, t_done;
wire [14:0]       t_x_addr;
wire [14:0]       t_x_addr_mac;
wire              t_mac_phase;
wire [DATA_W-1:0] t_x_i_mac;

// sh2 SRAM (conv2 capture -> tail read); flat[13:0] = c2_wr_idx / t_x_addr (C2_LEN=12288)
reg        sh2_ceb;
reg        sh2_web;
reg [SH2_AW-1:0] sh2_addr;
reg [DATA_W-1:0] sh2_din;
wire [DATA_W-1:0] sh2_q;

reg        sh2_ceb_n;
reg        sh2_web_n;
reg [SH2_AW-1:0] sh2_addr_n;
reg [DATA_W-1:0] sh2_din_n;

wire [SH2_AW-1:0] sh2_wr_addr = c2_wr_idx[SH2_AW-1:0];
wire [SH2_AW-1:0] sh2_rd_addr = t_x_addr_mac[SH2_AW-1:0];
wire        sh2_tail_rd_req  = (CS == S_TAIL) && t_busy && !t_mac_phase;

wire        sh2_ceb_mac;
wire        sh2_web_mac;
wire [SH2_AW-1:0] sh2_addr_mac;
wire [DATA_W-1:0] sh2_din_mac;

`ifdef SYNTHESIS
assign sh2_ceb_mac  = sh2_ceb;
assign sh2_web_mac  = sh2_web;
assign sh2_addr_mac = sh2_addr;
assign sh2_din_mac  = sh2_din;
`else
assign #1 sh2_ceb_mac  = sh2_ceb;
assign #1 sh2_web_mac  = sh2_web;
assign sh2_addr_mac    = sh2_addr;
assign #1 sh2_din_mac  = sh2_din;
`endif

Sram_tok2 u_sram_sh2 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sh2_ceb_mac),
    .WEB   (sh2_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sh2_addr_mac),
    .D     (sh2_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sh2_q)
);

// wgt_buf SRAM (Sram_v; shared conv1/conv2 — S_CONV1 vs S_CONV2 mutually exclusive)
reg        wgt_ceb;
reg        wgt_web;
reg [WGT_AW-1:0] wgt_addr;
reg [DATA_W-1:0] wgt_din;

reg        wgt_ceb_n;
reg        wgt_web_n;
reg [WGT_AW-1:0] wgt_addr_n;
reg [DATA_W-1:0] wgt_din_n;

wire [WGT_AW-1:0] wgt_wr_addr = c1_wgt_wr_en ? {{4{1'b0}}, c1_wgt_wr_addr} :
                                  {{4{1'b0}}, c2_wgt_wr_addr};
wire [WGT_AW-1:0] wgt_rd_addr = c1_wgt_rd_req ? {{4{1'b0}}, c1_wgt_rd_addr} :
                                  {{4{1'b0}}, c2_wgt_rd_addr};
wire [DATA_W-1:0] wgt_wr_data = c1_wgt_wr_en ? c1_wgt_wr_data : c2_wgt_wr_data;

wire        wgt_ceb_mac;
wire        wgt_web_mac;
wire [WGT_AW-1:0] wgt_addr_mac;
wire [DATA_W-1:0] wgt_din_mac;

`ifdef SYNTHESIS
assign wgt_ceb_mac  = wgt_ceb;
assign wgt_web_mac  = wgt_web;
assign wgt_addr_mac = wgt_addr;
assign wgt_din_mac  = wgt_din;
`else
assign #1 wgt_ceb_mac  = wgt_ceb;
assign #1 wgt_web_mac  = wgt_web;
assign wgt_addr_mac    = wgt_addr;
assign #1 wgt_din_mac  = wgt_din;
`endif

Sram_v u_sram_wgt (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (wgt_ceb_mac),
    .WEB   (wgt_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (wgt_addr_mac),
    .D     (wgt_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (wgt_q)
);

always @(*) begin
    wgt_ceb_n  = 1'b1;
    wgt_web_n  = 1'b1;
    wgt_addr_n = {WGT_AW{1'b0}};
    wgt_din_n  = {DATA_W{1'b0}};

    if (c1_wgt_wr_en || c2_wgt_wr_en) begin
        wgt_ceb_n  = 1'b0;
        wgt_web_n  = 1'b0;
        wgt_addr_n = wgt_wr_addr;
        wgt_din_n  = wgt_wr_data;
    end else if (c1_wgt_rd_req || c2_wgt_rd_req) begin
        wgt_ceb_n  = 1'b0;
        wgt_web_n  = 1'b1;
        wgt_addr_n = wgt_rd_addr;
    end
end

always @(posedge clk) begin
    if (reset) begin
        wgt_ceb  <= 1'b1;
        wgt_web  <= 1'b1;
        wgt_addr <= {WGT_AW{1'b0}};
        wgt_din  <= {DATA_W{1'b0}};
    end else begin
        wgt_ceb  <= wgt_ceb_n;
        wgt_web  <= wgt_web_n;
        wgt_addr <= wgt_addr_n;
        wgt_din  <= wgt_din_n;
    end
end

wire              tc_sig_v, to_v, ts_sig_v;
wire [DATA_W-1:0] tc_sig_d, to_d, ts_sig_d;

wire              b_busy, b_done, b_valid;
wire [DATA_W-1:0] b_data;
wire [1:0]        b_idx;

// conv1 input: phase0 read x_addr_nxt; phase1 MAC uses comb x_q (same as sh1/c2/sh2)
assign c1_x_i_mac = ((CS == S_CONV1) && c1_busy && c1_mac_phase) ? x_q : {DATA_W{1'b0}};

// conv2 input: verilog2 negedge c2_x_i_q<=sh1_buf[c2_x_addr] (reg, 0 latency).
// SRAM 1-cycle: phase0 posedge read x_addr_nxt; phase1 posedge MAC uses comb sh1_rd_q (Q@T+1).
// Do not negedge-latch Q into x_i (lags 1 MAC vs comb sh1_rd_q at phase1).
assign sh1_rd_q = sh1_rd_bank ? sh1_hi_q : sh1_lo_q;
assign c2_x_i_mac = ((CS == S_CONV2) && c2_busy && c2_mac_phase) ? sh1_rd_q : {DATA_W{1'b0}};

// tail input: phase0 read x_addr_nxt; phase1 MAC uses comb sh2_q (same contract as sh1/c2)
assign t_x_i_mac = ((CS == S_TAIL) && t_busy && t_mac_phase) ? sh2_q : {DATA_W{1'b0}};

// x_buf SRAM port mux (S_FILL write vs conv1 read; mutually exclusive by FSM)
always @(*) begin
    x_ceb_n  = 1'b1;
    x_web_n  = 1'b1;
    x_addr_n = {X_BUF_AW{1'b0}};
    x_din_n  = {DATA_W{1'b0}};

    if (x_fill_wr) begin
        x_ceb_n  = 1'b0;
        x_web_n  = 1'b0;
        x_addr_n = x_wr_addr;
        x_din_n  = a_i[DATA_W-1:0];
    end else if (x_c1_rd_req) begin
        x_ceb_n  = 1'b0;
        x_web_n  = 1'b1;
        x_addr_n = x_rd_addr;
    end
end

always @(posedge clk) begin
    if (reset) begin
        x_ceb  <= 1'b1;
        x_web  <= 1'b1;
        x_addr <= {X_BUF_AW{1'b0}};
        x_din  <= {DATA_W{1'b0}};
    end else begin
        x_ceb  <= x_ceb_n;
        x_web  <= x_web_n;
        x_addr <= x_addr_n;
        x_din  <= x_din_n;
    end
end

// sh2 SRAM port mux (conv2 write vs tail read; mutually exclusive by FSM)
always @(*) begin
    sh2_ceb_n  = 1'b1;
    sh2_web_n  = 1'b1;
    sh2_addr_n = {SH2_AW{1'b0}};
    sh2_din_n  = {DATA_W{1'b0}};

    if (c2_y_valid) begin
        sh2_ceb_n  = 1'b0;
        sh2_web_n  = 1'b0;
        sh2_addr_n = sh2_wr_addr;
        sh2_din_n  = c2_y_data;
    end else if (sh2_tail_rd_req) begin
        sh2_ceb_n  = 1'b0;
        sh2_web_n  = 1'b1;
        sh2_addr_n = sh2_rd_addr;
    end
end

always @(posedge clk) begin
    if (reset) begin
        sh2_ceb  <= 1'b1;
        sh2_web  <= 1'b1;
        sh2_addr <= {SH2_AW{1'b0}};
        sh2_din  <= {DATA_W{1'b0}};
    end else begin
        sh2_ceb  <= sh2_ceb_n;
        sh2_web  <= sh2_web_n;
        sh2_addr <= sh2_addr_n;
        sh2_din  <= sh2_din_n;
    end
end

// sh1 SRAM port mux next-state (conv1 write vs conv2 read; mutually exclusive by FSM)
always @(*) begin
    sh1_lo_ceb_n  = 1'b1;
    sh1_lo_web_n  = 1'b1;
    sh1_lo_addr_n = {SH1_BANK_AW{1'b0}};
    sh1_lo_din_n  = {DATA_W{1'b0}};
    sh1_hi_ceb_n  = 1'b1;
    sh1_hi_web_n  = 1'b1;
    sh1_hi_addr_n = {SH1_BANK_AW{1'b0}};
    sh1_hi_din_n  = {DATA_W{1'b0}};

    if (c1_y_valid) begin
        if (!sh1_wr_bank) begin
            sh1_lo_ceb_n  = 1'b0;
            sh1_lo_web_n  = 1'b0;
            sh1_lo_addr_n = sh1_wr_laddr;
            sh1_lo_din_n  = c1_y_data;
        end else begin
            sh1_hi_ceb_n  = 1'b0;
            sh1_hi_web_n  = 1'b0;
            sh1_hi_addr_n = sh1_wr_laddr;
            sh1_hi_din_n  = c1_y_data;
        end
    end else if (sh1_c2_rd_req) begin
        if (!sh1_rd_bank) begin
            sh1_lo_ceb_n  = 1'b0;
            sh1_lo_web_n  = 1'b1;
            sh1_lo_addr_n = sh1_rd_laddr;
        end else begin
            sh1_hi_ceb_n  = 1'b0;
            sh1_hi_web_n  = 1'b1;
            sh1_hi_addr_n = sh1_rd_laddr;
        end
    end
end

// Latch SRAM ports at posedge so A/CEB/WEB/D are stable before macro CLK (~clk) negedge
always @(posedge clk) begin
    if (reset) begin
        sh1_lo_ceb  <= 1'b1;
        sh1_lo_web  <= 1'b1;
        sh1_lo_addr <= {SH1_BANK_AW{1'b0}};
        sh1_lo_din  <= {DATA_W{1'b0}};
        sh1_hi_ceb  <= 1'b1;
        sh1_hi_web  <= 1'b1;
        sh1_hi_addr <= {SH1_BANK_AW{1'b0}};
        sh1_hi_din  <= {DATA_W{1'b0}};
    end else begin
        sh1_lo_ceb  <= sh1_lo_ceb_n;
        sh1_lo_web  <= sh1_lo_web_n;
        sh1_lo_addr <= sh1_lo_addr_n;
        sh1_lo_din  <= sh1_lo_din_n;
        sh1_hi_ceb  <= sh1_hi_ceb_n;
        sh1_hi_web  <= sh1_hi_web_n;
        sh1_hi_addr <= sh1_hi_addr_n;
        sh1_hi_din  <= sh1_hi_din_n;
    end
end

conv #(
    .IN_CH       (IN_CH   ),
    .OUT_CH      (C_SH1   ),
    .IN_H        (FEAT_H  ),
    .IN_W        (FEAT_W  ),
    .K           (3       ),
    .PAD         (1       ),
    .HAS_RELU    (1       ),
    .DATA_W      (DATA_W  ),
    .FRAC_W      (8       ),
    .ACC_W       (32      ),
    .ROM_PROFILE (1       ),
    .X_AW        (14      )
) u_conv1 (
    .clk     (clk       ),
    .rst_n   (rst_n     ),
    .start   (c1_start  ),
    .busy    (c1_busy   ),
    .done    (c1_done   ),
    .x_addr  (c1_x_addr     ),
    .x_i     (c1_x_i_mac    ),
    .y_valid (c1_y_valid    ),
    .y_data  (c1_y_data     ),
    .y_oc    (               ),
    .y_oh    (               ),
    .y_ow    (               ),
    .mac_phase_o   (c1_mac_phase  ),
    .x_addr_mac_rd (c1_x_addr_mac ),
    .wgt_wr_en     (c1_wgt_wr_en  ),
    .wgt_wr_addr   (c1_wgt_wr_addr),
    .wgt_wr_data   (c1_wgt_wr_data),
    .wgt_rd_req    (c1_wgt_rd_req ),
    .wgt_rd_addr   (c1_wgt_rd_addr),
    .wgt_rd_i      (wgt_q         )
);

conv #(
    .IN_CH       (C_SH1   ),
    .OUT_CH      (C_SH2   ),
    .IN_H        (FEAT_H  ),
    .IN_W        (FEAT_W  ),
    .K           (3       ),
    .PAD         (1       ),
    .HAS_RELU    (1       ),
    .DATA_W      (DATA_W  ),
    .FRAC_W      (8       ),
    .ACC_W       (32      ),
    .ROM_PROFILE (2       ),
    .X_AW        (15      )
) u_conv2 (
    .clk     (clk       ),
    .rst_n   (rst_n     ),
    .start   (c2_start  ),
    .busy    (c2_busy   ),
    .done    (c2_done   ),
    .x_addr  (c2_x_addr ),
    .x_i     (c2_x_i_mac),
    .y_valid (c2_y_valid),
    .y_data  (c2_y_data ),
    .y_oc    (           ),
    .y_oh    (           ),
    .y_ow    (           ),
    .mac_phase_o   (c2_mac_phase  ),
    .x_addr_mac_rd (c2_x_addr_mac ),
    .wgt_wr_en     (c2_wgt_wr_en  ),
    .wgt_wr_addr   (c2_wgt_wr_addr),
    .wgt_wr_data   (c2_wgt_wr_data),
    .wgt_rd_req    (c2_wgt_rd_req ),
    .wgt_rd_addr   (c2_wgt_rd_addr),
    .wgt_rd_i      (wgt_q         )
);

tail #(
    .DATA_W (DATA_W),
    .X_AW   (15    )
) u_tail (
    .clk              (clk      ),
    .rst_n            (rst_n    ),
    .start            (t_start  ),
    .busy             (t_busy   ),
    .done             (t_done   ),
    .x_addr           (t_x_addr ),
    .x_i              (t_x_i_mac),
    .mac_phase_o      (t_mac_phase  ),
    .x_addr_mac_rd    (t_x_addr_mac),
    .ctr_raw_y_valid  (         ),
    .ctr_raw_y_data   (         ),
    .ctr_raw_y_oh     (         ),
    .ctr_raw_y_ow     (         ),
    .ctr_y_valid      (tc_sig_v ),
    .ctr_y_data       (tc_sig_d ),
    .off_y_valid      (to_v     ),
    .off_y_data       (to_d     ),
    .off_y_sub        (         ),
    .off_y_oh         (         ),
    .off_y_ow         (         ),
    .size_raw_y_valid (         ),
    .size_raw_y_data  (         ),
    .size_raw_y_sub   (         ),
    .size_raw_y_oh    (         ),
    .size_raw_y_ow    (         ),
    .size_y_valid     (ts_sig_v ),
    .size_y_data      (ts_sig_d )
);

cal_bbox #(.DATA_W(DATA_W)) u_bbox (
    .clk           (clk     ),
    .rst_n         (rst_n   ),
    .start         (b_start ),
    .busy          (b_busy  ),
    .done          (b_done  ),
    .ctr_in_valid  (tc_sig_v),
    .ctr_in_data   (tc_sig_d),
    .size_in_valid (ts_sig_v),
    .size_in_data  (ts_sig_d),
    .size_in_sub   (1'b0    ),
    .off_in_valid  (to_v    ),
    .off_in_data   (to_d    ),
    .off_in_sub    (1'b0    ),
    .bbox_valid    (b_valid ),
    .bbox_data     (b_data  ),
    .bbox_idx      (b_idx   )
);

// FSM CS
always @(posedge clk) begin
    if (reset)
        CS <= S_IDLE;
    else
        CS <= NS;
end

// FSM NS
always @(*) begin
    NS = CS;
    case (CS)
        S_IDLE:  NS = start ? S_FILL : S_IDLE;
        S_FILL:  NS = (a_valid && (fill_cnt == TOT_VALS_M1)) ? S_CONV1 : S_FILL;
        S_CONV1: NS = c1_done ? S_CONV2 : S_CONV1;
        S_CONV2: NS = c2_done ? S_TAIL : S_CONV2;
        S_TAIL:  NS = t_done ? S_BBOX : S_TAIL;
        S_BBOX:  NS = b_done ? S_DONE : S_BBOX;
        S_DONE:  NS = S_IDLE;
        default: NS = S_IDLE;
    endcase
end

// fill_cnt (x_buf write via u_sram_x_buf comb mux when x_fill_wr)
always @(posedge clk) begin
    if (reset)
        fill_cnt <= 14'd0;
    else if (CS == S_IDLE)
        fill_cnt <= 14'd0;
    else if (CS == S_FILL && a_valid && (fill_cnt < TOT_VALS))
        fill_cnt <= fill_cnt + 14'd1;
end

// done, submodule starts, started flags, bbox_reg
always @(posedge clk) begin
    done     <= 1'b0;
    c1_start <= 1'b0;
    c2_start <= 1'b0;
    t_start  <= 1'b0;
    b_start  <= 1'b0;

    if (reset) begin
        c1_started <= 1'b0;
        c2_started <= 1'b0;
        t_started  <= 1'b0;
        b_started  <= 1'b0;
        c1_wr_idx  <= 32'd0;
        c2_wr_idx  <= 32'd0;
    end else begin
        case (CS)
            S_IDLE: begin
                c1_started <= 1'b0;
                c2_started <= 1'b0;
                t_started  <= 1'b0;
                b_started  <= 1'b0;
            end

            S_CONV1: begin
                if (!c1_started) begin
                    c1_start   <= 1'b1;
                    c1_started <= 1'b1;
                    c1_wr_idx  <= 32'd0;
                end
            end

            S_CONV2: begin
                if (!c2_started && !c2_busy) begin
                    c2_start   <= 1'b1;
                    c2_started <= 1'b1;
                    c2_wr_idx  <= 32'd0;
                end
            end

            S_TAIL: begin
                if (!t_started && !t_busy) begin
                    t_start   <= 1'b1;
                    t_started <= 1'b1;
                end
            end

            S_BBOX: begin
                if (!b_started && !b_busy) begin
                    b_start   <= 1'b1;
                    b_started <= 1'b1;
                end
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase

        if (c1_y_valid)
            c1_wr_idx <= c1_wr_idx + 32'd1;

        if (c2_y_valid)
            c2_wr_idx <= c2_wr_idx + 32'd1;

        if (b_valid)
            bbox_reg[b_idx] <= b_data;
    end
end

assign busy = (CS != S_IDLE) && (CS != S_DONE);
assign cx_o = bbox_reg[0];
assign cy_o = bbox_reg[1];
assign w_o  = bbox_reg[2];
assign h_o  = bbox_reg[3];

// ---------------------------------------------------------------------------
// `ifdef DUMP_HEAD_SH1_DEBUG -- sh1 SRAM vs box_head_shared_after_conv1_out golden
// Purpose: locate first conv1 write or conv2 read mismatch (timing / bank / addr).
// Golden: Activation/box_head_shared_after_conv1_out_bi.txt (C1_LEN=24576, row-major)
// VCS: +define+DUMP_HEAD_SH1_DEBUG +define+GOLDEN_ACT=\"./TXT_File/Activation\"
// ---------------------------------------------------------------------------
`ifdef DUMP_HEAD_SH1_DEBUG
`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

localparam SH1_DBG_WR_SAMPLES = 8;
localparam SH1_DBG_RD_SAMPLES = 8;
localparam SH1_DBG_MISM_CAP   = 16;

reg [15:0] sh1_gold [0:C1_LEN-1];
reg [31:0] sh1_wr_cnt;
reg [31:0] sh1_rd_log_cnt;
reg [31:0] sh1_wr_mism_cnt;
reg [31:0] sh1_rd_mism_cnt;
reg        sh1_dbg_c1_done_seen;
reg        sh1_dbg_c2_done_seen;

function sh1_dbg_sample_wr;
    input [31:0] cnt;
    input [13:0] flat;
    begin
        sh1_dbg_sample_wr =
            (cnt < SH1_DBG_WR_SAMPLES) ||
            (flat == 14'd0) ||
            (flat == 14'd1) ||
            (flat == 14'd12287) ||
            (flat == 14'd12288) ||
            (flat == C1_LEN - 1);
    end
endfunction

function sh1_dbg_sample_rd;
    input [31:0] cnt;
    input [13:0] flat;
    begin
        sh1_dbg_sample_rd =
            (cnt < SH1_DBG_RD_SAMPLES) ||
            (flat == 14'd0) ||
            (flat == 14'd1) ||
            (flat == 14'd12287) ||
            (flat == 14'd12288) ||
            (flat == C1_LEN - 1);
    end
endfunction

initial begin
    $readmemb({`GOLDEN_ACT, "/box_head_shared_after_conv1_out_bi.txt"}, sh1_gold);
    $display("[SH1_DBG] golden loaded C1_LEN=%0d file=%s/box_head_shared_after_conv1_out_bi.txt",
             C1_LEN, `GOLDEN_ACT);
end

// conv1 capture -> SRAM write (c1_y_valid cycle; same as legacy reg write)
always @(posedge clk) begin
    if (reset) begin
        sh1_wr_cnt           <= 32'd0;
        sh1_wr_mism_cnt      <= 32'd0;
        sh1_dbg_c1_done_seen <= 1'b0;
    end else begin
        if (c1_y_valid) begin
            if (sh1_dbg_sample_wr(sh1_wr_cnt, sh1_wr_flat15))
                $display("[SH1_WR] cycle=%0t flat=%0d bank=%0d laddr=%0h din=%04h y=%04h gold=%04h",
                         $time, sh1_wr_flat15, sh1_wr_bank, sh1_wr_laddr,
                         c1_y_data, c1_y_data, sh1_gold[sh1_wr_flat15]);
            if (c1_y_data !== sh1_gold[sh1_wr_flat15] &&
                sh1_wr_mism_cnt < SH1_DBG_MISM_CAP) begin
                $display("[SH1_WR_MISMATCH] flat=%0d rtl=%04h gold=%04h (conv1_y=%04h)",
                         sh1_wr_flat15, c1_y_data, sh1_gold[sh1_wr_flat15], c1_y_data);
                sh1_wr_mism_cnt <= sh1_wr_mism_cnt + 32'd1;
            end
            sh1_wr_cnt <= sh1_wr_cnt + 32'd1;
        end
        if (c1_done && !sh1_dbg_c1_done_seen) begin
            sh1_dbg_c1_done_seen <= 1'b1;
            $display("[SH1_DBG] conv1_done wr_cnt=%0d c1_wr_idx=%0d wr_mism=%0d (expect %0d y_valid)",
                     sh1_wr_cnt, c1_wr_idx, sh1_wr_mism_cnt, C1_LEN);
        end
    end
end

// conv2 phase0: read issued (check addr vs golden for *next* MAC flat = x_addr_mac)
always @(posedge clk) begin
    if (!reset && (CS == S_CONV2) && c2_busy && !c2_mac_phase &&
        sh1_dbg_sample_rd(sh1_rd_log_cnt, c2_x_addr_mac[14:0])) begin
        $display("[SH1_RD_ISSUE] cycle=%0t flat_nxt=%0d bank=%0d laddr=%0h q_now=%04h gold_nxt=%04h",
                 $time, c2_x_addr_mac[14:0], sh1_rd_bank, sh1_rd_laddr, sh1_rd_q,
                 sh1_gold[c2_x_addr_mac[14:0]]);
    end
end

// conv2 MAC phase1: MAC input is comb c2_x_i_mac (= sh1_rd_q); compare to gold[c2_x_addr]
always @(posedge clk) begin
    if (reset) begin
        sh1_rd_log_cnt       <= 32'd0;
        sh1_rd_mism_cnt      <= 32'd0;
        sh1_dbg_c2_done_seen <= 1'b0;
    end else if ((CS == S_CONV2) && c2_busy && (c2_mac_phase == 1'b1)) begin
        if (sh1_dbg_sample_rd(sh1_rd_log_cnt, c2_x_addr[14:0]))
            $display("[SH1_RD] cycle=%0t flat=%0d x_mac=%0h x_r=%0h x_i=%04h gold=%04h",
                     $time, c2_x_addr[14:0], c2_x_addr_mac, c2_x_addr, c2_x_i_mac,
                     sh1_gold[c2_x_addr[14:0]]);
        if (c2_x_i_mac !== sh1_gold[c2_x_addr[14:0]] &&
            sh1_rd_mism_cnt < SH1_DBG_MISM_CAP) begin
            $display("[SH1_RD_MISMATCH] flat=%0d rtl=%04h gold=%04h x_mac=%0h",
                     c2_x_addr[14:0], c2_x_i_mac, sh1_gold[c2_x_addr[14:0]], c2_x_addr_mac);
            sh1_rd_mism_cnt <= sh1_rd_mism_cnt + 32'd1;
        end
        sh1_rd_log_cnt <= sh1_rd_log_cnt + 32'd1;
    end
    if (c2_done && !sh1_dbg_c2_done_seen) begin
        sh1_dbg_c2_done_seen <= 1'b1;
        $display("[SH1_DBG] conv2_done rd_logged=%0d rd_mism=%0d",
                 sh1_rd_log_cnt, sh1_rd_mism_cnt);
    end
end

// FSM boundary markers
always @(posedge clk) begin
    if (!reset) begin
        if ((CS == S_FILL) && (NS == S_CONV1))
            $display("[SH1_DBG] enter S_CONV1");
        if ((CS == S_CONV1) && (NS == S_CONV2))
            $display("[SH1_DBG] enter S_CONV2");
        if ((CS == S_CONV2) && (NS == S_TAIL))
            $display("[SH1_DBG] enter S_TAIL");
    end
end

`endif

endmodule
