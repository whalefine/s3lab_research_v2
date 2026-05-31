// =============================================================================
// head_top.v -- verilog2 merged head (conv1 + conv2 + tail + cal_bbox)
// Plan B: S_FILL reads backbone norm output directly from Sram_tok1 (sram_tok1 port),
//         reorders to NCHW, and writes to Sram_v (x_buf).
//         a_i / a_valid inputs removed; S_FILL is self-driven (2-phase).
//
// x_buf: Sram_v (depth 8192)
//   write: S_FILL 2-phase read Sram_tok1 → NCHW reorder → write Sram_v
//   read:  S_CONV1 conv MAC phase0/phase1
//
// sh1_buf: Sram_q (lo) + Sram_k (hi)
// sh2_buf: Sram_tok2
// wgt_buf: Sram_tok1 (S_FILL: read backbone norm; S_CONV1/S_CONV2: weight prefetch)
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

    output wire        busy,
    output reg         done,

    output wire [DATA_W-1:0] cx_o,
    output wire [DATA_W-1:0] cy_o,
    output wire [DATA_W-1:0] w_o,
    output wire [DATA_W-1:0] h_o,

    // SRAM macro interfaces (instantiated in sglatrack_top; cal_bbox's Sram_qkm stays in cal_bbox)
    output wire                    sram_x_ceb_o,
    output wire                    sram_x_web_o,
    output wire [13:0]             sram_x_addr_o,
    output wire [DATA_W-1:0]       sram_x_din_o,
    input  wire [DATA_W-1:0]       sram_x_q_i,

    output wire                    sram_sh1_lo_ceb_o,
    output wire                    sram_sh1_lo_web_o,
    output wire [13:0]             sram_sh1_lo_addr_o,
    output wire [DATA_W-1:0]       sram_sh1_lo_din_o,
    input  wire [DATA_W-1:0]       sram_sh1_lo_q_i,

    output wire                    sram_sh1_hi_ceb_o,
    output wire                    sram_sh1_hi_web_o,
    output wire [13:0]             sram_sh1_hi_addr_o,
    output wire [DATA_W-1:0]       sram_sh1_hi_din_o,
    input  wire [DATA_W-1:0]       sram_sh1_hi_q_i,

    output wire                    sram_tok2_ceb_o,
    output wire                    sram_tok2_web_o,
    output wire [13:0]             sram_tok2_addr_o,
    output wire [DATA_W-1:0]       sram_tok2_din_o,
    input  wire [DATA_W-1:0]       sram_tok2_q_i,

    output wire                    sram_tok1_ceb_o,
    output wire                    sram_tok1_web_o,
    output wire [13:0]             sram_tok1_addr_o,
    output wire [DATA_W-1:0]       sram_tok1_din_o,
    input  wire [DATA_W-1:0]       sram_tok1_q_i,

    output wire                    sram_bbox_ceb_o,
    output wire                    sram_bbox_web_o,
    output wire [10:0]             sram_bbox_addr_o,
    output wire [DATA_W-1:0]       sram_bbox_din_o,
    input  wire [DATA_W-1:0]       sram_bbox_q_i
);

localparam FEAT_SZ     = FEAT_H * FEAT_W;
localparam SEARCH_SZ   = FEAT_SZ;          // 256 search spatial positions
localparam IN_LEN_HEAD = FEAT_SZ * IN_CH;  // 8192
localparam C1_LEN      = C_SH1 * FEAT_SZ;
localparam C2_LEN      = C_SH2 * FEAT_SZ;
localparam C1_HALF     = C1_LEN >> 1;
localparam X_BUF_AW    = 13;
localparam SH1_BANK_AW = 14;
localparam SH2_AW      = 14;
localparam WGT_AW      = 14;

wire rst_n = ~reset;

parameter S_IDLE  = 3'd0;
parameter S_FILL  = 3'd1;
parameter S_CONV1 = 3'd2;
parameter S_CONV2 = 3'd3;
parameter S_TAIL  = 3'd4;
parameter S_BBOX  = 3'd5;
parameter S_DONE  = 3'd6;

reg [2:0] CS, NS;

// Plan B S_FILL: 2-phase read Sram_tok1, write Sram_v (x_buf) with NCHW reorder
reg [7:0]  fill_n;      // search spatial index 0..SEARCH_SZ-1 (255)
reg [4:0]  fill_c;      // channel index 0..IN_CH-1 (31)
reg        fill_phase;  // 0=ADDR (read Sram_tok1), 1=USE (write Sram_v)

// Sram_tok1 read addr: backbone norm layout = tok*EMBED_DIM + ch (token-major)
wire [13:0] fill_tok1_addr = ({6'b0, fill_n} + LENS_Z[13:0]) * IN_CH[13:0]
                             + {9'b0, fill_c};
// Sram_v write addr: NCHW reorder (channel-major)
wire [X_BUF_AW-1:0] fill_xbuf_addr = {fill_c, fill_n};
wire        fill_last = (fill_n == SEARCH_SZ[7:0] - 8'd1)
                     && (fill_c == IN_CH[4:0] - 5'd1)
                     && (fill_phase == 1'b1);

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
// Port regs: comb _n -> posedge clk -> macro (1-cycle align with conv MAC phase0/1)
reg        x_ceb;
reg        x_web;
reg [X_BUF_AW-1:0] x_addr;
reg [DATA_W-1:0] x_din;
wire [DATA_W-1:0] x_q;

reg        x_ceb_n;
reg        x_web_n;
reg [X_BUF_AW-1:0] x_addr_n;
reg [DATA_W-1:0] x_din_n;

wire [X_BUF_AW-1:0] x_rd_addr   = c1_x_addr_mac[X_BUF_AW-1:0];
// Plan B: S_FILL write to x_buf driven by fill_phase=1 (USE phase after Sram_tok1 read)
wire        x_fill_wr    = (CS == S_FILL) && (fill_phase == 1'b1);
wire        x_c1_rd_req  = (CS == S_CONV1) && c1_busy && !c1_mac_phase;

assign sram_x_ceb_o  = x_ceb;
assign sram_x_web_o  = x_web;
assign sram_x_addr_o = {{(SH1_BANK_AW-X_BUF_AW){1'b0}}, x_addr};
assign sram_x_din_o  = x_din;
assign x_q           = sram_x_q_i;

// sh1 SRAM (conv1 capture -> conv2 read); flat[14:0] -> bank + 14b local (split at C1_HALF)
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

assign sram_sh1_lo_ceb_o  = sh1_lo_ceb;
assign sram_sh1_lo_web_o  = sh1_lo_web;
assign sram_sh1_lo_addr_o = sh1_lo_addr;
assign sram_sh1_lo_din_o  = sh1_lo_din;
assign sh1_lo_q           = sram_sh1_lo_q_i;

assign sram_sh1_hi_ceb_o  = sh1_hi_ceb;
assign sram_sh1_hi_web_o  = sh1_hi_web;
assign sram_sh1_hi_addr_o = sh1_hi_addr;
assign sram_sh1_hi_din_o  = sh1_hi_din;
assign sh1_hi_q           = sram_sh1_hi_q_i;

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

assign sram_tok2_ceb_o  = sh2_ceb;
assign sram_tok2_web_o  = sh2_web;
assign sram_tok2_addr_o = sh2_addr;
assign sram_tok2_din_o  = sh2_din;
assign sh2_q           = sram_tok2_q_i;

// wgt_buf SRAM (Sram_tok1; shared conv1/conv2 — S_CONV1 vs S_CONV2 mutually exclusive)
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

assign sram_tok1_ceb_o  = wgt_ceb;
assign sram_tok1_web_o  = wgt_web;
assign sram_tok1_addr_o = wgt_addr;
assign sram_tok1_din_o  = wgt_din;
assign wgt_q           = sram_tok1_q_i;

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

// SRAM port mux next-state (mutually exclusive sources per macro)
always @(*) begin
    x_ceb_n  = 1'b1;
    x_web_n  = 1'b1;
    x_addr_n = {X_BUF_AW{1'b0}};
    x_din_n  = {DATA_W{1'b0}};

    sh2_ceb_n  = 1'b1;
    sh2_web_n  = 1'b1;
    sh2_addr_n = {SH2_AW{1'b0}};
    sh2_din_n  = {DATA_W{1'b0}};

    sh1_lo_ceb_n  = 1'b1;
    sh1_lo_web_n  = 1'b1;
    sh1_lo_addr_n = {SH1_BANK_AW{1'b0}};
    sh1_lo_din_n  = {DATA_W{1'b0}};
    sh1_hi_ceb_n  = 1'b1;
    sh1_hi_web_n  = 1'b1;
    sh1_hi_addr_n = {SH1_BANK_AW{1'b0}};
    sh1_hi_din_n  = {DATA_W{1'b0}};

    wgt_ceb_n  = 1'b1;
    wgt_web_n  = 1'b1;
    wgt_addr_n = {WGT_AW{1'b0}};
    wgt_din_n  = {DATA_W{1'b0}};

    // Plan B S_FILL: phase1 writes Sram_v with data from Sram_tok1 Q
    if (x_fill_wr) begin
        x_ceb_n  = 1'b0;
        x_web_n  = 1'b0;
        x_addr_n = fill_xbuf_addr;
        x_din_n  = wgt_q;
    end else if (x_c1_rd_req) begin
        x_ceb_n  = 1'b0;
        x_web_n  = 1'b1;
        x_addr_n = x_rd_addr;
    end

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

    // Plan B S_FILL: phase0 reads Sram_tok1 (backbone norm output)
    if ((CS == S_FILL) && (fill_phase == 1'b0)) begin
        wgt_ceb_n  = 1'b0;
        wgt_web_n  = 1'b1;
        wgt_addr_n = fill_tok1_addr;
    end else if (c1_wgt_wr_en || c2_wgt_wr_en) begin
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
    .so_ceb_o      (sram_bbox_ceb_o ),
    .so_web_o      (sram_bbox_web_o ),
    .so_addr_o     (sram_bbox_addr_o),
    .so_din_o      (sram_bbox_din_o ),
    .so_q_i        (sram_bbox_q_i   ),
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
        S_FILL:  NS = fill_last ? S_CONV1 : S_FILL;
        S_CONV1: NS = c1_done ? S_CONV2 : S_CONV1;
        S_CONV2: NS = c2_done ? S_TAIL : S_CONV2;
        S_TAIL:  NS = t_done ? S_BBOX : S_TAIL;
        S_BBOX:  NS = b_done ? S_DONE : S_BBOX;
        S_DONE:  NS = S_IDLE;
        default: NS = S_IDLE;
    endcase
end

// Plan B fill counters: 2-phase (read Sram_tok1, write Sram_v)
always @(posedge clk) begin
    if (reset) begin
        fill_n     <= 8'd0;
        fill_c     <= 5'd0;
        fill_phase <= 1'b0;
    end else if (CS == S_IDLE) begin
        fill_n     <= 8'd0;
        fill_c     <= 5'd0;
        fill_phase <= 1'b0;
    end else if (CS == S_FILL) begin
        if (fill_phase == 1'b0) begin
            fill_phase <= 1'b1;
        end else begin
            fill_phase <= 1'b0;
            if (fill_c == IN_CH[4:0] - 5'd1) begin
                fill_c <= 5'd0;
                if (fill_n < SEARCH_SZ[7:0] - 8'd1)
                    fill_n <= fill_n + 8'd1;
            end else begin
                fill_c <= fill_c + 5'd1;
            end
        end
    end
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
        bbox_reg[0] <= {DATA_W{1'b0}};
        bbox_reg[1] <= {DATA_W{1'b0}};
        bbox_reg[2] <= {DATA_W{1'b0}};
        bbox_reg[3] <= {DATA_W{1'b0}};
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

endmodule
