// =============================================================================
// head_top.v -- verilog_head2 head (conv1 + conv2 + tail + cal_bbox)
// Plan A: conv1 reads backbone norm directly from Sram_tok1 (token-major, no S_FILL).
//   tok1 addr = (LENS_Z + spatial) * IN_CH + ic  ==  (64+n)*32+c  (same as TB preload)
//   conv1 x_addr_mac is NCHW flat {ic[4:0], n[7:0]} -> decode for tok1 read only
//
// x_buf: Sram_v unused in head path (port held idle; macro may remain in sglatrack_top)
//
// sh1_buf: Sram_q (lo) + Sram_k (hi)
// sh2_buf: Sram_tok2
// Sram_tok1: TB preload backbone (token-major); S_PIPE conv1 MAC reads activations
// Golden: Activation/backbone_after_norm_backbone_out_bi.txt (tok1 preload, token-major)
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
parameter S_PIPE  = 3'd2;   // conv1 + conv2 overlapped (no S_FILL) (dataflow)
parameter S_TAIL  = 3'd4;
parameter S_BBOX  = 3'd5;
parameter S_DONE  = 3'd6;

reg [2:0] CS, NS;

reg [DATA_W-1:0] bbox_reg [0:3];

reg c1_start, c2_start, t_start, b_start;
reg c1_started, c2_started, t_started, b_started;
reg c1_done_latch, c2_done_latch;

wire              c1_busy, c1_done, c1_y_valid;
wire [DATA_W-1:0] c1_y_data;
wire [7:0]        c1_y_oc;
wire [4:0]        c1_y_oh;
wire [4:0]        c1_y_ow;
wire [13:0]       c1_x_addr;
wire [13:0]       c1_x_addr_mac;
wire              c1_mac_phase;
wire              c1_mac_active;
wire [DATA_W-1:0] c1_x_i_mac;

wire              c2_busy, c2_done, c2_y_valid;
wire [DATA_W-1:0] c2_y_data;
wire [7:0]        c2_y_oc;
wire [4:0]        c2_y_oh;
wire [4:0]        c2_y_ow;
wire [14:0]       c2_x_addr;
wire [14:0]       c2_x_addr_mac;
wire              c2_mac_phase;
wire              c2_mac_active;
wire [4:0]        c2_cur_oh;
wire              c2_stall;
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
reg [31:0] c1_wr_idx;
reg [31:0] c2_wr_idx;

// conv1 row completion for conv2 pipeline (3x3 needs rows 0..oh2+1 through sh1)
reg [5:0] c1_rows_ready;
wire      c1_row_end = c1_y_valid && (c1_y_ow == FEAT_W[4:0] - 5'd1)
                     && (c1_y_oc == C_SH1[7:0] - 8'd1);
wire [5:0] c2_need_rows = (c2_cur_oh + 6'd2 > FEAT_H[5:0]) ? FEAT_H[5:0] : (c2_cur_oh + 6'd2);

// x_buf (Sram_v) — idle in Plan A; conv1 reads Sram_tok1 directly
assign sram_x_ceb_o  = 1'b1;
assign sram_x_web_o  = 1'b1;
assign sram_x_addr_o = 14'd0;
assign sram_x_din_o  = {DATA_W{1'b0}};

// conv1 MAC phase0: read tok1 with token-major addr (decode NCHW x_addr_mac_rd)
wire [13:0] c1_tok1_rd_addr = (LENS_Z[13:0] + {6'b0, c1_x_addr_mac[7:0]}) * IN_CH[13:0]
                             + {9'b0, c1_x_addr_mac[12:8]};
wire        c1_tok1_rd_req  = (CS == S_PIPE) && c1_busy && c1_mac_active;

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

wire [14:0]       sh1_wr_flat15  = c1_y_oc * FEAT_SZ[14:0]
                                 + c1_y_oh * FEAT_W[14:0]
                                 + c1_y_ow;
wire [14:0]       sh1_rd_flat15 = c2_x_addr_mac[14:0];
wire              sh1_wr_bank    = (sh1_wr_flat15 >= C1_HALF);
wire              sh1_rd_bank    = (sh1_rd_flat15 >= C1_HALF);
wire [SH1_BANK_AW-1:0] sh1_wr_laddr = sh1_wr_bank ?
    (sh1_wr_flat15 - C1_HALF) : sh1_wr_flat15[SH1_BANK_AW-1:0];
wire [SH1_BANK_AW-1:0] sh1_rd_laddr = sh1_rd_bank ?
    (sh1_rd_flat15 - C1_HALF) : sh1_rd_flat15[SH1_BANK_AW-1:0];
// Issue SRAM read on conv MAC phase0 (x_addr_mac_rd = x_addr_nxt; Q valid next posedge)
wire        sh1_c2_rd_req  = (CS == S_PIPE) && c2_mac_active;
assign c2_stall = c2_busy && c2_mac_active && (c1_rows_ready < c2_need_rows);

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

// NCHW flat: oc * (H*W) + oh * W + ow (must match tail x_addr; OC_PAR drain is not stream order)
wire [SH2_AW-1:0] sh2_wr_addr = c2_y_oc * FEAT_SZ[SH2_AW-1:0]
                              + c2_y_oh * FEAT_W[SH2_AW-1:0]
                              + c2_y_ow;
wire [SH2_AW-1:0] sh2_rd_addr = t_x_addr_mac[SH2_AW-1:0];
wire              t_mac_active;
// tail: 1-phase MAC read on mac_active
wire        sh2_tail_rd_req  = (CS == S_TAIL) && t_busy && t_mac_active;

assign sram_tok2_ceb_o  = sh2_ceb;
assign sram_tok2_web_o  = sh2_web;
assign sram_tok2_addr_o = sh2_addr;
assign sram_tok2_din_o  = sh2_din;
assign sh2_q           = sram_tok2_q_i;

// Sram_tok1 (TB preload; conv1 MAC reads activations; weights in conv internal ROM)
reg        tok1_ceb;
reg        tok1_web;
reg [WGT_AW-1:0] tok1_addr;
reg [DATA_W-1:0] tok1_din;

reg        tok1_ceb_n;
reg        tok1_web_n;
reg [WGT_AW-1:0] tok1_addr_n;
reg [DATA_W-1:0] tok1_din_n;

wire [DATA_W-1:0] tok1_q = sram_tok1_q_i;

assign sram_tok1_ceb_o  = tok1_ceb;
assign sram_tok1_web_o  = tok1_web;
assign sram_tok1_addr_o = tok1_addr;
assign sram_tok1_din_o  = tok1_din;

wire              tc_sig_v, to_v, ts_sig_v;
wire [DATA_W-1:0] tc_sig_d, to_d, ts_sig_d;

wire              b_busy, b_done, b_valid;
wire [DATA_W-1:0] b_data;
wire [1:0]        b_idx;

// conv1: 1-phase MAC pipe — read every MAC cycle; x_i valid when mac_phase_o (post-fill)
assign c1_x_i_mac = ((CS == S_PIPE) && c1_busy && c1_mac_phase) ? tok1_q : {DATA_W{1'b0}};

// conv2: pipelined MAC — read every MAC cycle; x_i when c2_mac_phase (tok from prior read).
assign sh1_rd_q = sh1_rd_bank ? sh1_hi_q : sh1_lo_q;
assign c2_x_i_mac = ((CS == S_PIPE) && c2_busy && c2_mac_phase) ? sh1_rd_q : {DATA_W{1'b0}};

// tail: pipelined MAC — read every MAC cycle; x_i when t_mac_phase.
assign t_x_i_mac = ((CS == S_TAIL) && t_busy && t_mac_phase) ? sh2_q : {DATA_W{1'b0}};

// SRAM port mux next-state (one if / else if chain per macro)
always @(*) begin
    tok1_ceb_n  = 1'b1;
    tok1_web_n  = 1'b1;
    tok1_addr_n = {WGT_AW{1'b0}};
    tok1_din_n  = {DATA_W{1'b0}};
    if (c1_tok1_rd_req) begin
        tok1_ceb_n  = 1'b0;
        tok1_web_n  = 1'b1;
        tok1_addr_n = c1_tok1_rd_addr;
    end
end

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

always @(*) begin
    sh1_lo_ceb_n  = 1'b1;
    sh1_lo_web_n  = 1'b1;
    sh1_lo_addr_n = {SH1_BANK_AW{1'b0}};
    sh1_lo_din_n  = {DATA_W{1'b0}};
    sh1_hi_ceb_n  = 1'b1;
    sh1_hi_web_n  = 1'b1;
    sh1_hi_addr_n = {SH1_BANK_AW{1'b0}};
    sh1_hi_din_n  = {DATA_W{1'b0}};
    if (c1_y_valid && !sh1_wr_bank) begin
        sh1_lo_ceb_n  = 1'b0;
        sh1_lo_web_n  = 1'b0;
        sh1_lo_addr_n = sh1_wr_laddr;
        sh1_lo_din_n  = c1_y_data;
    end else if (c1_y_valid && sh1_wr_bank) begin
        sh1_hi_ceb_n  = 1'b0;
        sh1_hi_web_n  = 1'b0;
        sh1_hi_addr_n = sh1_wr_laddr;
        sh1_hi_din_n  = c1_y_data;
    end else if (sh1_c2_rd_req && !sh1_rd_bank) begin
        sh1_lo_ceb_n  = 1'b0;
        sh1_lo_web_n  = 1'b1;
        sh1_lo_addr_n = sh1_rd_laddr;
    end else if (sh1_c2_rd_req && sh1_rd_bank) begin
        sh1_hi_ceb_n  = 1'b0;
        sh1_hi_web_n  = 1'b1;
        sh1_hi_addr_n = sh1_rd_laddr;
    end
end

always @(posedge clk) begin
    if (reset) begin
        tok1_ceb  <= 1'b1;
        tok1_web  <= 1'b1;
        tok1_addr <= {WGT_AW{1'b0}};
        tok1_din  <= {DATA_W{1'b0}};
    end else begin
        tok1_ceb  <= tok1_ceb_n;
        tok1_web  <= tok1_web_n;
        tok1_addr <= tok1_addr_n;
        tok1_din  <= tok1_din_n;
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
    .OC_PAR      (8       ),
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
    .y_oc    (c1_y_oc       ),
    .y_oh    (c1_y_oh       ),
    .y_ow    (c1_y_ow       ),
    .mac_phase_o   (c1_mac_phase  ),
    .x_addr_mac_rd (c1_x_addr_mac ),
    .wgt_wr_en     (c1_wgt_wr_en  ),
    .wgt_wr_addr   (c1_wgt_wr_addr),
    .wgt_wr_data   (c1_wgt_wr_data),
    .wgt_rd_req    (c1_wgt_rd_req ),
    .wgt_rd_addr   (c1_wgt_rd_addr),
    .wgt_rd_i      (16'd0         ),
    .stall         (1'b0          ),
    .mac_active_o  (c1_mac_active ),
    .cur_oh_o      (              )
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
    .OC_PAR      (8       ),
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
    .y_oc    (c2_y_oc   ),
    .y_oh    (c2_y_oh   ),
    .y_ow    (c2_y_ow   ),
    .mac_phase_o   (c2_mac_phase  ),
    .x_addr_mac_rd (c2_x_addr_mac ),
    .wgt_wr_en     (c2_wgt_wr_en  ),
    .wgt_wr_addr   (c2_wgt_wr_addr),
    .wgt_wr_data   (c2_wgt_wr_data),
    .wgt_rd_req    (c2_wgt_rd_req ),
    .wgt_rd_addr   (c2_wgt_rd_addr),
    .wgt_rd_i      (tok1_q        ),
    .stall         (c2_stall      ),
    .mac_active_o  (c2_mac_active ),
    .cur_oh_o      (c2_cur_oh     )
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
    .mac_active_o     (t_mac_active ),
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
        S_IDLE:  NS = start ? S_PIPE : S_IDLE;
        S_PIPE:  NS = (c1_done_latch && c2_done_latch) ? S_TAIL : S_PIPE;
        S_TAIL:  NS = t_done ? S_BBOX : S_TAIL;
        S_BBOX:  NS = b_done ? S_DONE : S_BBOX;
        S_DONE:  NS = S_IDLE;
        default: NS = S_IDLE;
    endcase
end

// Control: one posedge always; reset first; case(CS); stream idx after case
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
        c1_done_latch <= 1'b0;
        c2_done_latch <= 1'b0;
        c1_wr_idx     <= 32'd0;
        c2_wr_idx     <= 32'd0;
        c1_rows_ready <= 6'd0;
        bbox_reg[0]   <= {DATA_W{1'b0}};
        bbox_reg[1] <= {DATA_W{1'b0}};
        bbox_reg[2] <= {DATA_W{1'b0}};
        bbox_reg[3] <= {DATA_W{1'b0}};
    end else begin
        if (CS == S_IDLE) begin
            c1_started    <= 1'b0;
            c2_started    <= 1'b0;
            t_started     <= 1'b0;
            b_started     <= 1'b0;
            c1_done_latch <= 1'b0;
            c2_done_latch <= 1'b0;
            c1_rows_ready <= 6'd0;
        end

        if (c1_row_end)
            c1_rows_ready <= {1'b0, c1_y_oh} + 6'd1;

        case (CS)
            S_PIPE: begin
                if (c1_done)
                    c1_done_latch <= 1'b1;
                if (c2_done)
                    c2_done_latch <= 1'b1;
                if (!c1_started) begin
                    c1_start   <= 1'b1;
                    c1_started <= 1'b1;
                    c1_wr_idx  <= 32'd0;
                end
                // 3x3 conv2 @ oh2 needs sh1 rows 0..oh2+1 -> start after row 1 done
                if (!c2_started && (c1_rows_ready >= 6'd2)) begin
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
