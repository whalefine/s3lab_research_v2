// =============================================================================
// sglatrack_top.v  (verilog_backbone4, dim192 / Q7.7)
//
// Backbone4 top.  CURRENT STAGE: connects layer_norm_par (block0 norm1) to REAL
// activation SRAM macros (Sram_1920, 112-bit = 8 x 14-bit Q7.7, depth 1920) to
// verify the LayerNorm <-> SRAM read path under the CLK=clk posedge contract.
// Grows into the full backbone (attention / MLP sharing this activation SRAM).
//
// ACTIVATION SRAM ORGANISATION (x input, 32 feature/cycle)
//   - 4 banks of Sram_1920 (depth 1920, width 112 = 8 features x 14-bit).
//   - All banks share one word address  addr = tok*ROWS + row   (0..1919).
//   - bank b (0..3) stores features { row*32 + b*8 + s }, s=0..7, at D[s*14 +: 14].
//   - read: q_all = {q3,q2,q1,q0} (448-bit); lane i  -> q_all[i*14 +: 14]
//           = feature row*32+i ; sign-extend 14->16 -> x_q[i*16 +: 16].
//   - Q7.7 values fit in 14-bit [-8192,8191] so 14->16 sign-extension is lossless.
//
// SRAM read contract (Sram_1920, CLK=clk, posedge):
//   posedge T   : A/CEB/WEB stable
//   posedge T+1 : Q valid for A@T (registered-data, 1-cycle latency)
//   -> matches layer_norm_par's retimed (+1) ACCUM/NORM x streaming.
//
// SIM NOTE: streaming read address comes from posedge FFs, so in zero-delay RTL
//   it transitions at the sampling posedge -> spurious $hold violations and the
//   TSMC notifier X-corrupts Q.  Run VCS with +notimingcheck (functional RTL);
//   real silicon / gate-level SDF have clk-to-q margin and are unaffected.
//
// PRELOAD (single-port: write before LN runs, no read/write same cycle):
//   external pre_we / pre_addr / pre_wdata (448-bit = one row of 32 features).
//   bank mux gives pre_we priority; LN x_rd_* drives banks only when !pre_we.
//
// gamma/beta stay in ROM (weights, not activation SRAM): exposed as g/b ports
//   for the parent/TB to feed (2-phase preload inside layer_norm_par).
//
// Verilog-2001 synthesizable.  Replace Sram_1920 with PDK macro at synth
// (VCS: include the generated Sram_1920.v ; compiler: Sram_1920 1920 112 4 s).
// Golden:  Activation/backbone_blocks_0_after_norm1_out_bi.txt
// =============================================================================

module sglatrack_top #(
    parameter FEAT_DIM = 192,
    parameter N_TOKENS = 320,
    parameter LN_LANES = 32,
    parameter DATA_W   = 16,
    parameter MACRO_W  = 14,                 // SRAM stores 14-bit Q7.7
    parameter FEAT_AW  = 8,
    parameter X_AW     = 12,
    parameter TOK_AW   = 10,
    parameter ROW_AW   = 4,
    // derived bank geometry (used in port widths; keep in parameter list)
    parameter FEAT_PER_BANK = 8,                       // 112 / 14
    parameter BANKS         = LN_LANES / FEAT_PER_BANK, // 4
    parameter BANK_W        = FEAT_PER_BANK * MACRO_W,  // 112
    parameter BANK_AW       = 11                        // ceil(log2(1920))
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        start,

    // ---- activation preload write (one 448-bit word = 32 features = 1 row) ----
    input  wire                        pre_we,
    input  wire [BANK_AW-1:0]          pre_addr,
    input  wire [BANKS*BANK_W-1:0]     pre_wdata,

    // ---- gamma/beta ROM (weights stay in ROM; parent/TB feeds) ----
    output wire [FEAT_AW-1:0]          g_addr_o,
    input  wire signed [DATA_W-1:0]    g_i,
    output wire [FEAT_AW-1:0]          b_addr_o,
    input  wire signed [DATA_W-1:0]    b_i,

    // ---- LN outputs ----
    output wire                        busy,
    output wire                        done,
    output wire [LN_LANES*DATA_W-1:0]  y_o,
    output wire                        y_valid,
    output wire [TOK_AW-1:0]           y_tok_o,
    output wire [ROW_AW-1:0]           y_row_o
);

// -------------------------------------------------------------------------
// Derived (non-port) localparams
// -------------------------------------------------------------------------
localparam ROWS = FEAT_DIM / LN_LANES;   // 6

integer gi_i;

// -------------------------------------------------------------------------
// LN <-> SRAM wires
// -------------------------------------------------------------------------
wire                       x_rd_en;
wire [X_AW-1:0]            x_rd_addr;
reg  [LN_LANES*DATA_W-1:0] x_q_w;          // assembled 512-bit x bus to LN
wire [BANKS*BANK_W-1:0]    q_all;          // 448-bit concatenation of 4 bank Q

// shared single-port control (all banks lockstep on same word address)
reg                        bank_ceb;       // active-low chip enable
reg                        bank_web;       // active-low write enable (0=write,1=read)
reg  [BANK_AW-1:0]         bank_a;

// -------------------------------------------------------------------------
// bank mux (combinational): preload write has priority, else LN read.
// default-first to avoid latch ; single-port -> never read+write same cycle.
// -------------------------------------------------------------------------
always @(*) begin
    bank_ceb = 1'b1;                        // deselect (idle)
    bank_web = 1'b1;                        // read mode (safe idle)
    bank_a   = {BANK_AW{1'b0}};
    if (pre_we) begin
        bank_ceb = 1'b0;
        bank_web = 1'b0;
        bank_a   = pre_addr;
    end else if (x_rd_en) begin
        bank_ceb = 1'b0;
        bank_web = 1'b1;
        bank_a   = x_rd_addr[BANK_AW-1:0];
    end
end

// -------------------------------------------------------------------------
// x bus assembly: lane i <- q_all[i*14 +: 14], sign-extend 14->16
// -------------------------------------------------------------------------
always @(*) begin
    for (gi_i = 0; gi_i < LN_LANES; gi_i = gi_i + 1)
        x_q_w[gi_i*DATA_W +: DATA_W] =
            {{(DATA_W-MACRO_W){q_all[gi_i*MACRO_W + (MACRO_W-1)]}},
             q_all[gi_i*MACRO_W +: MACRO_W]};
end

// -------------------------------------------------------------------------
// 4 x Sram_1920 (CLK=clk, posedge, 1-cycle read latency)
// D = pre_wdata slice (only consumed when bank_web=0) ; Q -> q_all slice
// -------------------------------------------------------------------------
genvar b;
generate
    for (b = 0; b < BANKS; b = b + 1) begin : BANK
        Sram_1920 u_bank (
            .SLP    (1'b0),
            .DSLP   (1'b0),
            .SD     (1'b0),
            .PUDELAY(),
            .CLK    (clk),
            .CEB    (bank_ceb),
            .WEB    (bank_web),
            .BIST   (1'b0),
            .CEBM   (),
            .WEBM   (),
            .A      (bank_a),
            .D      (pre_wdata[b*BANK_W +: BANK_W]),
            .BWEB   ({BANK_W{1'b0}}),
            .AM     (),
            .DM     (),
            .BWEBM  ({BANK_W{1'b0}}),
            .RTSEL  (2'b01),
            .WTSEL  (2'b00),
            .Q      (q_all[b*BANK_W +: BANK_W])
        );
    end
endgenerate

// -------------------------------------------------------------------------
// LayerNorm core
// -------------------------------------------------------------------------
layer_norm_par #(
    .FEAT_DIM (FEAT_DIM),
    .N_TOKENS (N_TOKENS),
    .LN_LANES (LN_LANES)
) u_ln (
    .clk       (clk),
    .reset     (reset),
    .start     (start),
    .g_addr_o  (g_addr_o),
    .g_i       (g_i),
    .b_addr_o  (b_addr_o),
    .b_i       (b_i),
    .x_rd_en   (x_rd_en),
    .x_rd_addr (x_rd_addr),
    .x_q       (x_q_w),
    .busy      (busy),
    .done      (done),
    .y_o       (y_o),
    .y_valid   (y_valid),
    .y_tok_o   (y_tok_o),
    .y_row_o   (y_row_o)
);

endmodule
