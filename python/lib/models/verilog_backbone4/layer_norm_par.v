// =============================================================================
// layer_norm_par.v  (verilog_backbone4, dim192 / Q7.7)
//
// Parallel (LN_LANES-wide) LayerNorm.  Maps numpy layer_norm():
//   mean = sum(x)/N ; var = sum(x^2)/N - mean^2 ; y = (x-mean)*invstd*g + b
//
// Throughput strategy (minimise cycles, keep cycle time, add SRAM banks):
//   - x stored in LN_LANES banks: feature f -> bank (f % LN_LANES), row (f / LN_LANES).
//     All banks share one address (tok*ROWS + row); bank p returns feature row*LANES+p.
//   - ONE accumulation pass: each lane keeps its own partial s1_lane (Sum x) and
//     s2_lane (Sum x^2) -> NO adder tree on the streaming path (cycle-time safe,
//     same per-lane discipline as linear_ws).  ROWS = N/LANES cycles.
//   - variance one-pass, EXACT integer (no Q7.7 cancellation):
//       NUM = N*S2 - S1*S1  (>=0) ;  var_q77 = round(NUM / (N*N*2^F))
//   - inv_sqrt: Q7.7-native inv_sqrt_nr (var_q77 in, inv_std_q77 out, no shift).
//   - g/b preloaded into registers once (amortised over all N_TOKENS tokens).
//   - NORM pass: LANES-wide, 4-stage pipeline (ci -> *invstd -> *g -> +b), one
//     multiply/add per stage -> cycle-time safe.  ROWS cycles + drain.
//
// Per token ~ 2*ROWS + stat(~5) + inv(5) + drains.  For LANES=32: ~30 cycles.
//
// Fixed-point Q(INT).(FRAC)=Q7.7, DATA_W=16 storage, output clamped to 14-bit
// [SAT_MIN, SAT_MAX] = [-8192, 8191] (matches numpy fp() / *_bi.txt golden).
// Intermediate products clamped to full 16-bit to preserve precision.
//
// x bank read contract (CLK=clk, posedge-sampled): posedge T x_rd_addr stable ->
//   posedge T+2 x_q valid (registered-data SRAM).  ACCUM/NORM consume windows are
//   +1 vs the old ~clk model; xr_a at cnt captures the feature issued at cnt-1.
// g/b ROM read contract (2-phase, CLK=clk): addr held over phase0+phase1 (2 cycles)
//   so the posedge macro Q lands in phase1 -> capture in phase1, no extra retime.
//
// Verilog-2001 synthesizable. Saturation via blocking scratch (no function).
// Golden:  Activation/backbone_blocks_0_after_norm1_out_bi.txt
// Golden-Weight: Weight/backbone_blocks_0_norm1_{weight,bias}_bi.txt
// =============================================================================

module layer_norm_par #(
    parameter FEAT_DIM   = 192,
    parameter N_TOKENS   = 320,
    parameter LN_LANES   = 32,                 // must divide FEAT_DIM
    parameter DATA_W     = 16,
    parameter FRAC       = 7,
    parameter SAT_MAX    = 8191,               // 14-bit Q7.7 output max
    parameter SAT_MIN    = -8192,
    parameter S16_MAX    = 32767,              // intermediate 16-bit clamp
    parameter S16_MIN    = -32768,
    // reciprocals (constant multiplies; round-to-nearest via +half then >>shift)
    parameter MEAN_SHIFT = 22,
    parameter MEAN_RCP   = 21845,              // round(2^22 / 192) ~ 1/192
    parameter VAR_SHIFT  = 40,
    parameter VAR_RCP    = 233018,             // round(2^40 / (192*192*128)) ~ 1/4718592
    // derived
    parameter ROWS       = FEAT_DIM / LN_LANES, // 192/32 = 6
    parameter FEAT_AW    = 8,                   // ceil(log2(FEAT_DIM))
    parameter ROW_AW     = 4,                   // ceil(log2(ROWS)) headroom
    parameter X_AW       = 12,                  // ceil(log2(N_TOKENS*ROWS))
    parameter TOK_AW     = 10,
    parameter ACC1_W     = 24,                  // S1 width (sum of N 14-bit)
    parameter ACC2_W     = 40,                  // S2 width (sum of N 28-bit)
    parameter NUM_W      = 52                   // N*S2 - S1*S1
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire                       start,

    // gamma / beta ROM (2-phase, CLK=~clk); addr 0..FEAT_DIM-1
    output wire [FEAT_AW-1:0]         g_addr_o,
    input  wire signed [DATA_W-1:0]   g_i,
    output wire [FEAT_AW-1:0]         b_addr_o,
    input  wire signed [DATA_W-1:0]   b_i,

    // x activation banks (LN_LANES banks share one addr; lane p = feature row*LANES+p)
    output reg                        x_rd_en,
    output reg  [X_AW-1:0]            x_rd_addr,
    input  wire [LN_LANES*DATA_W-1:0] x_q,        // bank p data = x_q[p*DATA_W +: DATA_W]

    output wire                       busy,
    output reg                        done,

    // y stream: LN_LANES outputs/cycle for features [y_row_o*LANES .. +LANES-1]
    output reg  [LN_LANES*DATA_W-1:0] y_o,
    output reg                        y_valid,
    output reg  [TOK_AW-1:0]          y_tok_o,
    output reg  [ROW_AW-1:0]          y_row_o
);

// -------------------------------------------------------------------------
// States
// -------------------------------------------------------------------------
parameter S_IDLE  = 4'd0;
parameter S_WLOAD = 4'd1;   // load gamma/beta into regs (once)
parameter S_ACC   = 4'd2;   // streaming per-lane sum + sum-sq
parameter S_RED   = 4'd3;   // reduce lanes -> S1, S2
parameter S_MEAN  = 4'd4;   // mean, S1*S1, N*S2
parameter S_VAR   = 4'd5;   // NUM = N*S2 - S1*S1
parameter S_VAR2  = 4'd6;   // var_pre = NUM * VAR_RCP
parameter S_VAR3  = 4'd7;   // var_q77, kick inv_sqrt
parameter S_INV   = 4'd8;   // wait inv_sqrt
parameter S_NORM  = 4'd9;   // streaming normalise + output
parameter S_DONE  = 4'd10;

localparam [ROW_AW-1:0]  ROW_LAST  = ROWS - 1;
localparam [TOK_AW-1:0]  TOK_LAST  = N_TOKENS - 1;
localparam [FEAT_AW-1:0] FEAT_LAST = FEAT_DIM - 1;
localparam [5:0]         ACC_LAST  = ROWS + 2;     // last s2 accumulate beat (+1: posedge x latency)
localparam [5:0]         NORM_LAST = ROWS + 4;     // last y_valid beat (4-stage pipe, +1: posedge x latency)

integer i;

// -------------------------------------------------------------------------
// Registers
// -------------------------------------------------------------------------
reg [3:0]  state;
reg [3:0]  next_state;

reg [TOK_AW-1:0]  tok_cnt;
reg [5:0]         acc_cnt;
reg [5:0]         norm_cnt;

// gamma/beta preload (2-phase)
reg [FEAT_AW-1:0] wl_feat;
reg               wl_phase;
reg signed [DATA_W-1:0] gbuf [0:FEAT_DIM-1];
reg signed [DATA_W-1:0] bbuf [0:FEAT_DIM-1];

// per-lane partial accumulators + ACCUM pipeline
reg signed [ACC1_W-1:0] s1_lane [0:LN_LANES-1];
reg signed [ACC2_W-1:0] s2_lane [0:LN_LANES-1];
reg signed [DATA_W-1:0] xr_a    [0:LN_LANES-1];
reg signed [31:0]       sqr     [0:LN_LANES-1];

// reduced sums + statistics
reg signed [ACC1_W-1:0] S1;
reg signed [ACC2_W-1:0] S2;
reg signed [NUM_W-1:0]  s1s1;
reg signed [NUM_W-1:0]  s2n;
reg signed [NUM_W-1:0]  num;
reg signed [63:0]       var_pre;
reg signed [DATA_W-1:0] mean_q77;
reg signed [DATA_W-1:0] var_q77;
reg signed [DATA_W-1:0] inv_std_q77;

reg               inv_start;
wire              inv_busy;
wire              inv_done;
wire signed [15:0] inv_std_w;       // Q7.7 inv_sqrt result (native, no shift)

// NORM pipeline (LANES-wide)
reg signed [16:0]       ci_n    [0:LN_LANES-1];
reg signed [DATA_W-1:0] cistd_n [0:LN_LANES-1];
reg signed [DATA_W-1:0] wci_n   [0:LN_LANES-1];
reg [ROW_AW-1:0]        cap_row, rn2, rn3, rn4;
reg                     v1, v2, v3, v4;

// per-block blocking scratch (each name used in ONE always -> single-driver)
reg signed [ACC1_W-1:0] red1;       // lane reduction (comb)
reg signed [ACC2_W-1:0] red2;
reg signed [63:0]       mn_sh;      // S_MEAN
reg signed [63:0]       vr_sh;      // S_VAR3
reg signed [DATA_W-1:0] vr_v;
reg signed [33:0]       n3_p1;      // N3
reg signed [63:0]       n3_sh;
reg signed [31:0]       n4_p2;      // N4
reg signed [63:0]       n4_sh;
reg signed [17:0]       sc_y;       // N5

// -------------------------------------------------------------------------
// `ifndef SYNTHESIS init (avoid X in sim)
// -------------------------------------------------------------------------
`ifndef SYNTHESIS
integer ii;
initial begin
    for (ii = 0; ii < FEAT_DIM; ii = ii + 1) begin
        gbuf[ii] = {DATA_W{1'b0}};
        bbuf[ii] = {DATA_W{1'b0}};
    end
    for (ii = 0; ii < LN_LANES; ii = ii + 1) begin
        s1_lane[ii] = {ACC1_W{1'b0}};
        s2_lane[ii] = {ACC2_W{1'b0}};
        xr_a[ii]    = {DATA_W{1'b0}};
        sqr[ii]     = 32'sd0;
        ci_n[ii]    = 17'sd0;
        cistd_n[ii] = {DATA_W{1'b0}};
        wci_n[ii]   = {DATA_W{1'b0}};
    end
end
`endif

// -------------------------------------------------------------------------
// Wires
// -------------------------------------------------------------------------
assign busy = (state != S_IDLE);

assign g_addr_o = (state == S_WLOAD) ? wl_feat : {FEAT_AW{1'b0}};
assign b_addr_o = (state == S_WLOAD) ? wl_feat : {FEAT_AW{1'b0}};

wire wl_done  = (wl_feat == FEAT_LAST) && (wl_phase == 1'b1);
wire acc_done = (acc_cnt == ACC_LAST);
wire norm_done = (norm_cnt == NORM_LAST);
wire tok_last = (tok_cnt == TOK_LAST);

// inv_sqrt sub-module (Q7.7-native): var_q77 in, inv_std_q77 out (no shift)
inv_sqrt_nr u_inv_sqrt (
    .clk   (clk),
    .reset (reset),
    .start (inv_start),
    .v_i   (var_q77),
    .busy  (inv_busy),
    .done  (inv_done),
    .y_o   (inv_std_w)
);

// lane reduction (combinational; consumed only in S_RED, once per token)
always @(*) begin
    red1 = {ACC1_W{1'b0}};
    red2 = {ACC2_W{1'b0}};
    for (i = 0; i < LN_LANES; i = i + 1) begin
        red1 = red1 + s1_lane[i];
        red2 = red2 + s2_lane[i];
    end
end

// -------------------------------------------------------------------------
// FSM seg1: state register
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM seg2: next-state (ternary per branch, no nested if per style rule)
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:  next_state = start    ? S_WLOAD : state;
        S_WLOAD: next_state = wl_done  ? S_ACC   : state;
        S_ACC:   next_state = acc_done ? S_RED   : state;
        S_RED:   next_state = S_MEAN;
        S_MEAN:  next_state = S_VAR;
        S_VAR:   next_state = S_VAR2;
        S_VAR2:  next_state = S_VAR3;
        S_VAR3:  next_state = S_INV;
        S_INV:   next_state = inv_done  ? S_NORM : state;
        S_NORM:  next_state = norm_done ? (tok_last ? S_DONE : S_ACC) : state;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// -------------------------------------------------------------------------
// gamma/beta 2-phase preload counters + capture
// -------------------------------------------------------------------------
// wl_phase: toggle each WLOAD cycle (sole driver)
always @(posedge clk) begin
    if (reset)                 wl_phase <= 1'b0;
    else if (state != S_WLOAD) wl_phase <= 1'b0;
    else                       wl_phase <= ~wl_phase;
end

// wl_feat: advance on phase1 up to FEAT_LAST, then hold (sole driver)
always @(posedge clk) begin
    if (reset)                     wl_feat <= {FEAT_AW{1'b0}};
    else if (state != S_WLOAD)     wl_feat <= {FEAT_AW{1'b0}};
    else if (wl_phase == 1'b0)     wl_feat <= wl_feat;
    else if (wl_feat != FEAT_LAST) wl_feat <= wl_feat + {{(FEAT_AW-1){1'b0}}, 1'b1};
    else                           wl_feat <= wl_feat;
end

always @(posedge clk) begin
    if (state == S_WLOAD && wl_phase == 1'b1) begin
        gbuf[wl_feat] <= g_i;
        bbuf[wl_feat] <= b_i;
    end
end

// -------------------------------------------------------------------------
// token counter (advance after each token's NORM completes)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        tok_cnt <= {TOK_AW{1'b0}};
    else if (state == S_IDLE)
        tok_cnt <= {TOK_AW{1'b0}};
    else if (state == S_NORM && norm_done && !tok_last)
        tok_cnt <= tok_cnt + {{(TOK_AW-1){1'b0}}, 1'b1};
end

// -------------------------------------------------------------------------
// ACCUM counter
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        acc_cnt <= 6'd0;
    else if (state != S_ACC)
        acc_cnt <= 6'd0;
    else if (!acc_done)
        acc_cnt <= acc_cnt + 6'd1;
end

// x read port (combinational): ACCUM issues row=acc_cnt, NORM issues row=norm_cnt.
// posedge SRAM: x_q for addr issued at cnt=r is valid +1 cycle later (consumed at cnt=r+1).
always @(*) begin
    x_rd_en   = 1'b0;
    x_rd_addr = {X_AW{1'b0}};
    if (state == S_ACC && acc_cnt < ROWS) begin
        x_rd_en   = 1'b1;
        x_rd_addr = tok_cnt * ROWS + acc_cnt;
    end else if (state == S_NORM && norm_cnt < ROWS) begin
        x_rd_en   = 1'b1;
        x_rd_addr = tok_cnt * ROWS + norm_cnt;
    end
end

// x capture (shared by ACCUM A1 and NORM N1; phases never overlap -> single driver)
// +1 posedge x latency: at cnt=r the bus carries the feature issued at cnt=r-1,
// so capture window is [1..ROWS] (xr_a <- row cnt-1).
always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            xr_a[i] <= {DATA_W{1'b0}};
    end else if ((state == S_ACC  && acc_cnt  >= 6'd1 && acc_cnt  <= ROWS) ||
                 (state == S_NORM && norm_cnt >= 6'd1 && norm_cnt <= ROWS)) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            xr_a[i] <= $signed(x_q[i*DATA_W +: DATA_W]);
    end
end

always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LN_LANES; i = i + 1) begin
            s1_lane[i] <= {ACC1_W{1'b0}};
            sqr[i]     <= 32'sd0;
        end
    end else if (state == S_ACC && acc_cnt == 6'd0) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            s1_lane[i] <= {ACC1_W{1'b0}};
    end else if (state == S_ACC && acc_cnt >= 6'd2 && acc_cnt <= (ROWS + 6'd1)) begin
        for (i = 0; i < LN_LANES; i = i + 1) begin
            s1_lane[i] <= s1_lane[i] +
                {{(ACC1_W-DATA_W){xr_a[i][DATA_W-1]}}, xr_a[i]};
            sqr[i]     <= $signed(xr_a[i]) * $signed(xr_a[i]);
        end
    end
end

always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            s2_lane[i] <= {ACC2_W{1'b0}};
    end else if (state == S_ACC && acc_cnt == 6'd0) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            s2_lane[i] <= {ACC2_W{1'b0}};
    end else if (state == S_ACC && acc_cnt >= 6'd3 && acc_cnt <= ACC_LAST) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            s2_lane[i] <= s2_lane[i] +
                {{(ACC2_W-32){sqr[i][31]}}, sqr[i]};
    end
end

// -------------------------------------------------------------------------
// statistics: reduce -> mean / var -> inv_sqrt kick
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        S1 <= {ACC1_W{1'b0}};
        S2 <= {ACC2_W{1'b0}};
    end else if (state == S_RED) begin
        S1 <= red1;
        S2 <= red2;
    end
end

always @(posedge clk) begin
    if (reset) begin
        mean_q77 <= 16'sd0;
        s1s1     <= {NUM_W{1'b0}};
        s2n      <= {NUM_W{1'b0}};
    end else if (state == S_MEAN) begin
        // mean = round(S1 / 192)
        mn_sh = ($signed(S1) * $signed(MEAN_RCP) + (64'sd1 <<< (MEAN_SHIFT-1))) >>> MEAN_SHIFT;
        mean_q77 <= (mn_sh > S16_MAX) ? S16_MAX[15:0] :
                    (mn_sh < S16_MIN) ? S16_MIN[15:0] : mn_sh[15:0];
        s1s1 <= $signed(S1) * $signed(S1);
        s2n  <= $signed(S2) * $signed(FEAT_DIM);
    end
end

always @(posedge clk) begin
    if (reset)
        num <= {NUM_W{1'b0}};
    else if (state == S_VAR)
        num <= s2n - s1s1;                 // = N*S2 - S1^2 >= 0
end

always @(posedge clk) begin
    if (reset)
        var_pre <= 64'sd0;
    else if (state == S_VAR2)
        var_pre <= $signed(num) * $signed(VAR_RCP);
end

// var_q77: clamp variance to [1, SAT_MAX] Q7.7 at S_VAR3 (sole driver)
always @(posedge clk) begin
    if (reset)
        var_q77 <= 16'sd0;
    else if (state == S_VAR3) begin
        vr_sh = (var_pre + (64'sd1 <<< (VAR_SHIFT-1))) >>> VAR_SHIFT;
        // clamp to [1, SAT_MAX] (positive Q7.7, avoid 0 for inv_sqrt)
        vr_v = (vr_sh > SAT_MAX) ? SAT_MAX[15:0] :
               (vr_sh < 64'sd1)  ? 16'sd1 : vr_sh[15:0];
        var_q77 <= vr_v;
    end
end

// inv_start pulse: 1 only at S_VAR3 (sole driver)
always @(posedge clk) begin
    if (reset)                 inv_start <= 1'b0;
    else if (state == S_VAR3)  inv_start <= 1'b1;
    else                       inv_start <= 1'b0;
end

// inv_std (Q7.7) captured directly from Q7.7-native inv_sqrt (no shift)
always @(posedge clk) begin
    if (reset)
        inv_std_q77 <= 16'sd0;
    else if (state == S_INV && inv_done)
        inv_std_q77 <= inv_std_w;
end

// -------------------------------------------------------------------------
// NORM counter + 4-stage pipeline (LANES-wide)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        norm_cnt <= 6'd0;
    else if (state != S_NORM)
        norm_cnt <= 6'd0;
    else if (!norm_done)
        norm_cnt <= norm_cnt + 6'd1;
end

// valid + row tags propagate with the pipeline
always @(posedge clk) begin
    if (reset || state != S_NORM) begin
        v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        cap_row <= {ROW_AW{1'b0}};
        rn2 <= {ROW_AW{1'b0}};
        rn3 <= {ROW_AW{1'b0}};
        rn4 <= {ROW_AW{1'b0}};
    end else begin
        // +1 posedge x latency: xr_a at norm_cnt holds row norm_cnt-1, so v1/cap_row
        // track [1..ROWS] with row = norm_cnt-1.
        v1 <= (norm_cnt >= 6'd1 && norm_cnt <= ROWS);
        cap_row <= (norm_cnt - 6'd1);
        v2 <= v1; rn2 <= cap_row;
        v3 <= v2; rn3 <= rn2;
        v4 <= v3; rn4 <= rn3;
    end
end

// N2 centered = x - mean (17-bit, no clamp)
always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            ci_n[i] <= 17'sd0;
    end else if (state == S_NORM) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            ci_n[i] <= $signed({xr_a[i][DATA_W-1], xr_a[i]}) -
                       $signed({mean_q77[DATA_W-1], mean_q77});
    end
end

// N3 cistd = sat16(round(ci * inv_std >> FRAC))
always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            cistd_n[i] <= {DATA_W{1'b0}};
    end else if (state == S_NORM) begin
        for (i = 0; i < LN_LANES; i = i + 1) begin
            n3_p1 = $signed(ci_n[i]) * $signed(inv_std_q77);
            n3_sh = (n3_p1 + (64'sd1 <<< (FRAC-1))) >>> FRAC;
            cistd_n[i] <= (n3_sh > S16_MAX) ? S16_MAX[15:0] :
                          (n3_sh < S16_MIN) ? S16_MIN[15:0] : n3_sh[15:0];
        end
    end
end

// N4 wci = sat16(round(cistd * gamma >> FRAC)); gamma indexed by rn3
always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LN_LANES; i = i + 1)
            wci_n[i] <= {DATA_W{1'b0}};
    end else if (state == S_NORM) begin
        for (i = 0; i < LN_LANES; i = i + 1) begin
            n4_p2 = $signed(cistd_n[i]) * $signed(gbuf[rn3*LN_LANES + i]);
            // n4_p2 is signed reg -> keep signed (no concat) so >>> is arithmetic
            n4_sh = (n4_p2 + (64'sd1 <<< (FRAC-1))) >>> FRAC;
            wci_n[i] <= (n4_sh > S16_MAX) ? S16_MAX[15:0] :
                        (n4_sh < S16_MIN) ? S16_MIN[15:0] : n4_sh[15:0];
        end
    end
end

// N5 y = sat14(wci + beta); beta indexed by rn4
always @(posedge clk) begin
    if (reset) begin
        y_o     <= {(LN_LANES*DATA_W){1'b0}};
        y_valid <= 1'b0;
        y_tok_o <= {TOK_AW{1'b0}};
        y_row_o <= {ROW_AW{1'b0}};
    end else if (state == S_NORM && v4) begin
        for (i = 0; i < LN_LANES; i = i + 1) begin
            sc_y = $signed(wci_n[i]) + $signed(bbuf[rn4*LN_LANES + i]);
            y_o[i*DATA_W +: DATA_W] <= (sc_y > SAT_MAX) ? SAT_MAX[15:0] :
                                       (sc_y < SAT_MIN) ? SAT_MIN[15:0] : sc_y[15:0];
        end
        y_valid <= 1'b1;
        y_tok_o <= tok_cnt;
        y_row_o <= rn4;
    end else begin
        y_valid <= 1'b0;
    end
end

// done pulse
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE)
        done <= 1'b1;
    else
        done <= 1'b0;
end

endmodule
