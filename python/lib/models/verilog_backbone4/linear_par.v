// =============================================================================
// linear_par.v  (verilog_backbone4, dim192 / Q7.7)  -- B2 engine
//
// Weight-stationary linear layer with P-feature parallelism + pipelined adder
// tree.  y = x @ W^T + b, computed LANES neurons in parallel, P features/cycle.
//
//   throughput  = LANES * P  MAC/cycle
//   cycle time  : kept = 1 multiply / 1 add  (adder tree is fully PIPELINED;
//                 each of log2(P) levels registers its sums, so the critical
//                 path never grows with P).
//
// Bit-exactness vs linear_ws (truncating integer datapath, NO rounding):
//   acc[lane]  = SUM_f ( w_int[lane][f] * x_int[f] )      (exact integer)
//   y_int[lane]= sat14( (acc >>> FRAC) + bias_int[lane] )  (>>>FRAC = floor)
//   The adder tree carries FULL precision (no intermediate truncation), so the
//   reorganized summation is identical to linear_ws's serial accumulation.
//
// Dataflow (per neuron group of LANES neurons, then neu_base += LANES):
//   S_WLOAD : 2-phase ROM read -> w_buf[lane][feat]      (each weight read once)
//   S_BLOAD : 2-phase ROM read -> bias_reg[lane]
//   S_MAC   : continuous stream of N_TOKENS*FS issues (FS = IN_DIM/P steps/token)
//             pipeline:  x word (P feat) -> capture xreg + wreg[lane][p]
//                        -> prod_r[lane][p] -> 5-level adder tree -> a5[lane]
//                        -> acc[lane] (restart on step0, emit on stepFS-1)
//             each completed token emits LANES results in ONE y beat.
//
// x  read contract (P-wide word, 1P SRAM, CLK=clk posedge, 1-cycle latency):
//   posedge T  : x_rd_addr = tok*FS + step           (P features/word)
//   posedge T+1: x_q = word@T  (P features, lane p -> x_q[p*DATA_W +: DATA_W])
//   x flat layout = token-major P-feature words (matches 4xSram_1920 for IN=192).
//
// Weight ROM (2-phase, CLK=clk posedge): w_addr = (neu_base+lane)*IN_DIM + feat
//   addr@phase0 -> w_i captured @phase1 (phase slack absorbs 1-cycle latency).
// Bias ROM: b_addr = neu_base + lane.
//
// Pipeline tag alignment (issue -> acc): TAP parameters below; off-by-one is a
//   one-line TAP change.  WMUL_TAP feeds w_buf read; ACC_TAP feeds restart/emit.
//
// Verilog-2001 synthesizable.  P must be a power of two; IN_DIM % P == 0.
// =============================================================================

module linear_par #(
    parameter IN_DIM    = 192,
    parameter OUT_DIM   = 768,
    parameter N_TOKENS  = 320,
    parameter LANES     = 16,
    parameter P         = 32,
    parameter DATA_W    = 16,
    parameter PROD_W    = 32,
    parameter TREE_W    = 40,
    parameter ACC_W     = 48,
    parameter FRAC_BITS = 7,
    parameter SAT_MAX   = 8191,
    parameter SAT_MIN   = -8192,
    parameter FEAT_AW   = 10,   // ceil(log2(IN_DIM)) headroom (<=768)
    parameter XW_AW     = 14,   // ceil(log2(N_TOKENS*IN_DIM/P)) headroom
    parameter W_ADDR_W  = 20,   // ceil(log2(OUT_DIM*IN_DIM)) headroom
    parameter B_ADDR_W  = 10,
    parameter NEU_AW    = 10,
    parameter TOK_AW    = 10,
    parameter STEP_AW   = 5     // ceil(log2(IN_DIM/P)) ; 24 -> 5
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire                       start,

    // activation read port (P features/word; token-major word addr)
    output reg                        x_rd_en,
    output reg  [XW_AW-1:0]           x_rd_addr,
    input  wire [P*DATA_W-1:0]        x_q,

    // weight / bias ROM (2-phase)
    output wire [W_ADDR_W-1:0]        w_addr_o,
    input  wire signed [DATA_W-1:0]   w_i,
    output wire [B_ADDR_W-1:0]        b_addr_o,
    input  wire signed [DATA_W-1:0]   b_i,

    output wire                       busy,
    output reg                        done,

    // y stream: one beat per token = LANES neuron results in parallel
    output reg  signed [LANES*DATA_W-1:0] y_o,
    output reg                        y_valid,
    output reg  [TOK_AW-1:0]          y_tok_o,
    output reg  [NEU_AW-1:0]          y_neu_base_o
);

// -------------------------------------------------------------------------
// Local params
// -------------------------------------------------------------------------
parameter S_IDLE  = 3'd0;
parameter S_WLOAD = 3'd1;
parameter S_BLOAD = 3'd2;
parameter S_MAC   = 3'd3;
parameter S_DONE  = 3'd4;

localparam integer FS        = IN_DIM / P;          // steps per token
localparam integer GROUPS    = OUT_DIM / LANES;
localparam [STEP_AW-1:0] STEP_LAST = FS - 1;
localparam [TOK_AW-1:0]  TOK_LAST  = N_TOKENS - 1;
localparam [FEAT_AW-1:0] FEAT_LAST = IN_DIM - 1;

// pipeline taps (issue -> stage). Derivation (tg[n] holds issue-k info during
// cycle k+1+n): xreg/wreg capture at k+1 using tg[0]; prod@k+3; a1@k+4 ..
// a5@k+8; acc/emit consume a5 during k+8 using tg[7].
localparam integer WMUL_TAP = 0;   // wreg read aligned with xreg (cap stage)
localparam integer ACC_TAP  = 7;   // a5/acc/emit use flags delayed 7 from tg[0]
localparam integer TAGD     = 8;   // tag shift depth (0..7)

// MAC issue counter total = N_TOKENS*FS ; drain adds ACC_TAP
localparam integer MAC_ISSUES = N_TOKENS * FS;

integer gi, gj, gl, gp;

// -------------------------------------------------------------------------
// Registers
// -------------------------------------------------------------------------
reg [2:0] state, next_state;

reg [NEU_AW-1:0]  neu_base;

// weight/bias preload (2-phase)
reg [NEU_AW-1:0]  wl_lane;     // 0..LANES-1
reg [FEAT_AW-1:0] wl_feat;
reg               wl_phase;
reg [NEU_AW-1:0]  bl_lane;
reg               bl_phase;

// MAC issue counters
reg [TOK_AW-1:0]  iss_tok;
reg [STEP_AW-1:0] iss_step;
reg [TOK_AW-1:0]  mac_done_cnt; // counts emitted tokens for group end detect
reg               mac_run;      // issuing phase active

// bias storage (1D).  NOTE: per-lane w_buf / wreg / prod_r / adder-tree regs
// (a1..a4) live INSIDE the generate block as per-lane 1D arrays -- NO module
// has a [LANES][..] vector array (project rule: no 3D regs).
reg signed [DATA_W-1:0] bias_reg [0:LANES-1];

// datapath shared regs (1D only): x word, per-lane partial sum & accumulator
reg signed [DATA_W-1:0] xreg     [0:P-1];
reg signed [TREE_W-1:0] a5       [0:LANES-1];
reg signed [ACC_W-1:0]  acc      [0:LANES-1];

// tag pipeline (scalar, shared across lanes): {valid,first,last,tok,step}
reg               tg_val   [0:TAGD-1];
reg               tg_first [0:TAGD-1];
reg               tg_last  [0:TAGD-1];
reg [TOK_AW-1:0]  tg_tok   [0:TAGD-1];
reg [STEP_AW-1:0] tg_step  [0:TAGD-1];

`ifndef SYNTHESIS
integer ii;
initial begin
    for (ii = 0; ii < LANES; ii = ii + 1) begin
        bias_reg[ii] = {DATA_W{1'b0}};
        acc[ii]      = {ACC_W{1'b0}};
    end
end
`endif

// -------------------------------------------------------------------------
// Wires / control
// -------------------------------------------------------------------------
wire wl_done = (wl_lane == LANES-1) && (wl_feat == FEAT_LAST) && (wl_phase == 1'b1);
wire bl_done = (bl_lane == LANES-1) && (bl_phase == 1'b1);
wire iss_last = (iss_tok == TOK_LAST) && (iss_step == STEP_LAST);
wire mac_all_emitted = (mac_done_cnt == N_TOKENS); // all tokens of group out
wire group_last = ((neu_base + LANES) >= OUT_DIM);

assign busy = (state != S_IDLE);

wire [NEU_AW-1:0] wl_neu = neu_base + wl_lane;
assign w_addr_o = (state == S_WLOAD) ? (wl_neu * IN_DIM + wl_feat) : {W_ADDR_W{1'b0}};
assign b_addr_o = (state == S_BLOAD) ? (neu_base + bl_lane) : {B_ADDR_W{1'b0}};

// step used for wreg read at cap stage (tag delayed WMUL_TAP)
wire [STEP_AW-1:0] wmul_step = tg_step[WMUL_TAP];
wire               wmul_val  = tg_val[WMUL_TAP];

// acc-stage flags (tag delayed ACC_TAP)
wire               acc_val   = tg_val[ACC_TAP];
wire               acc_first = tg_first[ACC_TAP];
wire               acc_last  = tg_last[ACC_TAP];
wire [TOK_AW-1:0]  acc_tok   = tg_tok[ACC_TAP];

// -------------------------------------------------------------------------
// FSM
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:  next_state = start ? S_WLOAD : state;
        S_WLOAD: next_state = wl_done ? S_BLOAD : state;
        S_BLOAD: next_state = bl_done ? S_MAC : state;
        S_MAC:   next_state = (mac_all_emitted && !group_last) ? S_WLOAD :
                              (mac_all_emitted &&  group_last) ? S_DONE  : state;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// neu_base
always @(posedge clk) begin
    if (reset)                          neu_base <= {NEU_AW{1'b0}};
    else if (state == S_IDLE && start)  neu_base <= {NEU_AW{1'b0}};
    else if (state == S_MAC && mac_all_emitted && !group_last)
                                        neu_base <= neu_base + LANES;
end

// -------------------------------------------------------------------------
// Weight preload counters (2-phase)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)                  wl_phase <= 1'b0;
    else if (state != S_WLOAD)  wl_phase <= 1'b0;
    else                        wl_phase <= ~wl_phase;
end

always @(posedge clk) begin
    if (reset)                     wl_feat <= {FEAT_AW{1'b0}};
    else if (state != S_WLOAD)     wl_feat <= {FEAT_AW{1'b0}};
    else if (wl_phase == 1'b0)     wl_feat <= wl_feat;
    else if (wl_feat == FEAT_LAST) wl_feat <= {FEAT_AW{1'b0}};
    else                           wl_feat <= wl_feat + 1'b1;
end

always @(posedge clk) begin
    if (reset)                  wl_lane <= {NEU_AW{1'b0}};
    else if (state != S_WLOAD)  wl_lane <= {NEU_AW{1'b0}};
    else if (wl_phase == 1'b1 && wl_feat == FEAT_LAST && wl_lane != LANES-1)
                                wl_lane <= wl_lane + 1'b1;
    else                        wl_lane <= wl_lane;
end

// NOTE: w_buf write moved into the per-lane generate block (each lane stores
// its own 1D w_buf and self-selects by wl_lane==L) to avoid a 3D reg array.

// -------------------------------------------------------------------------
// Bias preload counters (2-phase)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)                  bl_phase <= 1'b0;
    else if (state != S_BLOAD)  bl_phase <= 1'b0;
    else                        bl_phase <= ~bl_phase;
end

always @(posedge clk) begin
    if (reset)                  bl_lane <= {NEU_AW{1'b0}};
    else if (state != S_BLOAD)  bl_lane <= {NEU_AW{1'b0}};
    else if (bl_phase == 1'b1 && bl_lane != LANES-1)
                                bl_lane <= bl_lane + 1'b1;
    else                        bl_lane <= bl_lane;
end

always @(posedge clk) begin
    if (reset) begin
        for (gi = 0; gi < LANES; gi = gi + 1) bias_reg[gi] <= {DATA_W{1'b0}};
    end else if (state == S_BLOAD && bl_phase == 1'b1)
        bias_reg[bl_lane] <= b_i;
end

// -------------------------------------------------------------------------
// MAC issue counters (step inner, token outer); run until all issued
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)                          mac_run <= 1'b0;
    else if (state == S_BLOAD && bl_done) mac_run <= 1'b1;       // enter MAC next
    else if (state == S_MAC && iss_last)  mac_run <= 1'b0;
    else if (state != S_MAC)            mac_run <= 1'b0;
end

always @(posedge clk) begin
    if (reset)                       iss_step <= {STEP_AW{1'b0}};
    else if (state != S_MAC)         iss_step <= {STEP_AW{1'b0}};
    else if (!mac_run)               iss_step <= {STEP_AW{1'b0}};
    else if (iss_step == STEP_LAST)  iss_step <= {STEP_AW{1'b0}};
    else                             iss_step <= iss_step + 1'b1;
end

always @(posedge clk) begin
    if (reset)                                       iss_tok <= {TOK_AW{1'b0}};
    else if (state != S_MAC)                         iss_tok <= {TOK_AW{1'b0}};
    else if (!mac_run)                               iss_tok <= {TOK_AW{1'b0}};
    else if (iss_step == STEP_LAST && iss_tok != TOK_LAST) iss_tok <= iss_tok + 1'b1;
    else if (iss_step == STEP_LAST)                  iss_tok <= {TOK_AW{1'b0}};
end

// x read port (combinational issue; addr = tok*FS + step)
always @(*) begin
    x_rd_en   = 1'b0;
    x_rd_addr = {XW_AW{1'b0}};
    if (state == S_MAC && mac_run) begin
        x_rd_en   = 1'b1;
        x_rd_addr = iss_tok * FS + iss_step;
    end
end

// emitted-token counter (group end when == N_TOKENS)
always @(posedge clk) begin
    if (reset)                      mac_done_cnt <= {TOK_AW{1'b0}};
    else if (state != S_MAC)        mac_done_cnt <= {TOK_AW{1'b0}};
    else if (acc_val && acc_last)   mac_done_cnt <= mac_done_cnt + 1'b1;
end

// -------------------------------------------------------------------------
// Tag pipeline (loaded at issue, shifted TAGD deep)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        for (gi = 0; gi < TAGD; gi = gi + 1) begin
            tg_val[gi]   <= 1'b0;
            tg_first[gi] <= 1'b0;
            tg_last[gi]  <= 1'b0;
            tg_tok[gi]   <= {TOK_AW{1'b0}};
            tg_step[gi]  <= {STEP_AW{1'b0}};
        end
    end else begin
        tg_val[0]   <= (state == S_MAC) && mac_run;
        tg_first[0] <= (iss_step == {STEP_AW{1'b0}});
        tg_last[0]  <= (iss_step == STEP_LAST);
        tg_tok[0]   <= iss_tok;
        tg_step[0]  <= iss_step;
        for (gj = 1; gj < TAGD; gj = gj + 1) begin
            tg_val[gj]   <= tg_val[gj-1];
            tg_first[gj] <= tg_first[gj-1];
            tg_last[gj]  <= tg_last[gj-1];
            tg_tok[gj]   <= tg_tok[gj-1];
            tg_step[gj]  <= tg_step[gj-1];
        end
    end
end

// -------------------------------------------------------------------------
// Datapath: x capture (whole P-feat word) ; runs while tag valid at tap0..
// xreg captured one cycle after issue (x_q valid then)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        for (gp = 0; gp < P; gp = gp + 1) xreg[gp] <= {DATA_W{1'b0}};
    end else if (tg_val[0]) begin
        for (gp = 0; gp < P; gp = gp + 1)
            xreg[gp] <= x_q[gp*DATA_W +: DATA_W];
    end
end

// -------------------------------------------------------------------------
// Per-lane datapath (generate): wreg, prod, adder tree, acc, output SAT
// -------------------------------------------------------------------------
genvar L;
generate
for (L = 0; L < LANES; L = L + 1) begin : LANE

    // per-lane storage / pipeline regs (ALL 1D : packed vector + 1 unpacked dim)
    reg signed [DATA_W-1:0] w_buf  [0:IN_DIM-1];   // this lane's weights
    reg signed [DATA_W-1:0] wreg   [0:P-1];
    reg signed [PROD_W-1:0] prod_r [0:P-1];
    reg signed [TREE_W-1:0] a1     [0:15];
    reg signed [TREE_W-1:0] a2     [0:7];
    reg signed [TREE_W-1:0] a3     [0:3];
    reg signed [TREE_W-1:0] a4     [0:1];

`ifndef SYNTHESIS
    integer iw;
    initial for (iw = 0; iw < IN_DIM; iw = iw + 1) w_buf[iw] = {DATA_W{1'b0}};
`endif

    // weight load (2-phase): this lane self-selects by wl_lane == L
    always @(posedge clk) begin : wload
        if (state == S_WLOAD && wl_phase == 1'b1 && wl_lane == L)
            w_buf[wl_feat] <= w_i;
    end

    // wreg: read w_buf[wmul_step*P + p] aligned with xreg (cap stage)
    always @(posedge clk) begin : wcap
        integer p;
        if (reset) begin
            for (p = 0; p < P; p = p + 1) wreg[p] <= {DATA_W{1'b0}};
        end else if (wmul_val) begin
            for (p = 0; p < P; p = p + 1)
                wreg[p] <= w_buf[wmul_step*P + p];
        end
    end

    // multiply (reg x reg)
    always @(posedge clk) begin : mul
        integer p;
        if (reset) begin
            for (p = 0; p < P; p = p + 1) prod_r[p] <= {PROD_W{1'b0}};
        end else begin
            for (p = 0; p < P; p = p + 1)
                prod_r[p] <= wreg[p] * xreg[p];
        end
    end

    // adder tree level 1: 32 -> 16
    always @(posedge clk) begin : t1
        integer j;
        for (j = 0; j < 16; j = j + 1)
            a1[j] <= {{(TREE_W-PROD_W){prod_r[2*j][PROD_W-1]}}, prod_r[2*j]} +
                     {{(TREE_W-PROD_W){prod_r[2*j+1][PROD_W-1]}}, prod_r[2*j+1]};
    end
    // level 2: 16 -> 8
    always @(posedge clk) begin : t2
        integer j;
        for (j = 0; j < 8; j = j + 1) a2[j] <= a1[2*j] + a1[2*j+1];
    end
    // level 3: 8 -> 4
    always @(posedge clk) begin : t3
        integer j;
        for (j = 0; j < 4; j = j + 1) a3[j] <= a2[2*j] + a2[2*j+1];
    end
    // level 4: 4 -> 2
    always @(posedge clk) begin : t4
        integer j;
        for (j = 0; j < 2; j = j + 1) a4[j] <= a3[2*j] + a3[2*j+1];
    end
    // level 5: 2 -> 1 (partial sum of P products for one step) -> module a5[L]
    always @(posedge clk) begin : t5
        a5[L] <= a4[0] + a4[1];
    end

    // accumulate: restart on first step, add otherwise (acc-stage tag)
    always @(posedge clk) begin : accum
        if (reset)
            acc[L] <= {ACC_W{1'b0}};
        else if (acc_val && acc_first)
            acc[L] <= {{(ACC_W-TREE_W){a5[L][TREE_W-1]}}, a5[L]};
        else if (acc_val)
            acc[L] <= acc[L] + {{(ACC_W-TREE_W){a5[L][TREE_W-1]}}, a5[L]};
    end

end
endgenerate

// -------------------------------------------------------------------------
// Output SAT: when acc-stage tag is last, full sum = acc + a5, then
// (>>>FRAC) + bias, clamp 14-bit.  Emit LANES results in one beat.
// -------------------------------------------------------------------------
reg signed [ACC_W-1:0]  sum_full   [0:LANES-1];
reg signed [ACC_W-1:0]  shr_b      [0:LANES-1];
reg signed [DATA_W-1:0] sat_lane_w [0:LANES-1];

always @(*) begin
    for (gl = 0; gl < LANES; gl = gl + 1) begin
        sum_full[gl] = acc[gl] + {{(ACC_W-TREE_W){a5[gl][TREE_W-1]}}, a5[gl]};
        // NOTE: keep both operands SIGNED.  An unsigned concat here would make
        // the whole expression unsigned context and demote (sum_full >>> FRAC)
        // from arithmetic to LOGICAL shift -> negative sums blow up to ~2^41
        // and saturate to +SAT_MAX.  $signed() forces arithmetic shift (matches
        // the self-ref oracle which shifts in a standalone signed statement).
        shr_b[gl]    = $signed(sum_full[gl] >>> FRAC_BITS) +
                       $signed({{(ACC_W-DATA_W){bias_reg[gl][DATA_W-1]}}, bias_reg[gl]});
        sat_lane_w[gl] = (shr_b[gl] > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
                         (shr_b[gl] < SAT_MIN) ? SAT_MIN[DATA_W-1:0] :
                         shr_b[gl][DATA_W-1:0];
    end
end

always @(posedge clk) begin
    if (reset) begin
        y_o          <= {(LANES*DATA_W){1'b0}};
        y_valid      <= 1'b0;
        y_tok_o      <= {TOK_AW{1'b0}};
        y_neu_base_o <= {NEU_AW{1'b0}};
    end else if (acc_val && acc_last) begin
        for (gl = 0; gl < LANES; gl = gl + 1)
            y_o[gl*DATA_W +: DATA_W] <= sat_lane_w[gl];
        y_valid      <= 1'b1;
        y_tok_o      <= acc_tok;
        y_neu_base_o <= neu_base;
    end else begin
        y_valid <= 1'b0;
    end
end

// done pulse
always @(posedge clk) begin
    if (reset)               done <= 1'b0;
    else if (state == S_DONE) done <= 1'b1;
    else                     done <= 1'b0;
end

endmodule
