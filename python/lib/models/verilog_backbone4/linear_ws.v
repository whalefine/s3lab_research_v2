// =============================================================================
// linear_ws.v  (verilog_backbone4, dim192 / Q7.7)
//
// Weight-Stationary, neuron-parallel linear layer (y = x @ W^T + b).
// Generalizes the proven backbone3 mlp_ws FC1 dataflow into a standalone,
// LANES-parameterized linear engine (no ReLU, single x read port).
//
// Dataflow (per neuron group of LANES neurons):
//   S_WLOAD : 2-phase ROM read  -> w_lane[lane][feat]   (each weight read once)
//   S_BLOAD : 2-phase ROM read  -> bias_reg[lane]
//   S_MAC   : for each token, pipelined x read (1/cycle) broadcast to LANES
//             multipliers; acc[lane] += x[feat]*w_lane[lane][feat]
//   S_SAT   : per token, LANES neuron results (>>>FRAC + bias + sat to OUT range)
//   repeat tokens 0..N_TOKENS-1, then neu_base += LANES until OUT_DIM done
//
// Key property (vs backbone3 linear_vec8 = activation-stationary, 1 MAC/cycle):
//   - each weight is read from ROM exactly ONCE (resident in reg during MAC)
//   - activation read is 1 word/cycle, broadcast to LANES lanes -> LANES MAC/cycle
//   - weight ROM bandwidth stays 1 word/cycle; no ROM/SRAM banking needed
//   - to scale throughput, raise LANES (8 -> 32): widen w_lane/mul/acc arrays only
//
// Fixed-point: Q(INT_BITS).(FRAC_BITS) = Q7.7 (dim192). DATA_W=16 storage,
//   numeric range clamped to [SAT_MIN, SAT_MAX] = [-8192, 8191]. Shift = FRAC.
//   ACC_W must hold IN_DIM products of DATA_W x DATA_W.
//
// ROM read contract (CLK = clk, posedge): 2-phase preload; addr@phase0 -> data
//   captured @phase1 (phase slack absorbs the 1-posedge macro latency, no retime).
// x  read contract (1P, CLK = clk, posedge): addr@T sampled @T+1 -> x_q valid @T+2.
//   streaming MAC consumes x ONE cycle later than the old ~clk model (windows +1).
//   x flat addr = tok * IN_DIM + feat (token-major, row-major).
//
// Weight flat addr (parent ROM mux): w_addr = neu * IN_DIM + feat (neuron-major).
// Bias addr: b_addr = neu.
//
// Verilog-2001 synthesizable. Saturation via wire only (no function).
// =============================================================================

module linear_ws #(
    parameter IN_DIM    = 192,
    parameter OUT_DIM   = 576,
    parameter N_TOKENS  = 320,
    parameter LANES     = 64,
    parameter DATA_W    = 16,
    parameter ACC_W     = 36,
    parameter FRAC_BITS = 7,
    parameter SAT_MAX   = 8191,
    parameter SAT_MIN   = -8192,
    parameter X_AW      = 17,   // ceil(log2(N_TOKENS*IN_DIM)) headroom
    parameter W_ADDR_W  = 18,   // ceil(log2(OUT_DIM*IN_DIM)) headroom
    parameter B_ADDR_W  = 10,   // ceil(log2(OUT_DIM))
    parameter FEAT_AW   = 8,    // ceil(log2(IN_DIM))
    parameter NEU_AW    = 10,   // ceil(log2(OUT_DIM))
    parameter TOK_AW    = 10    // ceil(log2(N_TOKENS))
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire                       start,

    // activation read port (token-major flat addr; 1-cycle latency)
    output reg                        x_rd_en,
    output reg  [X_AW-1:0]            x_rd_addr,
    input  wire signed [DATA_W-1:0]   x_q,

    // weight / bias ROM (2-phase, CLK=~clk)
    output wire [W_ADDR_W-1:0]        w_addr_o,
    input  wire signed [DATA_W-1:0]   w_i,
    output wire [B_ADDR_W-1:0]        b_addr_o,
    input  wire signed [DATA_W-1:0]   b_i,

    output wire                       busy,
    output reg                        done,

    // y stream
    output reg  signed [DATA_W-1:0]   y_o,
    output reg                        y_valid,
    output reg  [TOK_AW-1:0]          y_tok_o,
    output reg  [NEU_AW-1:0]          y_neu_o
);

// -------------------------------------------------------------------------
// Parameters / localparams
// -------------------------------------------------------------------------
parameter S_IDLE    = 3'd0;
parameter S_WLOAD   = 3'd1;
parameter S_BLOAD   = 3'd2;
parameter S_MAC     = 3'd3;
parameter S_SAT     = 3'd4;
parameter S_DONE_ST = 3'd5;

localparam [FEAT_AW-1:0] FEAT_LAST = IN_DIM - 1;
localparam [NEU_AW-1:0]  NEU_LAST  = OUT_DIM - 1;
localparam [TOK_AW-1:0]  TOK_LAST  = N_TOKENS - 1;
localparam integer       GROUPS    = OUT_DIM / LANES;

// MAC pipeline boundaries (mac_cnt drives feat 0..IN_DIM-1 + drain).
// posedge SRAM adds 1 read-latency cycle vs old ~clk model -> +1 drain (IN_DIM+3).
localparam [FEAT_AW+1:0] MAC_LAST  = IN_DIM + 3;

// SAT pipeline runs LANES + 2 beats
localparam [7:0]         SAT_LAST  = LANES + 1;

integer i;

// -------------------------------------------------------------------------
// Registers
// -------------------------------------------------------------------------
reg [2:0]  state;
reg [2:0]  next_state;

reg [NEU_AW-1:0]  neu_base;
reg [TOK_AW-1:0]  tok_cnt;

// weight preload (2-phase, separate lane/feat counters; IN_DIM not power-of-2)
// lane counters 8-bit to support LANES up to 128 (sat_lane counts to LANES+1)
reg [7:0]         wl_lane;
reg [FEAT_AW-1:0] wl_feat;
reg               wl_phase;

reg [7:0]         bl_lane;
reg               bl_phase;

reg [FEAT_AW+1:0] mac_cnt;
reg [7:0]         sat_lane;

reg signed [DATA_W-1:0] w_lane   [0:LANES-1][0:IN_DIM-1];
reg signed [DATA_W-1:0] bias_reg [0:LANES-1];
reg signed [ACC_W-1:0]  acc      [0:LANES-1];

reg signed [DATA_W-1:0] w_mul_r  [0:LANES-1];
reg signed [DATA_W-1:0] mac_x_r;
reg signed [ACC_W-1:0]  prod_r   [0:LANES-1];

reg signed [ACC_W-1:0]  sat_mid_r;
reg                      sat_s1_valid;
reg [7:0]                sat_s1_lane;

reg signed [DATA_W-1:0]  bias_pick;       // combinational mux (declared as reg)
reg signed [ACC_W-1:0]   acc_pick_full;

`ifndef SYNTHESIS
integer init_i, init_l;
initial begin
    for (init_l = 0; init_l < LANES; init_l = init_l + 1) begin
        for (init_i = 0; init_i < IN_DIM; init_i = init_i + 1)
            w_lane[init_l][init_i] = {DATA_W{1'b0}};
        bias_reg[init_l] = {DATA_W{1'b0}};
        acc[init_l]      = {ACC_W{1'b0}};
    end
end
`endif

// -------------------------------------------------------------------------
// Wires
// -------------------------------------------------------------------------
wire        wl_done = (wl_lane == LANES - 1) && (wl_feat == FEAT_LAST) &&
                      (wl_phase == 1'b1);
wire        bl_done = (bl_lane == LANES - 1) && (bl_phase == 1'b1);
wire        mac_done = (mac_cnt == MAC_LAST);
wire        sat_done = (sat_lane == SAT_LAST);
wire        tok_last = (tok_cnt == TOK_LAST);
wire        group_last = ((neu_base + LANES) >= OUT_DIM);

// pipeline enables (see timing in header)
// posedge SRAM (CLK=clk): addr issued at mac_cnt==c yields x_q[c] one cycle later,
// i.e. when mac_cnt==c+1. So issue feat c at mac_cnt==c (0..IN_DIM-1) but CAPTURE
// {x_q=x[c], w_lane[*][c]} at mac_cnt==c+1; mul one cycle later, acc one more.
wire        x_issue  = (state == S_MAC) && (mac_cnt < IN_DIM);                        // issue feat 0..IN_DIM-1
wire        mac_x_cap = (state == S_MAC) && (mac_cnt >= 1) && (mac_cnt <= IN_DIM);    // capture x[mac_cnt-1]
wire        mul_en   = (state == S_MAC) && (mac_cnt >= 2) && (mac_cnt <= IN_DIM + 1); // 2..IN_DIM+1
wire        acc_en   = (state == S_MAC) && (mac_cnt >= 3) && (mac_cnt <= IN_DIM + 2); // 3..IN_DIM+2

// weight preload neuron index and flat addr
wire [NEU_AW-1:0] wl_neu = neu_base + wl_lane;

assign busy = (state != S_IDLE);

assign w_addr_o = (state == S_WLOAD) ?
    (wl_neu * IN_DIM + wl_feat) : {W_ADDR_W{1'b0}};

assign b_addr_o = (state == S_BLOAD) ?
    (neu_base + bl_lane) : {B_ADDR_W{1'b0}};

// -------------------------------------------------------------------------
// SAT datapath (combinational)
// -------------------------------------------------------------------------
wire signed [ACC_W-1:0] sat_shr8  = acc_pick_full >>> FRAC_BITS;
wire signed [ACC_W-1:0] sat_add_b = sat_shr8 +
    {{(ACC_W-DATA_W){bias_pick[DATA_W-1]}}, bias_pick};
wire signed [DATA_W-1:0] sat_val =
    (sat_mid_r > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
    (sat_mid_r < SAT_MIN) ? SAT_MIN[DATA_W-1:0] :
    sat_mid_r[DATA_W-1:0];

// -------------------------------------------------------------------------
// FSM segment 1: state register
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state logic
// next_state via ternary in each case branch (no nested if per style rule)
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:    next_state = start    ? S_WLOAD : state;
        S_WLOAD:   next_state = wl_done  ? S_BLOAD : state;
        S_BLOAD:   next_state = bl_done  ? S_MAC   : state;
        S_MAC:     next_state = mac_done ? S_SAT   : state;
        S_SAT:     next_state = (sat_done && !tok_last)              ? S_MAC     :
                               (sat_done && tok_last && !group_last) ? S_WLOAD   :
                               (sat_done && tok_last && group_last)  ? S_DONE_ST : state;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// -------------------------------------------------------------------------
// neu_base: 0 on start; += LANES after each group's last token
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        neu_base <= {NEU_AW{1'b0}};
    else if (state == S_IDLE && start)
        neu_base <= {NEU_AW{1'b0}};
    else if (state == S_SAT && sat_done && tok_last && !group_last)
        neu_base <= neu_base + LANES;
end

// tok_cnt: 0 at WLOAD entry / start; ++ after each token's SAT; wrap on group end
always @(posedge clk) begin
    if (reset)
        tok_cnt <= {TOK_AW{1'b0}};
    else if (state == S_IDLE)
        tok_cnt <= {TOK_AW{1'b0}};
    else if (state == S_SAT && sat_done && !tok_last)
        tok_cnt <= tok_cnt + {{(TOK_AW-1){1'b0}}, 1'b1};
    else if (state == S_SAT && sat_done && tok_last)
        tok_cnt <= {TOK_AW{1'b0}};
end

// -------------------------------------------------------------------------
// Weight preload counters (2-phase ROM read) + capture
// -------------------------------------------------------------------------
// wl_phase: toggle each WLOAD cycle (sole driver)
always @(posedge clk) begin
    if (reset)                 wl_phase <= 1'b0;
    else if (state != S_WLOAD) wl_phase <= 1'b0;
    else                       wl_phase <= ~wl_phase;
end

// wl_feat: advance on phase1; wrap at FEAT_LAST (sole driver)
always @(posedge clk) begin
    if (reset)                      wl_feat <= {FEAT_AW{1'b0}};
    else if (state != S_WLOAD)      wl_feat <= {FEAT_AW{1'b0}};
    else if (wl_phase == 1'b0)      wl_feat <= wl_feat;
    else if (wl_feat == FEAT_LAST)  wl_feat <= {FEAT_AW{1'b0}};
    else                            wl_feat <= wl_feat + {{(FEAT_AW-1){1'b0}}, 1'b1};
end

// wl_lane: +1 when a full feat row completes (phase1 & feat==LAST) (sole driver)
always @(posedge clk) begin
    if (reset)                 wl_lane <= 8'd0;
    else if (state != S_WLOAD) wl_lane <= 8'd0;
    else if (wl_phase == 1'b1 && wl_feat == FEAT_LAST && wl_lane != LANES - 1)
                               wl_lane <= wl_lane + 8'd1;
    else                       wl_lane <= wl_lane;
end

// Weight capture into reg file (phase 1 = ROM data valid for addr@phase0)
always @(posedge clk) begin
    if (state == S_WLOAD && wl_phase == 1'b1)
        w_lane[wl_lane][wl_feat] <= w_i;
end

// -------------------------------------------------------------------------
// Bias preload counters (2-phase) + capture
// -------------------------------------------------------------------------
// bl_phase: toggle each BLOAD cycle (sole driver)
always @(posedge clk) begin
    if (reset)                 bl_phase <= 1'b0;
    else if (state != S_BLOAD) bl_phase <= 1'b0;
    else                       bl_phase <= ~bl_phase;
end

// bl_lane: +1 on phase1 until LANES-1 (sole driver)
always @(posedge clk) begin
    if (reset)                 bl_lane <= 8'd0;
    else if (state != S_BLOAD) bl_lane <= 8'd0;
    else if (bl_phase == 1'b1 && bl_lane != LANES - 1)
                               bl_lane <= bl_lane + 8'd1;
    else                       bl_lane <= bl_lane;
end

always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LANES; i = i + 1)
            bias_reg[i] <= {DATA_W{1'b0}};
    end else if (state == S_BLOAD && bl_phase == 1'b1)
        bias_reg[bl_lane] <= b_i;
end

// -------------------------------------------------------------------------
// MAC counter
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        mac_cnt <= {(FEAT_AW+2){1'b0}};
    else if (state != S_MAC)
        mac_cnt <= {(FEAT_AW+2){1'b0}};
    else if (!mac_done)
        mac_cnt <= mac_cnt + {{(FEAT_AW+1){1'b0}}, 1'b1};
end

// x read port (combinational: issue feat=mac_cnt at cycle f so x_q is x[f] at the
// posedge ending cycle f, aligned with the comb w_lane[*][f] read; a registered
// addr would add a cycle and break MAC alignment).
always @(*) begin
    x_rd_en   = 1'b0;
    x_rd_addr = {X_AW{1'b0}};
    if (x_issue) begin
        x_rd_en   = 1'b1;
        x_rd_addr = tok_cnt * IN_DIM + mac_cnt;
    end
end

// x / w capture: posedge SRAM (CLK=clk) -> when mac_cnt==c+1 (c=feat), x_q holds
// x[c]; pair it with the comb reg-file read w_lane[*][c] = w_lane[*][mac_cnt-1] so
// the multiplier inputs are aligned for the same feature c.
// (the reg-file mux is cut from the multiplier path by this single capture stage.)
always @(posedge clk) begin
    if (reset) begin
        mac_x_r <= {DATA_W{1'b0}};
        for (i = 0; i < LANES; i = i + 1)
            w_mul_r[i] <= {DATA_W{1'b0}};
    end else if (mac_x_cap) begin
        mac_x_r <= x_q;
        for (i = 0; i < LANES; i = i + 1)
            w_mul_r[i] <= w_lane[i][(mac_cnt - 1'b1)];   // feat = mac_cnt-1
    end
end

// Multiply stage (reg x reg)
always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LANES; i = i + 1)
            prod_r[i] <= {ACC_W{1'b0}};
    end else if (mul_en) begin
        for (i = 0; i < LANES; i = i + 1)
            prod_r[i] <= w_mul_r[i] * mac_x_r;
    end
end

// Accumulate (cleared at mac_cnt==0 of each token)
always @(posedge clk) begin
    if (reset) begin
        for (i = 0; i < LANES; i = i + 1)
            acc[i] <= {ACC_W{1'b0}};
    end else if (state == S_MAC && mac_cnt == {(FEAT_AW+2){1'b0}}) begin
        for (i = 0; i < LANES; i = i + 1)
            acc[i] <= {ACC_W{1'b0}};
    end else if (acc_en) begin
        for (i = 0; i < LANES; i = i + 1)
            acc[i] <= acc[i] + prod_r[i];
    end
end

// -------------------------------------------------------------------------
// SAT counter
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        sat_lane <= 5'd0;
    else if (state != S_SAT)
        sat_lane <= 5'd0;
    else if (!sat_done)
        sat_lane <= sat_lane + 5'd1;
end

// acc / bias mux (combinational); acc_pick kept for waveform readability only
always @(*) begin
    acc_pick_full = {ACC_W{1'b0}};
    bias_pick     = {DATA_W{1'b0}};
    if (sat_lane < LANES) begin
        acc_pick_full = acc[sat_lane];
        bias_pick     = bias_reg[sat_lane];
    end
end

// SAT stage 1: >>>FRAC + bias -> sat_mid_r
always @(posedge clk) begin
    if (reset) begin
        sat_mid_r    <= {ACC_W{1'b0}};
        sat_s1_valid <= 1'b0;
        sat_s1_lane  <= 4'd0;
    end else if (state == S_SAT && sat_lane < LANES) begin
        sat_mid_r    <= sat_add_b;
        sat_s1_valid <= 1'b1;
        sat_s1_lane  <= sat_lane;
    end else begin
        sat_s1_valid <= 1'b0;
    end
end

// SAT stage 2: clamp -> y_o / y_valid (tok/neu tags)
always @(posedge clk) begin
    if (reset) begin
        y_o     <= {DATA_W{1'b0}};
        y_valid <= 1'b0;
        y_tok_o <= {TOK_AW{1'b0}};
        y_neu_o <= {NEU_AW{1'b0}};
    end else if (sat_s1_valid) begin
        y_o     <= sat_val;
        y_valid <= 1'b1;
        y_tok_o <= tok_cnt;
        y_neu_o <= neu_base + sat_s1_lane;
    end else begin
        y_valid <= 1'b0;
    end
end

// Done pulse
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE_ST)
        done <= 1'b1;
    else
        done <= 1'b0;
end

endmodule
