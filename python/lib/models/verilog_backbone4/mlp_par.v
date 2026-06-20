// =============================================================================
// mlp_par.v  (verilog_backbone4, dim192 / Q7.7)  -- block MLP on ONE B2 engine
//
//   mlp_out = fc2( relu( fc1(x_norm2) ) )
//     fc1 : EMBED(192) -> HIDDEN(768)   + ReLU(max(0,.))
//     fc2 : HIDDEN(768) -> EMBED(192)
//
// Reuses a SINGLE LANES x P MAC array (same proven datapath as linear_par):
//   LANES neurons in parallel, P features/cycle, pipelined adder tree.
//   The engine runs TWICE (pass1 = fc1+ReLU, pass2 = fc2).  Pass1 writes the
//   ReLU'd hidden activations into an internal token-major buffer; pass2 reads
//   that buffer as its x source.  Weight-stationary: each weight read once per
//   (pass, neuron group).
//
// Bit-exactness target = integer self-reference (truncating, NO rounding), the
//   same convention as linear_ws / linear_par:
//     h1[n1] = sat14( (SUM_f x_int*w1_int) >>> FRAC + b1_int )
//     h2[n1] = max(0, h1[n1])                          (ReLU)
//     y [n2] = sat14( (SUM_k h2*w2_int) >>> FRAC + b2_int )
//   (numpy dim192/Q7.7 golden uses float matmul + round, so it differs by a few
//    LSB; the TB reports that as info, gates PASS/FAIL on the self-ref.)
//
// x read contract (P-wide word, 1P SRAM, CLK=clk posedge, 1-cycle latency):
//   posedge T  : x_rd_addr = tok*FS1 + step       (FS1 = EMBED/P = 6)
//   posedge T+1: x_q valid for addr@T  (lane p -> x_q[p*DATA_W +: DATA_W])
// hidden buffer read (pass2) is registered with the SAME 1-cycle latency so the
//   tag-pipeline alignment (WMUL_TAP/ACC_TAP) is identical for both passes.
//
// Weight ROMs (2-phase, CLK=clk posedge): w1/b1 for fc1, w2/b2 for fc2.
//   w1_addr = (neu_base+lane)*EMBED  + feat ;  b1_addr = neu_base+lane
//   w2_addr = (neu_base+lane)*HIDDEN + feat ;  b2_addr = neu_base+lane
//
// NOTE (synthesis): the hidden buffer is modeled here as a reg array
//   (N_TOKENS*HIDDEN x DATA_W).  For the full N_TOKENS=320 design this must be
//   mapped to SRAM (token-major, P-feature words for the pass2 read port); see
//   SRAM_suggestion.mdc.  Functional v1 keeps it as a reg array.
//
// Verilog-2001 synthesizable.  P power of two; EMBED % P == 0; HIDDEN % P == 0.
// =============================================================================

module mlp_par #(
    parameter EMBED     = 192,
    parameter HIDDEN    = 768,
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
    parameter IN_MAX    = 768,  // max(EMBED, HIDDEN)
    parameter FEAT_AW   = 10,   // ceil(log2(IN_MAX))
    parameter XW_AW     = 14,   // ceil(log2(N_TOKENS*EMBED/P)) headroom
    parameter W1_AW     = 20,   // ceil(log2(HIDDEN*EMBED))
    parameter W2_AW     = 20,   // ceil(log2(EMBED*HIDDEN))
    parameter B_AW      = 10,
    parameter NEU_AW    = 10,
    parameter TOK_AW    = 10,
    parameter STEP_AW   = 5,    // ceil(log2(HIDDEN/P)) = ceil(log2(24)) = 5
    parameter HID_AW    = 20    // ceil(log2(N_TOKENS*HIDDEN))
) (
    input  wire                           clk,
    input  wire                           reset,
    input  wire                           start,

    // fc1 activation read port (P features/word; token-major; EMBED/P words/tok)
    output reg                            x_rd_en,
    output reg  [XW_AW-1:0]               x_rd_addr,
    input  wire [P*DATA_W-1:0]            x_q,

    // fc1 weight / bias ROM (2-phase)
    output wire [W1_AW-1:0]               w1_addr_o,
    input  wire signed [DATA_W-1:0]       w1_i,
    output wire [B_AW-1:0]                b1_addr_o,
    input  wire signed [DATA_W-1:0]       b1_i,

    // fc2 weight / bias ROM (2-phase)
    output wire [W2_AW-1:0]               w2_addr_o,
    input  wire signed [DATA_W-1:0]       w2_i,
    output wire [B_AW-1:0]                b2_addr_o,
    input  wire signed [DATA_W-1:0]       b2_i,

    output wire                           busy,
    output reg                            done,

    // fc2 output stream: one beat per token = LANES neuron results in parallel
    output reg  signed [LANES*DATA_W-1:0] y_o,
    output reg                            y_valid,
    output reg  [TOK_AW-1:0]              y_tok_o,
    output reg  [NEU_AW-1:0]              y_neu_base_o
);

// -------------------------------------------------------------------------
// Local params
// -------------------------------------------------------------------------
// loader (producer) states : preload weights+bias into the inactive buffer
localparam LD_IDLE = 3'd0, LD_W = 3'd1, LD_B = 3'd2, LD_ADV = 3'd3, LD_WAIT = 3'd4;
// compute (consumer) states : stream all tokens through the active buffer's MAC
localparam CP_IDLE = 3'd0, CP_WAIT = 3'd1, CP_MAC = 3'd2, CP_ADV = 3'd3, CP_DONE = 3'd4;

localparam integer FS1 = EMBED  / P;   // 6  steps/token (fc1)
localparam integer FS2 = HIDDEN / P;   // 24 steps/token (fc2)

// hidden SRAM geometry: token-major, 2 banks x (LANES features = 256-bit word).
//   bank0 = lower 16 of each 32-feature group, bank1 = upper 16.
//   reading both banks at the same word addr -> 32 features/cycle (= P).
localparam integer BANK_W    = LANES * DATA_W;     // 256
localparam integer HID_WORDS = N_TOKENS * FS2;     // words per bank (= tok*24)

localparam [TOK_AW-1:0] TOK_LAST = N_TOKENS - 1;

// pipeline taps (issue -> stage). identical to linear_par:
//   tg[n] holds issue-k info during cycle k+1+n; xreg/wreg cap at k+1 (tg[0]);
//   prod@k+3; a1@k+4 .. a5@k+8; acc/emit consume a5 during k+8 using tg[7].
localparam integer WMUL_TAP = 0;
localparam integer ACC_TAP  = 7;
localparam integer TAGD     = 8;

integer gi, gj, gl, gp;

// -------------------------------------------------------------------------
// Registers
// -------------------------------------------------------------------------
// ===================  SCHEME A: weight double-buffer overlap  ============
// Two concurrent engines share the proven MAC datapath:
//   LOADER  (producer) : 2-phase weight+bias preload into the INACTIVE buffer
//   COMPUTE (consumer) : streams all N_TOKENS through the ACTIVE buffer's MAC
// Each group's weight load is hidden under the previous group's MAC.  Ping-pong
// buffers (0/1) + buf_filled[] handshake keep the loader exactly 1 group ahead.
// The weight ROM is driven ONLY by the loader; x / hidden SRAM ONLY by compute
// -> no shared-port conflict.  Functional result identical to the serial
// version (emit order + tags unchanged, TB self-ref untouched).
// =========================================================================

// loader FSM
reg [2:0]         ld_state, ld_next;
reg               ld_pass;        // 1 = fc1 weights, 0 = fc2 weights
reg [NEU_AW-1:0]  ld_neu_base;
reg               ld_buf;         // buffer being filled (0/1)

// compute FSM
reg [2:0]         cp_state, cp_next;
reg               cp_pass;        // 1 = fc1 pass, 0 = fc2 pass
reg [NEU_AW-1:0]  cp_neu_base;
reg               cp_buf;         // buffer being consumed (0/1)

// ping-pong "filled" handshake (set by loader at LD_ADV, cleared by compute at CP_ADV)
reg               buf_filled0, buf_filled1;

// weight/bias preload counters (2-phase, loader)
reg [NEU_AW-1:0]  wl_lane;
reg [FEAT_AW-1:0] wl_feat;
reg               wl_phase;
reg [NEU_AW-1:0]  bl_lane;
reg               bl_phase;

// MAC issue counters (compute)
reg [TOK_AW-1:0]  iss_tok;
reg [STEP_AW-1:0] iss_step;
reg [TOK_AW-1:0]  mac_done_cnt;
reg               mac_run;

// bias storage : one set per buffer (1D arrays, no 3D reg).  NOTE: per-lane
// w_buf0/w_buf1 / wreg / prod_r / adder-tree regs (a1..a4) live INSIDE the
// generate block as per-lane 1D arrays -- NO module has a [LANES][..] array.
reg signed [DATA_W-1:0] bias_reg0 [0:LANES-1];
reg signed [DATA_W-1:0] bias_reg1 [0:LANES-1];

// hidden activation buffer -> modeled as 1P SRAM (inferable RAM, 1-cycle read).
//   pass1 = WRITE only (ReLU emit) ; pass2 = READ only (fc2 x source).  pass1
//   fully precedes pass2, so a single port per bank is sufficient (no R/W same
//   cycle).  Synthesis infers RAM; replace with a depth-HID_WORDS x BANK_W PDK
//   macro at backend (see SRAM_suggestion.mdc).  No 3D reg (each bank is 2D).
reg [BANK_W-1:0]   hid_bank0 [0:HID_WORDS-1];   // features [s*32 +  0 .. +15]
reg [BANK_W-1:0]   hid_bank1 [0:HID_WORDS-1];   // features [s*32 + 16 .. +31]
reg [P*DATA_W-1:0] hid_word_r;   // registered pass2 read word (1-cycle latency)

// datapath shared regs (1D only): x word, per-lane partial sum & accumulator
reg signed [DATA_W-1:0] xreg     [0:P-1];
reg signed [TREE_W-1:0] a5       [0:LANES-1];
reg signed [ACC_W-1:0]  acc      [0:LANES-1];

// tag pipeline
reg               tg_val   [0:TAGD-1];
reg               tg_first [0:TAGD-1];
reg               tg_last  [0:TAGD-1];
reg [TOK_AW-1:0]  tg_tok   [0:TAGD-1];
reg [STEP_AW-1:0] tg_step  [0:TAGD-1];

`ifndef SYNTHESIS
integer ii;
initial begin
    for (ii = 0; ii < LANES; ii = ii + 1) begin
        bias_reg0[ii] = {DATA_W{1'b0}};
        bias_reg1[ii] = {DATA_W{1'b0}};
        acc[ii]       = {ACC_W{1'b0}};
    end
    for (ii = 0; ii < HID_WORDS; ii = ii + 1) begin
        hid_bank0[ii] = {BANK_W{1'b0}};
        hid_bank1[ii] = {BANK_W{1'b0}};
    end
end
`endif

// -------------------------------------------------------------------------
// Per-engine dimension selects (combinational)
//   fc1(pass=1): in=EMBED  out=HIDDEN fs=FS1 ; fc2(pass=0): in=HIDDEN out=EMBED fs=FS2
// -------------------------------------------------------------------------
// loader-side (drives weight ROM + feat counter)
wire [FEAT_AW-1:0] ld_in_dim   = ld_pass ? EMBED[FEAT_AW-1:0] : HIDDEN[FEAT_AW-1:0];
wire [FEAT_AW-1:0] ld_feat_last = ld_in_dim - 1'b1;
wire [NEU_AW-1:0]  ld_out_dim  = ld_pass ? HIDDEN[NEU_AW-1:0] : EMBED[NEU_AW-1:0];
wire ld_grp_last = ((ld_neu_base + LANES) >= ld_out_dim);
wire ld_is_last  = (!ld_pass) && ld_grp_last;     // just loaded the final group
// loader next group (after current load completes)
wire               ld_nxt_pass = ld_pass && !ld_grp_last;     // fc1 -> fc1 until last, then fc2
wire [NEU_AW-1:0]  ld_nxt_nb   = (ld_pass && ld_grp_last) ? {NEU_AW{1'b0}}
                                                          : (ld_neu_base + LANES);
// compute-side (drives MAC step/token + group)
wire [STEP_AW-1:0] cp_fs        = cp_pass ? FS1[STEP_AW-1:0] : FS2[STEP_AW-1:0];
wire [STEP_AW-1:0] cp_step_last = cp_fs - 1'b1;
wire [NEU_AW-1:0]  cp_out_dim   = cp_pass ? HIDDEN[NEU_AW-1:0] : EMBED[NEU_AW-1:0];
wire cp_grp_last = ((cp_neu_base + LANES) >= cp_out_dim);
wire cp_is_last  = (!cp_pass) && cp_grp_last;      // last fc2 group = whole MLP done
wire               cp_nxt_pass = cp_pass && !cp_grp_last;
wire [NEU_AW-1:0]  cp_nxt_nb   = (cp_pass && cp_grp_last) ? {NEU_AW{1'b0}}
                                                          : (cp_neu_base + LANES);

// -------------------------------------------------------------------------
// Control wires
// -------------------------------------------------------------------------
wire wl_done = (wl_lane == LANES-1) && (wl_feat == ld_feat_last) && (wl_phase == 1'b1);
wire bl_done = (bl_lane == LANES-1) && (bl_phase == 1'b1);
wire iss_last = (iss_tok == TOK_LAST) && (iss_step == cp_step_last);
wire mac_all_emitted = (mac_done_cnt == N_TOKENS);

// buffer handshake selects
wire ld_tgt_filled = ld_buf ? buf_filled1 : buf_filled0;   // target buffer busy?
wire cp_src_filled = cp_buf ? buf_filled1 : buf_filled0;    // active buffer ready?

assign busy = (cp_state != CP_IDLE) || (ld_state != LD_IDLE);

wire [NEU_AW-1:0] wl_neu = ld_neu_base + wl_lane;

// weight/bias ROM addresses : driven ONLY by the loader engine
assign w1_addr_o = (ld_state == LD_W &&  ld_pass) ? (wl_neu * EMBED  + wl_feat) : {W1_AW{1'b0}};
assign w2_addr_o = (ld_state == LD_W && !ld_pass) ? (wl_neu * HIDDEN + wl_feat) : {W2_AW{1'b0}};
assign b1_addr_o = (ld_state == LD_B &&  ld_pass) ? (ld_neu_base + bl_lane) : {B_AW{1'b0}};
assign b2_addr_o = (ld_state == LD_B && !ld_pass) ? (ld_neu_base + bl_lane) : {B_AW{1'b0}};

wire signed [DATA_W-1:0] w_i_sel = ld_pass ? w1_i : w2_i;
wire signed [DATA_W-1:0] b_i_sel = ld_pass ? b1_i : b2_i;

// tag-delayed control
wire [STEP_AW-1:0] wmul_step = tg_step[WMUL_TAP];
wire               wmul_val  = tg_val[WMUL_TAP];
wire               acc_val   = tg_val[ACC_TAP];
wire               acc_first = tg_first[ACC_TAP];
wire               acc_last  = tg_last[ACC_TAP];
wire [TOK_AW-1:0]  acc_tok   = tg_tok[ACC_TAP];

// x source mux: fc1 = external SRAM word, fc2 = registered hidden word
wire [P*DATA_W-1:0] x_in_word = cp_pass ? x_q : hid_word_r;

// =========================================================================
// FSM 1 : LOADER (producer) -- preload weights+bias of the next group into the
// inactive buffer.  LD_W (2-phase weights) -> LD_B (2-phase bias) -> LD_ADV
// (publish buffer) -> LD_WAIT (until next target buffer is free) -> LD_W ...
// =========================================================================
always @(posedge clk) begin
    if (reset) ld_state <= LD_IDLE;
    else       ld_state <= ld_next;
end

always @(*) begin
    ld_next = ld_state;
    case (ld_state)
        LD_IDLE: if (start)          ld_next = LD_W;     // first group (buffer free)
        LD_W:    if (wl_done)        ld_next = LD_B;
        LD_B:    if (bl_done)        ld_next = LD_ADV;
        LD_ADV:                      ld_next = ld_is_last ? LD_IDLE : LD_WAIT;
        LD_WAIT: if (!ld_tgt_filled) ld_next = LD_W;     // next target buffer free
        default:                     ld_next = LD_IDLE;
    endcase
end

// loader group pointer + target buffer (advance at LD_ADV)
always @(posedge clk) begin
    if (reset) begin
        ld_pass <= 1'b1; ld_neu_base <= {NEU_AW{1'b0}}; ld_buf <= 1'b0;
    end else if (ld_state == LD_IDLE && start) begin
        ld_pass <= 1'b1; ld_neu_base <= {NEU_AW{1'b0}}; ld_buf <= 1'b0;
    end else if (ld_state == LD_ADV) begin
        ld_pass     <= ld_nxt_pass;
        ld_neu_base <= ld_nxt_nb;
        ld_buf      <= ~ld_buf;
    end
end

// =========================================================================
// FSM 2 : COMPUTE (consumer) -- stream all tokens through the active buffer.
// CP_WAIT (until active buffer ready) -> CP_MAC (issue + drain) -> CP_ADV
// (release buffer, advance) -> CP_WAIT ... -> CP_DONE after last fc2 group.
// =========================================================================
always @(posedge clk) begin
    if (reset) cp_state <= CP_IDLE;
    else       cp_state <= cp_next;
end

always @(*) begin
    cp_next = cp_state;
    case (cp_state)
        CP_IDLE: if (start)           cp_next = CP_WAIT;
        CP_WAIT: if (cp_src_filled)   cp_next = CP_MAC;
        CP_MAC:  if (mac_all_emitted) cp_next = CP_ADV;
        CP_ADV:                       cp_next = cp_is_last ? CP_DONE : CP_WAIT;
        CP_DONE:                      cp_next = CP_IDLE;
        default:                      cp_next = CP_IDLE;
    endcase
end

// compute group pointer + active buffer (advance at CP_ADV)
always @(posedge clk) begin
    if (reset) begin
        cp_pass <= 1'b1; cp_neu_base <= {NEU_AW{1'b0}}; cp_buf <= 1'b0;
    end else if (cp_state == CP_IDLE && start) begin
        cp_pass <= 1'b1; cp_neu_base <= {NEU_AW{1'b0}}; cp_buf <= 1'b0;
    end else if (cp_state == CP_ADV) begin
        cp_pass     <= cp_nxt_pass;
        cp_neu_base <= cp_nxt_nb;
        cp_buf      <= ~cp_buf;
    end
end

// -------------------------------------------------------------------------
// Ping-pong handshake : loader SETS filled at LD_ADV, compute CLEARS at CP_ADV.
// ld_buf and cp_buf stay opposite while both run -> never touch same buffer.
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        buf_filled0 <= 1'b0;
        buf_filled1 <= 1'b0;
    end else begin
        if      (ld_state == LD_ADV && ld_buf == 1'b0) buf_filled0 <= 1'b1;
        else if (cp_state == CP_ADV && cp_buf == 1'b0) buf_filled0 <= 1'b0;
        if      (ld_state == LD_ADV && ld_buf == 1'b1) buf_filled1 <= 1'b1;
        else if (cp_state == CP_ADV && cp_buf == 1'b1) buf_filled1 <= 1'b0;
    end
end

// -------------------------------------------------------------------------
// Weight preload counters (2-phase, active in LD_W)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)                  wl_phase <= 1'b0;
    else if (ld_state != LD_W)  wl_phase <= 1'b0;
    else                        wl_phase <= ~wl_phase;
end

always @(posedge clk) begin
    if (reset)                        wl_feat <= {FEAT_AW{1'b0}};
    else if (ld_state != LD_W)        wl_feat <= {FEAT_AW{1'b0}};
    else if (wl_phase == 1'b0)        wl_feat <= wl_feat;
    else if (wl_feat == ld_feat_last) wl_feat <= {FEAT_AW{1'b0}};
    else                              wl_feat <= wl_feat + 1'b1;
end

always @(posedge clk) begin
    if (reset)                  wl_lane <= {NEU_AW{1'b0}};
    else if (ld_state != LD_W)  wl_lane <= {NEU_AW{1'b0}};
    else if (wl_phase == 1'b1 && wl_feat == ld_feat_last && wl_lane != LANES-1)
                                wl_lane <= wl_lane + 1'b1;
    else                        wl_lane <= wl_lane;
end

// NOTE: w_buf0/w_buf1 writes live in the per-lane generate block (self-select
// by wl_lane==L and ld_buf) to avoid a 3D reg array.

// -------------------------------------------------------------------------
// Bias preload counters (2-phase, active in LD_B)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)                  bl_phase <= 1'b0;
    else if (ld_state != LD_B)  bl_phase <= 1'b0;
    else                        bl_phase <= ~bl_phase;
end

always @(posedge clk) begin
    if (reset)                  bl_lane <= {NEU_AW{1'b0}};
    else if (ld_state != LD_B)  bl_lane <= {NEU_AW{1'b0}};
    else if (bl_phase == 1'b1 && bl_lane != LANES-1)
                                bl_lane <= bl_lane + 1'b1;
    else                        bl_lane <= bl_lane;
end

// bias write into the loader's target buffer (single driver per bias array)
always @(posedge clk) begin
    if (reset) begin
        for (gi = 0; gi < LANES; gi = gi + 1) begin
            bias_reg0[gi] <= {DATA_W{1'b0}};
            bias_reg1[gi] <= {DATA_W{1'b0}};
        end
    end else if (ld_state == LD_B && bl_phase == 1'b1) begin
        if (ld_buf == 1'b0) bias_reg0[bl_lane] <= b_i_sel;
        else                bias_reg1[bl_lane] <= b_i_sel;
    end
end

// -------------------------------------------------------------------------
// MAC issue counters (compute, step inner / token outer)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)                                     mac_run <= 1'b0;
    else if (cp_state == CP_WAIT && cp_src_filled) mac_run <= 1'b1;  // entering CP_MAC
    else if (cp_state == CP_MAC  && iss_last)      mac_run <= 1'b0;
    else if (cp_state != CP_MAC)                   mac_run <= 1'b0;
end

always @(posedge clk) begin
    if (reset)                          iss_step <= {STEP_AW{1'b0}};
    else if (cp_state != CP_MAC)        iss_step <= {STEP_AW{1'b0}};
    else if (!mac_run)                  iss_step <= {STEP_AW{1'b0}};
    else if (iss_step == cp_step_last)  iss_step <= {STEP_AW{1'b0}};
    else                                iss_step <= iss_step + 1'b1;
end

always @(posedge clk) begin
    if (reset)                                                 iss_tok <= {TOK_AW{1'b0}};
    else if (cp_state != CP_MAC)                               iss_tok <= {TOK_AW{1'b0}};
    else if (!mac_run)                                         iss_tok <= {TOK_AW{1'b0}};
    else if (iss_step == cp_step_last && iss_tok != TOK_LAST)  iss_tok <= iss_tok + 1'b1;
    else if (iss_step == cp_step_last)                         iss_tok <= {TOK_AW{1'b0}};
end

// emitted-token counter
always @(posedge clk) begin
    if (reset)                    mac_done_cnt <= {TOK_AW{1'b0}};
    else if (cp_state != CP_MAC)  mac_done_cnt <= {TOK_AW{1'b0}};
    else if (acc_val && acc_last) mac_done_cnt <= mac_done_cnt + 1'b1;
end

// -------------------------------------------------------------------------
// x read: fc1 drives external SRAM addr; fc2 reads hidden SRAM (registered)
// -------------------------------------------------------------------------
always @(*) begin
    x_rd_en   = 1'b0;
    x_rd_addr = {XW_AW{1'b0}};
    if (cp_state == CP_MAC && mac_run && cp_pass) begin
        x_rd_en   = 1'b1;
        x_rd_addr = iss_tok * FS1 + iss_step;
    end
end

// pass2 hidden read: 1P SRAM, addr = tok*FS2 + step, both banks lockstep ->
// 32 features/cycle.  Registered Q (1-cycle latency, same as external x SRAM):
//   low 256b  = bank0 = features [step*32 +  0 .. +15]
//   high 256b = bank1 = features [step*32 + 16 .. +31]
always @(posedge clk) begin : hidrd
    if (cp_state == CP_MAC && mac_run && !cp_pass) begin
        hid_word_r[0      +: BANK_W] <= hid_bank0[iss_tok * FS2 + iss_step];
        hid_word_r[BANK_W +: BANK_W] <= hid_bank1[iss_tok * FS2 + iss_step];
    end
end

// -------------------------------------------------------------------------
// Tag pipeline
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
        tg_val[0]   <= (cp_state == CP_MAC) && mac_run;
        tg_first[0] <= (iss_step == {STEP_AW{1'b0}});
        tg_last[0]  <= (iss_step == cp_step_last);
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
// Datapath: x capture (whole P-feat word, 1 cycle after issue)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        for (gp = 0; gp < P; gp = gp + 1) xreg[gp] <= {DATA_W{1'b0}};
    end else if (tg_val[0]) begin
        for (gp = 0; gp < P; gp = gp + 1)
            xreg[gp] <= x_in_word[gp*DATA_W +: DATA_W];
    end
end

// -------------------------------------------------------------------------
// Per-lane datapath (generate): wreg, prod, adder tree, acc
// -------------------------------------------------------------------------
genvar L;
generate
for (L = 0; L < LANES; L = L + 1) begin : LANE

    // per-lane storage / pipeline regs (ALL 1D : packed vector + 1 unpacked dim).
    // DOUBLE BUFFER (scheme A): w_buf0/w_buf1 -- loader fills the inactive one
    // (ld_buf) while compute reads the active one (cp_buf).
    reg signed [DATA_W-1:0] w_buf0 [0:IN_MAX-1];   // this lane's weights, buffer 0
    reg signed [DATA_W-1:0] w_buf1 [0:IN_MAX-1];   // this lane's weights, buffer 1
    reg signed [DATA_W-1:0] wreg   [0:P-1];
    reg signed [PROD_W-1:0] prod_r [0:P-1];
    reg signed [TREE_W-1:0] a1     [0:15];
    reg signed [TREE_W-1:0] a2     [0:7];
    reg signed [TREE_W-1:0] a3     [0:3];
    reg signed [TREE_W-1:0] a4     [0:1];

`ifndef SYNTHESIS
    integer iw;
    initial for (iw = 0; iw < IN_MAX; iw = iw + 1) begin
        w_buf0[iw] = {DATA_W{1'b0}};
        w_buf1[iw] = {DATA_W{1'b0}};
    end
`endif

    // weight load (2-phase): this lane self-selects by wl_lane==L, into ld_buf.
    // Each w_buf0/w_buf1 is written only here (single driver per buffer).
    always @(posedge clk) begin : wload
        if (ld_state == LD_W && wl_phase == 1'b1 && wl_lane == L) begin
            if (ld_buf == 1'b0) w_buf0[wl_feat] <= w_i_sel;
            else                w_buf1[wl_feat] <= w_i_sel;
        end
    end

    // wcap: read the ACTIVE buffer (cp_buf) at wmul_step*P+p, aligned with xreg
    always @(posedge clk) begin : wcap
        integer p;
        if (reset) begin
            for (p = 0; p < P; p = p + 1) wreg[p] <= {DATA_W{1'b0}};
        end else if (wmul_val) begin
            if (cp_buf == 1'b0)
                for (p = 0; p < P; p = p + 1) wreg[p] <= w_buf0[wmul_step*P + p];
            else
                for (p = 0; p < P; p = p + 1) wreg[p] <= w_buf1[wmul_step*P + p];
        end
    end

    always @(posedge clk) begin : mul
        integer p;
        if (reset) begin
            for (p = 0; p < P; p = p + 1) prod_r[p] <= {PROD_W{1'b0}};
        end else begin
            for (p = 0; p < P; p = p + 1)
                prod_r[p] <= wreg[p] * xreg[p];
        end
    end

    always @(posedge clk) begin : t1
        integer j;
        for (j = 0; j < 16; j = j + 1)
            a1[j] <= {{(TREE_W-PROD_W){prod_r[2*j][PROD_W-1]}}, prod_r[2*j]} +
                     {{(TREE_W-PROD_W){prod_r[2*j+1][PROD_W-1]}}, prod_r[2*j+1]};
    end
    always @(posedge clk) begin : t2
        integer j;
        for (j = 0; j < 8; j = j + 1) a2[j] <= a1[2*j] + a1[2*j+1];
    end
    always @(posedge clk) begin : t3
        integer j;
        for (j = 0; j < 4; j = j + 1) a3[j] <= a2[2*j] + a2[2*j+1];
    end
    always @(posedge clk) begin : t4
        integer j;
        for (j = 0; j < 2; j = j + 1) a4[j] <= a3[2*j] + a3[2*j+1];
    end
    always @(posedge clk) begin : t5
        a5[L] <= a4[0] + a4[1];
    end

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
// Output SAT (combinational).  Keep operands SIGNED so the >>> stays
// ARITHMETIC (an unsigned concat would demote it to logical shift and blow
// up negative sums to +SAT_MAX -- see linear_par.v note).
// -------------------------------------------------------------------------
reg signed [ACC_W-1:0]  sum_full   [0:LANES-1];
reg signed [ACC_W-1:0]  shr_b      [0:LANES-1];
reg signed [DATA_W-1:0] sat_lane_w [0:LANES-1];
reg signed [DATA_W-1:0] relu_lane  [0:LANES-1];
reg signed [DATA_W-1:0] bias_mux   [0:LANES-1];   // active-buffer bias (cp_buf)

always @(*) begin
    for (gl = 0; gl < LANES; gl = gl + 1) begin
        bias_mux[gl] = cp_buf ? bias_reg1[gl] : bias_reg0[gl];
        sum_full[gl] = acc[gl] + {{(ACC_W-TREE_W){a5[gl][TREE_W-1]}}, a5[gl]};
        shr_b[gl]    = $signed(sum_full[gl] >>> FRAC_BITS) +
                       $signed({{(ACC_W-DATA_W){bias_mux[gl][DATA_W-1]}}, bias_mux[gl]});
        sat_lane_w[gl] = (shr_b[gl] > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
                         (shr_b[gl] < SAT_MIN) ? SAT_MIN[DATA_W-1:0] :
                         shr_b[gl][DATA_W-1:0];
        // ReLU for fc1 (pass1): clamp negatives to 0
        relu_lane[gl] = (sat_lane_w[gl][DATA_W-1]) ? {DATA_W{1'b0}} : sat_lane_w[gl];
    end
end

// -------------------------------------------------------------------------
// Emit: pass1 -> write ReLU'd result into hidden SRAM (no y_valid);
//       pass2 -> drive external y_o stream.
//
// pass1 emits LANES(=16) consecutive neurons [neu_base .. neu_base+15] per
// token.  neu_base is a multiple of 16, so the 16 results land in exactly ONE
// bank (bank = neu_base[4]) as a FULL 256-bit word -> no bit-write enable
// needed.  word addr = acc_tok*FS2 + (neu_base>>5).  Each bank is written only
// here (single driver); banks are read only in pass2 (no R/W same cycle).
// -------------------------------------------------------------------------
always @(posedge clk) begin : hidwr
    integer l;
    if (acc_val && acc_last && cp_pass) begin
        if (cp_neu_base[4] == 1'b0) begin
            for (l = 0; l < LANES; l = l + 1)
                hid_bank0[acc_tok * FS2 + (cp_neu_base >> 5)][l*DATA_W +: DATA_W]
                    <= relu_lane[l];
        end else begin
            for (l = 0; l < LANES; l = l + 1)
                hid_bank1[acc_tok * FS2 + (cp_neu_base >> 5)][l*DATA_W +: DATA_W]
                    <= relu_lane[l];
        end
    end
end

always @(posedge clk) begin
    if (reset) begin
        y_o          <= {(LANES*DATA_W){1'b0}};
        y_valid      <= 1'b0;
        y_tok_o      <= {TOK_AW{1'b0}};
        y_neu_base_o <= {NEU_AW{1'b0}};
    end else if (acc_val && acc_last && !cp_pass) begin
        for (gl = 0; gl < LANES; gl = gl + 1)
            y_o[gl*DATA_W +: DATA_W] <= sat_lane_w[gl];
        y_valid      <= 1'b1;
        y_tok_o      <= acc_tok;
        y_neu_base_o <= cp_neu_base;
    end else begin
        y_valid <= 1'b0;
    end
end

// done pulse
always @(posedge clk) begin
    if (reset)                   done <= 1'b0;
    else if (cp_state == CP_DONE) done <= 1'b1;
    else                         done <= 1'b0;
end

endmodule
