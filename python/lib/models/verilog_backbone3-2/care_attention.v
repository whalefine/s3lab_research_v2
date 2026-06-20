// =============================================================================
// care_attention.v  (WS rewrite -- methods A+B+F+G+H)
//
// Monolithic Q8.8 CARE attention replacing 6-submodule pipeline with:
//   A  Weight-Stationary QKV    (eliminates vec_mac8 2-phase bottleneck)
//   B  Weight-Stationary PROJ   (same)
//   F  Token-Stationary KV      (8-way outer product, ~86 % KV reduction)
//   G  Fused QKV SPLIT          (relu6(rnd_shr8(y*S_Q88)) in SAT pipeline)
//   H  Fused K_MEAN into KV     (accumulate km while computing outer product)
//
// Flow: WS-QKV(+G) -> KV(+H) -> KV_SCALE -> QKM -> ZR -> CORE
//                    -> WS-PROJ(B) -> OUT
//
// SRAM/ROM read contract (1P, CLK=clk, posedge registered-data):
//   posedge T:   addr, CEB=0, WEB=1
//   posedge T+1: Q valid for addr@T  (macro Q registered on posedge clk)
// Every read path is retimed so the addr is presented one cycle before the consume:
//   S_MAC : norm_rd / sram_ao addr are COMBINATIONAL (feat=mac_cnt @T -> Q@T+1).
//   S_QKM : even cnt issues sram_q addr; odd cnt MACs on macro Q sram_q_q_i directly.
//   S_KV  : KV_RD extended to cnt 9; k/v_buf captured at cnt 2..9 (addr cnt 1..8).
//   S_CORE: sram_q addr @AT_Q_ADDR (cap @AT_Q_CAP); sram_qkm addr @AT_ZR_ADDR.
//   S_ZR  : addr @zr_sub==1; zr_x_r captured @zr_sub==2 (4 sub-states).
//   WLOAD/BLOAD/OUT : addr stable across 2 phases, consumed on the 2nd -> unchanged.
//
// SRAMs (single-port, CLK=clk, posedge 1-cycle read latency):
//   sram_q   : Q write (QKV), QKM/CORE read, PROJ write (reuse), OUT read
//   sram_k   : K write (QKV), KV read
//   sram_v   : V write (QKV), KV read
//   sram_qkm : QKM write, ZR read/write, CORE read (zr values)
//   sram_ao  : CORE write (ao), PROJ read (combinational addr)
//
// wgt_addr_o (13-bit): QKV 0..3071, PROJ 3072..4095 (same as original)
// bias_addr_o (8-bit): new top-level port (QKV 0..95, PROJ 0..31)
// =============================================================================

module care_attention #(
    parameter EMBED_DIM    = 32,
    parameter NUM_HEADS    = 4,
    parameter HEAD_DIM     = 8,
    parameter N_TOKENS     = 320,
    parameter S_Q88        = 152,
    parameter RELU6_MAX    = 1536,
    parameter RCP_N_NUM    = 205,
    parameter RCP_N_SHIFT  = 16,
    parameter KV_Q88_ROUND = 8388608
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output reg         norm_rd_en,
    output reg [13:0]  norm_rd_flat,
    input  wire signed [15:0] norm_x,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [12:0] wgt_addr_o,
    output wire [7:0]  bias_addr_o,

    output wire        busy,
    output reg         done,

    output reg  signed [15:0] y_o,
    output reg                y_valid,
    output reg  [6:0]         y_neu_o,
    output reg  [8:0]         px_tok_o,

    output reg         sram_q_ceb_o,
    output reg         sram_q_web_o,
    output reg  [13:0] sram_q_addr_o,
    output reg  [15:0] sram_q_din_o,
    input  wire [15:0] sram_q_q_i,

    output reg         sram_k_ceb_o,
    output reg         sram_k_web_o,
    output reg  [13:0] sram_k_addr_o,
    output reg  [15:0] sram_k_din_o,
    input  wire [15:0] sram_k_q_i,

    output reg         sram_v_ceb_o,
    output reg         sram_v_web_o,
    output reg  [13:0] sram_v_addr_o,
    output reg  [15:0] sram_v_din_o,
    input  wire [15:0] sram_v_q_i,

    output reg         sram_qkm_ceb_o,
    output reg         sram_qkm_web_o,
    output reg  [13:0] sram_qkm_addr_o,
    output reg  [15:0] sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i,

    output reg         sram_ao_ceb_o,
    output reg         sram_ao_web_o,
    output reg  [13:0] sram_ao_addr_o,
    output reg  [15:0] sram_ao_din_o,
    input  wire [15:0] sram_ao_q_i
);

// =========================================================================
// Parameters / localparam
// =========================================================================
localparam LANES      = 8;
localparam IN_DIM     = EMBED_DIM;
localparam QKV_GROUPS = 12;            // 96/8
localparam PROJ_GROUPS = 4;            // 32/8
localparam WL_MAX     = LANES * IN_DIM - 1;   // 255
localparam OUT_TOTAL  = N_TOKENS * EMBED_DIM;  // 10240
localparam KV_ELEMS   = NUM_HEADS * HEAD_DIM * HEAD_DIM; // 256
localparam NTxHD      = N_TOKENS * HEAD_DIM;   // 2560

parameter S_IDLE     = 4'd0;
parameter S_WLOAD    = 4'd1;
parameter S_BLOAD    = 4'd2;
parameter S_MAC      = 4'd3;
parameter S_SAT      = 4'd4;
parameter S_KV       = 4'd5;
parameter S_KV_SCALE = 4'd6;
parameter S_QKM      = 4'd7;
parameter S_ZR       = 4'd8;
parameter S_CORE     = 4'd9;
parameter S_OUT      = 4'd10;
parameter S_DONE_ST  = 4'd11;

localparam KV_CLEAR = 2'd0;
localparam KV_RD    = 2'd1;
localparam KV_MAC   = 2'd2;

localparam AT_Q_ADDR  = 3'd0;
localparam AT_Q_CAP   = 3'd1;
localparam AT_ZR_ADDR = 3'd2;
localparam AT_ZR_CAP  = 3'd3;
localparam AT_MAC8    = 3'd4;
localparam AT_DOT     = 3'd5;
localparam AT_AO      = 3'd6;

// =========================================================================
// Registers
// =========================================================================
reg [3:0] state, next_state;
reg       ws_phase;          // 0=QKV, 1=PROJ
reg [3:0] group_cnt;
reg [8:0] tok_cnt;

// WS weight preload
reg [7:0] wl_cnt;
reg       wl_phase;

// WS bias preload
reg [2:0] bl_cnt;
reg       bl_phase;

// WS MAC
reg [5:0] mac_cnt;

// WS weight register files (8 lanes x 32)
reg signed [15:0] w_lane0 [0:IN_DIM-1];
reg signed [15:0] w_lane1 [0:IN_DIM-1];
reg signed [15:0] w_lane2 [0:IN_DIM-1];
reg signed [15:0] w_lane3 [0:IN_DIM-1];
reg signed [15:0] w_lane4 [0:IN_DIM-1];
reg signed [15:0] w_lane5 [0:IN_DIM-1];
reg signed [15:0] w_lane6 [0:IN_DIM-1];
reg signed [15:0] w_lane7 [0:IN_DIM-1];

reg signed [15:0] bias_reg [0:7];
reg signed [31:0] acc      [0:7];
reg signed [31:0] prod_r   [0:7];

reg signed [15:0] w_rd_r0, w_rd_r1, w_rd_r2, w_rd_r3;
reg signed [15:0] w_rd_r4, w_rd_r5, w_rd_r6, w_rd_r7;

// SAT pipeline
reg [3:0]  sat_lane;
reg signed [31:0] sat_mid_r;
reg        sat_s1_valid;
reg [2:0]  sat_s1_lane;
reg signed [15:0] sat_val_r;
reg        sat_wr_pending;
reg [2:0]  sat_wr_lane;

reg signed [31:0] acc_pick;
reg signed [15:0] bias_pick;

// PROJ sram_ao read: CLK(clk) combinational address (feat=mac_cnt). No registered
// control / prefetch needed -- addr@T -> macro Q@T+1 lines up with the PROJ MAC the
// same way as the QKV norm path.

// S_OUT
reg [13:0] out_cnt;
reg        out_phase;

// S_KV
reg [1:0]  kv_head;
reg [8:0]  kv_tok;
reg [1:0]  kv_sub;
reg [3:0]  kv_rd_cnt;
reg [2:0]  kv_mac_dk;
reg signed [15:0] k_buf [0:7];
reg signed [15:0] v_buf [0:7];
reg signed [47:0] kv_acc [0:63];
reg signed [31:0] km_acc [0:7];
reg signed [15:0] kv_buf [0:KV_ELEMS-1];
reg signed [15:0] km_buf [0:31];

// S_KV_SCALE
reg [6:0] kvs_idx;

// S_QKM (2-phase per d: even=READ addr, odd=MAC on registered macro Q; 17 beats per h,n)
reg [1:0]  qkm_h;
reg [8:0]  qkm_n;
reg [4:0]  qkm_cnt;
reg signed [31:0] qkm_acc;

// S_ZR
reg [1:0]  zr_sub;
reg [1:0]  zr_h;
reg [8:0]  zr_n;
reg        zr_start_r;
reg signed [15:0] zr_x_r;

// S_CORE
reg [2:0]  at_phase;
reg [2:0]  q_load_dk;
reg [1:0]  at_h;
reg [8:0]  at_n;
reg [2:0]  at_dout;
reg signed [48:0] at_acc;
reg signed [15:0] core_zr_r;
reg signed [15:0] at_dot_sat_r;
reg signed [15:0] q_buf [0:7];

// QKM write registration (SRAM timing: qkm_cnt resets before negedge)
reg        qkm_wr_pending;
reg [10:0] qkm_wr_addr_r;
reg [15:0] qkm_wr_data_r;

// ZR write registration (SRAM timing: zr_sub resets before negedge)
reg        zr_wr_pending;
reg [10:0] zr_wr_addr_r;
reg [15:0] zr_wr_data_r;

integer i_lane;

`ifndef SYNTHESIS
integer init_i;
initial begin
    for (init_i = 0; init_i < IN_DIM; init_i = init_i + 1) begin
        w_lane0[init_i] = 16'sd0; w_lane1[init_i] = 16'sd0;
        w_lane2[init_i] = 16'sd0; w_lane3[init_i] = 16'sd0;
        w_lane4[init_i] = 16'sd0; w_lane5[init_i] = 16'sd0;
        w_lane6[init_i] = 16'sd0; w_lane7[init_i] = 16'sd0;
    end
    for (init_i = 0; init_i < 8; init_i = init_i + 1) begin
        bias_reg[init_i] = 16'sd0; acc[init_i] = 32'sd0;
        prod_r[init_i] = 32'sd0;
        k_buf[init_i] = 16'sd0; v_buf[init_i] = 16'sd0;
        q_buf[init_i] = 16'sd0; km_acc[init_i] = 32'sd0;
    end
    for (init_i = 0; init_i < 64; init_i = init_i + 1)
        kv_acc[init_i] = 48'sd0;
    for (init_i = 0; init_i < KV_ELEMS; init_i = init_i + 1)
        kv_buf[init_i] = 16'sd0;
    for (init_i = 0; init_i < 32; init_i = init_i + 1)
        km_buf[init_i] = 16'sd0;
end
`endif

// =========================================================================
// Wires
// =========================================================================

// WS address decode
wire [2:0] wl_lane = wl_cnt[7:5];
wire [4:0] wl_feat = wl_cnt[4:0];

wire wl_done  = (wl_cnt == WL_MAX[7:0]) && (wl_phase == 1'b1);
wire bl_done  = (bl_cnt == 3'd7) && (bl_phase == 1'b1);

wire [5:0] mac_limit = IN_DIM[5:0];   // 32
wire mul_en   = (state == S_MAC) && (mac_cnt >= 6'd1) && (mac_cnt <= mac_limit);
wire acc_en   = (state == S_MAC) && (mac_cnt >= 6'd2);
wire mac_done = (mac_cnt == mac_limit + 6'd1);

wire sat_done   = (sat_lane == 4'd9);
wire tok_last   = (tok_cnt == N_TOKENS[8:0] - 9'd1);
wire group_last = (ws_phase == 1'b0) ?
                  (group_cnt == QKV_GROUPS[3:0] - 4'd1) :
                  (group_cnt == PROJ_GROUPS[3:0] - 4'd1);

wire is_split = (ws_phase == 1'b0) && (group_cnt < 4'd8);

// MAC input mux
wire signed [15:0] mac_x = (ws_phase == 1'b0) ? norm_x : $signed(sram_ao_q_i);

// SAT pipeline combinational
wire signed [31:0] sat_shr8  = acc_pick >>> 8;
wire signed [31:0] sat_add_b = sat_shr8 + {{16{bias_pick[15]}}, bias_pick};

wire signed [15:0] sat_pre =
    (sat_mid_r > 32'sd32767)  ? 16'sh7FFF :
    (sat_mid_r < -32'sd32768) ? 16'sh8000 : sat_mid_r[15:0];

// Fused SPLIT: relu6(sat16(rnd_shr8(sat_pre * S_Q88)))
wire signed [31:0] sp_prod = $signed(sat_pre) * $signed(S_Q88[15:0]);
wire signed [31:0] sp_rnd  = sp_prod + 32'sd128;
wire signed [31:0] sp_shr  = sp_rnd >>> 8;
wire signed [15:0] sp_sat  =
    (sp_shr > 32'sd32767)  ? 16'sh7FFF :
    (sp_shr < -32'sd32768) ? 16'sh8000 : sp_shr[15:0];
wire signed [15:0] sp_relu6 =
    (sp_sat[15])                               ? 16'sd0 :
    (sp_sat > $signed(RELU6_MAX[15:0]))        ? RELU6_MAX[15:0] : sp_sat;

wire signed [15:0] sat_final = is_split ? sp_relu6 : sat_pre;

// QKV / PROJ SRAM write flat addresses
wire [13:0] qkv_wr_flat =
    {12'd0, group_cnt[1:0]} * NTxHD[13:0] +
    {5'd0, tok_cnt} * HEAD_DIM[13:0] +
    {11'd0, sat_wr_lane};
wire [13:0] proj_wr_flat = {tok_cnt, 5'b0} + {8'd0, group_cnt[1:0], sat_wr_lane};

// S_KV SRAM read flat
wire [13:0] kv_sram_rd_flat =
    {12'd0, kv_head} * NTxHD[13:0] +
    {5'd0, kv_tok} * HEAD_DIM[13:0] +
    {11'd0, kv_rd_cnt[2:0] - 3'd1};

// KV outer-product wires
wire signed [31:0] kv_p0 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[0]);
wire signed [31:0] kv_p1 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[1]);
wire signed [31:0] kv_p2 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[2]);
wire signed [31:0] kv_p3 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[3]);
wire signed [31:0] kv_p4 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[4]);
wire signed [31:0] kv_p5 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[5]);
wire signed [31:0] kv_p6 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[6]);
wire signed [31:0] kv_p7 = $signed(k_buf[kv_mac_dk]) * $signed(v_buf[7]);

// KV_SCALE combinational
wire [5:0]  kvs_kv_idx = kvs_idx[5:0];
wire [2:0]  kvs_km_idx = kvs_idx[2:0];

wire signed [63:0] kvs_prod = $signed(kv_acc[kvs_kv_idx]) * $signed(16'sd205);
wire signed [63:0] kvs_rnd  = kvs_prod + 64'sd8388608;
wire signed [63:0] kvs_shr  = kvs_rnd >>> 24;
wire signed [15:0] kvs_val  =
    (kvs_shr > 64'sd32767)  ? 16'sh7FFF :
    (kvs_shr < -64'sd32768) ? 16'sh8000 : kvs_shr[15:0];

wire signed [47:0] kms_prod = $signed(km_acc[kvs_km_idx]) * $signed(16'sd205);
wire signed [47:0] kms_rnd  = kms_prod + 48'sd32768;
wire signed [47:0] kms_shr  = kms_rnd >>> 16;
wire signed [15:0] kms_val  =
    (kms_shr > 48'sd32767)  ? 16'sh7FFF :
    (kms_shr < -48'sd32768) ? 16'sh8000 : kms_shr[15:0];

// S_QKM combinational (qkm_d_idx valid on even READ and following odd MAC)
wire [2:0]  qkm_d_idx = qkm_cnt[3:1];
wire [13:0] qkm_q_flat =
    {12'd0, qkm_h} * NTxHD[13:0] +
    {5'd0, qkm_n} * HEAD_DIM[13:0] +
    {11'd0, qkm_d_idx};
wire signed [15:0] qkm_km_val = km_buf[{qkm_h, qkm_d_idx}];
// CLK(clk): even cnt issues sram_q read (addr@cnt -> Q@cnt+1); odd cnt consumes the
// registered macro output sram_q_q_i directly (no same-cycle qkm_q_r latch needed).
wire signed [31:0] qkm_term   = $signed(sram_q_q_i) * qkm_km_val;
wire signed [31:0] qkm_rnd    = qkm_acc + 32'sd128;
wire signed [31:0] qkm_shr    = qkm_rnd >>> 8;
wire signed [15:0] qkm_sat    =
    (qkm_shr > 32'sd32767)  ? 16'sh7FFF :
    (qkm_shr < -32'sd32768) ? 16'sh8000 : qkm_shr[15:0];
wire signed [15:0] qkm_clamp  = (qkm_sat < 16'sd1) ? 16'sd1 : qkm_sat;

wire [10:0] qkm_wr_addr = {2'd0, qkm_h} * N_TOKENS[10:0] + {2'd0, qkm_n};

// S_ZR wires
wire [10:0] zr_flat = {2'd0, zr_h} * N_TOKENS[10:0] + {2'd0, zr_n};

wire        recip_busy;
wire        recip_done;
wire signed [15:0] recip_y_o;

wire zr_last = (zr_h == NUM_HEADS[1:0] - 2'd1) &&
               (zr_n == N_TOKENS[8:0]  - 9'd1);

// S_CORE wires
wire [13:0] at_q_flat =
    {12'd0, at_h} * NTxHD[13:0] +
    {5'd0, at_n} * HEAD_DIM[13:0] +
    {11'd0, q_load_dk};
wire [10:0] at_zr_flat = {2'd0, at_h} * N_TOKENS[10:0] + {2'd0, at_n};
wire [13:0] at_ao_flat =
    {5'd0, at_n} * EMBED_DIM[13:0] +
    {12'd0, at_h} * HEAD_DIM[13:0] +
    {11'd0, at_dout};

wire [7:0] at_kv0 = {at_h, 3'd0, at_dout};
wire [7:0] at_kv1 = {at_h, 3'd1, at_dout};
wire [7:0] at_kv2 = {at_h, 3'd2, at_dout};
wire [7:0] at_kv3 = {at_h, 3'd3, at_dout};
wire [7:0] at_kv4 = {at_h, 3'd4, at_dout};
wire [7:0] at_kv5 = {at_h, 3'd5, at_dout};
wire [7:0] at_kv6 = {at_h, 3'd6, at_dout};
wire [7:0] at_kv7 = {at_h, 3'd7, at_dout};

wire signed [31:0] at_t0 = $signed(q_buf[0]) * $signed(kv_buf[at_kv0]);
wire signed [31:0] at_t1 = $signed(q_buf[1]) * $signed(kv_buf[at_kv1]);
wire signed [31:0] at_t2 = $signed(q_buf[2]) * $signed(kv_buf[at_kv2]);
wire signed [31:0] at_t3 = $signed(q_buf[3]) * $signed(kv_buf[at_kv3]);
wire signed [31:0] at_t4 = $signed(q_buf[4]) * $signed(kv_buf[at_kv4]);
wire signed [31:0] at_t5 = $signed(q_buf[5]) * $signed(kv_buf[at_kv5]);
wire signed [31:0] at_t6 = $signed(q_buf[6]) * $signed(kv_buf[at_kv6]);
wire signed [31:0] at_t7 = $signed(q_buf[7]) * $signed(kv_buf[at_kv7]);

wire signed [48:0] at_acc_par =
    {{17{at_t0[31]}}, at_t0} + {{17{at_t1[31]}}, at_t1} +
    {{17{at_t2[31]}}, at_t2} + {{17{at_t3[31]}}, at_t3} +
    {{17{at_t4[31]}}, at_t4} + {{17{at_t5[31]}}, at_t5} +
    {{17{at_t6[31]}}, at_t6} + {{17{at_t7[31]}}, at_t7};

wire signed [48:0] at_dot_shr = (at_acc + 49'sd128) >>> 8;
wire signed [15:0] at_dot_sat =
    (at_dot_shr > 49'sd32767)  ? 16'sh7FFF :
    (at_dot_shr < -49'sd32768) ? 16'sh8000 : at_dot_shr[15:0];

wire signed [31:0] at_zp_raw = $signed(at_dot_sat_r) * $signed(core_zr_r);
wire signed [31:0] at_zp_rnd = at_zp_raw + 32'sd128;
wire signed [31:0] at_zp_shr = at_zp_rnd >>> 8;
wire signed [15:0] at_ao_sat =
    (at_zp_shr > 32'sd32767)  ? 16'sh7FFF :
    (at_zp_shr < -32'sd32768) ? 16'sh8000 : at_zp_shr[15:0];

wire at_dout_last = (at_dout == HEAD_DIM[2:0] - 3'd1);
wire at_hn_last   = (at_h == NUM_HEADS[1:0] - 2'd1) &&
                    (at_n == N_TOKENS[8:0]  - 9'd1);
wire at_done_now  = at_hn_last && at_dout_last && (at_phase == AT_AO);

// S_OUT
wire out_done = (out_cnt == OUT_TOTAL[13:0] - 14'd1) && (out_phase == 1'b1);

assign busy = (state != S_IDLE);

// =========================================================================
// recip_nr instantiation (for S_ZR)
// =========================================================================
recip_nr u_recip (
    .clk   (clk),
    .reset (reset),
    .start (zr_start_r),
    .x_i   (zr_x_r),
    .busy  (recip_busy),
    .done  (recip_done),
    .y_o   (recip_y_o)
);

// =========================================================================
// ROM address generation
// =========================================================================
wire [6:0] wl_neu_qkv  = {group_cnt[3:0], wl_lane};
wire [4:0] wl_neu_proj = {group_cnt[1:0], wl_lane};

assign wgt_addr_o =
    (state == S_WLOAD) ?
        ((ws_phase == 1'b0) ?
            {1'b0, wl_neu_qkv, wl_feat} :
            (13'd3072 + {3'd0, wl_neu_proj, wl_feat})) :
    (state == S_BLOAD) ?
        ((ws_phase == 1'b0) ? 13'd0 : 13'd3072) :
    13'd0;

assign bias_addr_o =
    (state == S_BLOAD) ?
        {1'b0, group_cnt[3:0], bl_cnt} : 8'd0;

// =========================================================================
// FSM state register
// =========================================================================
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM next-state logic
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:     if (start) next_state = S_WLOAD;
        S_WLOAD:    if (wl_done) next_state = S_BLOAD;
        S_BLOAD:    if (bl_done) next_state = S_MAC;
        S_MAC:      if (mac_done) next_state = S_SAT;
        S_SAT: begin
            if (sat_done && !tok_last)
                next_state = S_MAC;
            else if (sat_done && tok_last && !group_last)
                next_state = S_WLOAD;
            else if (sat_done && tok_last && group_last && ws_phase == 1'b0)
                next_state = S_KV;
            else if (sat_done && tok_last && group_last)
                next_state = S_OUT;
        end
        S_KV: begin
            if (kv_sub == KV_MAC && kv_mac_dk == 3'd7 &&
                kv_tok == N_TOKENS[8:0] - 9'd1)
                next_state = S_KV_SCALE;
        end
        S_KV_SCALE: begin
            if (kvs_idx == 7'd71 && kv_head == NUM_HEADS[1:0] - 2'd1)
                next_state = S_QKM;
            else if (kvs_idx == 7'd71)
                next_state = S_KV;
        end
        S_QKM: begin
            if (qkm_cnt == 5'd16 &&
                qkm_h == NUM_HEADS[1:0] - 2'd1 &&
                qkm_n == N_TOKENS[8:0] - 9'd1)
                next_state = S_ZR;
        end
        S_ZR: begin
            if (zr_sub == 2'd3 && recip_done && zr_last)
                next_state = S_CORE;
        end
        S_CORE:     if (at_done_now) next_state = S_WLOAD;
        S_OUT:      if (out_done) next_state = S_DONE_ST;
        S_DONE_ST:  next_state = S_IDLE;
        default:    next_state = S_IDLE;
    endcase
end

// =========================================================================
// ws_phase (0=QKV, 1=PROJ)
// =========================================================================
always @(posedge clk) begin
    if (reset)
        ws_phase <= 1'b0;
    else if (state == S_IDLE && start)
        ws_phase <= 1'b0;
    else if (state == S_CORE && at_done_now)
        ws_phase <= 1'b1;
end

// group_cnt
always @(posedge clk) begin
    if (reset)
        group_cnt <= 4'd0;
    else if (state == S_IDLE)
        group_cnt <= 4'd0;
    else if (state == S_CORE && at_done_now)
        group_cnt <= 4'd0;
    else if (state == S_SAT && sat_done && tok_last && !group_last)
        group_cnt <= group_cnt + 4'd1;
    else if (state == S_SAT && sat_done && tok_last && group_last)
        group_cnt <= 4'd0;
end

// tok_cnt
always @(posedge clk) begin
    if (reset)
        tok_cnt <= 9'd0;
    else if (state == S_IDLE)
        tok_cnt <= 9'd0;
    else if (state == S_SAT && sat_done && !tok_last)
        tok_cnt <= tok_cnt + 9'd1;
    else if (state == S_SAT && sat_done && tok_last)
        tok_cnt <= 9'd0;
end

// =========================================================================
// S_WLOAD: weight preload (2-phase ROM read -> w_lane registers)
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        wl_cnt   <= 8'd0;
        wl_phase <= 1'b0;
    end else if (state != S_WLOAD) begin
        wl_cnt   <= 8'd0;
        wl_phase <= 1'b0;
    end else if (wl_phase == 1'b0)
        wl_phase <= 1'b1;
    else if (wl_cnt != WL_MAX[7:0]) begin
        wl_phase <= 1'b0;
        wl_cnt   <= wl_cnt + 8'd1;
    end else
        wl_phase <= 1'b0;
end

// Weight capture
always @(posedge clk) begin
    if (state == S_WLOAD && wl_phase == 1'b1) begin
        case (wl_lane)
            3'd0: w_lane0[wl_feat] <= wgt_i;
            3'd1: w_lane1[wl_feat] <= wgt_i;
            3'd2: w_lane2[wl_feat] <= wgt_i;
            3'd3: w_lane3[wl_feat] <= wgt_i;
            3'd4: w_lane4[wl_feat] <= wgt_i;
            3'd5: w_lane5[wl_feat] <= wgt_i;
            3'd6: w_lane6[wl_feat] <= wgt_i;
            3'd7: w_lane7[wl_feat] <= wgt_i;
            default: ;
        endcase
    end
end

// =========================================================================
// S_BLOAD: bias preload (2-phase ROM read -> bias_reg)
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        bl_cnt   <= 3'd0;
        bl_phase <= 1'b0;
    end else if (state != S_BLOAD) begin
        bl_cnt   <= 3'd0;
        bl_phase <= 1'b0;
    end else if (bl_phase == 1'b0)
        bl_phase <= 1'b1;
    else if (bl_cnt != 3'd7) begin
        bl_phase <= 1'b0;
        bl_cnt   <= bl_cnt + 3'd1;
    end else
        bl_phase <= 1'b0;
end

// Bias capture
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            bias_reg[i_lane] <= 16'sd0;
    end else if (state == S_BLOAD && bl_phase == 1'b1)
        bias_reg[bl_cnt] <= bias_i;
end

// =========================================================================
// S_MAC: 3-stage pipelined MAC (w_rd -> prod -> acc)
// =========================================================================
// MAC counter
always @(posedge clk) begin
    if (reset)        mac_cnt <= 6'd0;
    else if (state != S_MAC) mac_cnt <= 6'd0;
    else if (!mac_done) mac_cnt <= mac_cnt + 6'd1;
end

// Weight read pipeline register
always @(posedge clk) begin
    if (state == S_MAC && mac_cnt < mac_limit) begin
        w_rd_r0 <= w_lane0[mac_cnt[4:0]];
        w_rd_r1 <= w_lane1[mac_cnt[4:0]];
        w_rd_r2 <= w_lane2[mac_cnt[4:0]];
        w_rd_r3 <= w_lane3[mac_cnt[4:0]];
        w_rd_r4 <= w_lane4[mac_cnt[4:0]];
        w_rd_r5 <= w_lane5[mac_cnt[4:0]];
        w_rd_r6 <= w_lane6[mac_cnt[4:0]];
        w_rd_r7 <= w_lane7[mac_cnt[4:0]];
    end
end

// Multiply stage
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            prod_r[i_lane] <= 32'sd0;
    end else if (mul_en) begin
        prod_r[0] <= w_rd_r0 * mac_x;
        prod_r[1] <= w_rd_r1 * mac_x;
        prod_r[2] <= w_rd_r2 * mac_x;
        prod_r[3] <= w_rd_r3 * mac_x;
        prod_r[4] <= w_rd_r4 * mac_x;
        prod_r[5] <= w_rd_r5 * mac_x;
        prod_r[6] <= w_rd_r6 * mac_x;
        prod_r[7] <= w_rd_r7 * mac_x;
    end
end

// Accumulator
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc[i_lane] <= 32'sd0;
    end else if (state == S_MAC && mac_cnt == 6'd0) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc[i_lane] <= 32'sd0;
    end else if (acc_en) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc[i_lane] <= acc[i_lane] + prod_r[i_lane];
    end
end

// =========================================================================
// Norm read (QKV MAC input from parent sram via norm_rd)
// =========================================================================
// CLK(clk): combinational read address. Present feat=mac_cnt during cycle T so the
// posedge macro returns norm_x in T+1, matching the MAC mul at mac_cnt (w_rd_r[k-1] *
// x[k-1]). A registered norm_rd would slip x one cycle late under CLK(clk).
always @(*) begin
    if (state == S_MAC && ws_phase == 1'b0 && mac_cnt < IN_DIM[5:0]) begin
        norm_rd_en   = 1'b1;
        norm_rd_flat = {tok_cnt, 5'b0} + {8'd0, mac_cnt[5:0]};
    end else begin
        norm_rd_en   = 1'b0;
        norm_rd_flat = 14'd0;
    end
end

// PROJ sram_ao read: combinational address driven directly in the sram_ao mux below.

// =========================================================================
// S_SAT: 3-stage pipeline (shr8+bias -> sat/split -> SRAM write)
// =========================================================================
// SAT counter
always @(posedge clk) begin
    if (reset)        sat_lane <= 4'd0;
    else if (state != S_SAT) sat_lane <= 4'd0;
    else              sat_lane <= sat_lane + 4'd1;
end

// SAT acc/bias mux
always @(*) begin
    acc_pick  = 32'sd0;
    bias_pick = 16'sd0;
    case (sat_lane[2:0])
        3'd0: begin acc_pick = acc[0]; bias_pick = bias_reg[0]; end
        3'd1: begin acc_pick = acc[1]; bias_pick = bias_reg[1]; end
        3'd2: begin acc_pick = acc[2]; bias_pick = bias_reg[2]; end
        3'd3: begin acc_pick = acc[3]; bias_pick = bias_reg[3]; end
        3'd4: begin acc_pick = acc[4]; bias_pick = bias_reg[4]; end
        3'd5: begin acc_pick = acc[5]; bias_pick = bias_reg[5]; end
        3'd6: begin acc_pick = acc[6]; bias_pick = bias_reg[6]; end
        3'd7: begin acc_pick = acc[7]; bias_pick = bias_reg[7]; end
        default: ;
    endcase
end

// SAT stage 1: >>>8 + bias -> sat_mid_r
always @(posedge clk) begin
    if (reset) begin
        sat_mid_r    <= 32'sd0;
        sat_s1_valid <= 1'b0;
        sat_s1_lane  <= 3'd0;
    end else if (state == S_SAT && sat_lane <= 4'd7) begin
        sat_mid_r    <= sat_add_b;
        sat_s1_valid <= 1'b1;
        sat_s1_lane  <= sat_lane[2:0];
    end else begin
        sat_s1_valid <= 1'b0;
    end
end

// SAT stage 2: clamp + fused SPLIT -> sat_val_r
always @(posedge clk) begin
    if (reset) begin
        sat_val_r      <= 16'sd0;
        sat_wr_pending <= 1'b0;
        sat_wr_lane    <= 3'd0;
    end else if (sat_s1_valid) begin
        sat_val_r      <= sat_final;
        sat_wr_pending <= 1'b1;
        sat_wr_lane    <= sat_s1_lane;
    end else begin
        sat_wr_pending <= 1'b0;
    end
end

// =========================================================================
// S_KV: Token-Stationary outer product + fused K_MEAN
// =========================================================================
// KV counters
always @(posedge clk) begin
    if (reset) begin
        kv_head    <= 2'd0;
        kv_tok     <= 9'd0;
        kv_sub     <= KV_CLEAR;
        kv_rd_cnt  <= 4'd0;
        kv_mac_dk  <= 3'd0;
    end else if (state == S_IDLE) begin
        kv_head <= 2'd0;
    end else if (state == S_SAT && sat_done && tok_last && group_last &&
                 ws_phase == 1'b0) begin
        kv_head <= 2'd0;
        kv_tok  <= 9'd0;
        kv_sub  <= KV_CLEAR;
    end else if (state == S_KV_SCALE && kvs_idx == 7'd71 &&
                 kv_head != NUM_HEADS[1:0] - 2'd1) begin
        kv_head <= kv_head + 2'd1;
        kv_tok  <= 9'd0;
        kv_sub  <= KV_CLEAR;
    end else if (state == S_KV) begin
        case (kv_sub)
            KV_CLEAR: begin
                kv_sub    <= KV_RD;
                kv_rd_cnt <= 4'd0;
            end
            KV_RD: begin
                // CLK(clk): addr@cnt -> macro Q@cnt+1, so the last element (addr@cnt8)
                // is captured at cnt9. Extend KV_RD by one beat (8 -> 9) before KV_MAC.
                if (kv_rd_cnt == 4'd9) begin
                    kv_sub    <= KV_MAC;
                    kv_mac_dk <= 3'd0;
                end else
                    kv_rd_cnt <= kv_rd_cnt + 4'd1;
            end
            KV_MAC: begin
                if (kv_mac_dk == 3'd7 &&
                    kv_tok != N_TOKENS[8:0] - 9'd1) begin
                    kv_tok    <= kv_tok + 9'd1;
                    kv_sub    <= KV_RD;
                    kv_rd_cnt <= 4'd0;
                end else if (kv_mac_dk != 3'd7)
                    kv_mac_dk <= kv_mac_dk + 3'd1;
            end
            default: kv_sub <= KV_CLEAR;
        endcase
    end
end

// KV k_buf / v_buf capture (CLK(clk): macro Q valid 1 cycle after addr).
// addr for idx (kv_rd_cnt-1) is driven at kv_rd_cnt 1..8; capture idx (kv_rd_cnt-2)
// one cycle later at kv_rd_cnt 2..9.
always @(posedge clk) begin
    if (state == S_KV && kv_sub == KV_RD && kv_rd_cnt >= 4'd2 && kv_rd_cnt <= 4'd9)
    begin
        k_buf[kv_rd_cnt[3:0] - 4'd2] <= $signed(sram_k_q_i);
        v_buf[kv_rd_cnt[3:0] - 4'd2] <= $signed(sram_v_q_i);
    end
end

// KV accumulator clear and outer product
always @(posedge clk) begin
    if (state == S_KV && kv_sub == KV_CLEAR) begin
        for (i_lane = 0; i_lane < 64; i_lane = i_lane + 1)
            kv_acc[i_lane] <= 48'sd0;
        for (i_lane = 0; i_lane < 8; i_lane = i_lane + 1)
            km_acc[i_lane] <= 32'sd0;
    end else if (state == S_KV && kv_sub == KV_MAC) begin
        kv_acc[{kv_mac_dk, 3'd0}] <= kv_acc[{kv_mac_dk, 3'd0}] + {{16{kv_p0[31]}}, kv_p0};
        kv_acc[{kv_mac_dk, 3'd1}] <= kv_acc[{kv_mac_dk, 3'd1}] + {{16{kv_p1[31]}}, kv_p1};
        kv_acc[{kv_mac_dk, 3'd2}] <= kv_acc[{kv_mac_dk, 3'd2}] + {{16{kv_p2[31]}}, kv_p2};
        kv_acc[{kv_mac_dk, 3'd3}] <= kv_acc[{kv_mac_dk, 3'd3}] + {{16{kv_p3[31]}}, kv_p3};
        kv_acc[{kv_mac_dk, 3'd4}] <= kv_acc[{kv_mac_dk, 3'd4}] + {{16{kv_p4[31]}}, kv_p4};
        kv_acc[{kv_mac_dk, 3'd5}] <= kv_acc[{kv_mac_dk, 3'd5}] + {{16{kv_p5[31]}}, kv_p5};
        kv_acc[{kv_mac_dk, 3'd6}] <= kv_acc[{kv_mac_dk, 3'd6}] + {{16{kv_p6[31]}}, kv_p6};
        kv_acc[{kv_mac_dk, 3'd7}] <= kv_acc[{kv_mac_dk, 3'd7}] + {{16{kv_p7[31]}}, kv_p7};
        km_acc[kv_mac_dk] <= km_acc[kv_mac_dk] + {{16{k_buf[kv_mac_dk][15]}}, k_buf[kv_mac_dk]};
    end
end

// =========================================================================
// S_KV_SCALE: scale kv_acc -> kv_buf, km_acc -> km_buf
// =========================================================================
always @(posedge clk) begin
    if (reset)
        kvs_idx <= 7'd0;
    else if (state != S_KV_SCALE)
        kvs_idx <= 7'd0;
    else if (kvs_idx != 7'd71)
        kvs_idx <= kvs_idx + 7'd1;
end

// kv_buf write
always @(posedge clk) begin
    if (state == S_KV_SCALE && kvs_idx < 7'd64)
        kv_buf[{kv_head, kvs_kv_idx}] <= kvs_val;
end

// km_buf write
always @(posedge clk) begin
    if (state == S_KV_SCALE && kvs_idx >= 7'd64 && kvs_idx < 7'd72)
        km_buf[{kv_head, kvs_km_idx}] <= kms_val;
end

// =========================================================================
// S_QKM: dot product q * km -> sram_qkm
// =========================================================================
// QKM counters (0..16 per h,n: even READ, odd MAC, 16=write latch)
always @(posedge clk) begin
    if (reset) begin
        qkm_h   <= 2'd0;
        qkm_n   <= 9'd0;
        qkm_cnt <= 5'd0;
    end else if (state != S_QKM) begin
        qkm_h   <= 2'd0;
        qkm_n   <= 9'd0;
        qkm_cnt <= 5'd0;
    end else if (qkm_cnt == 5'd16 &&
               qkm_n != N_TOKENS[8:0] - 9'd1) begin
        qkm_cnt <= 5'd0;
        qkm_n   <= qkm_n + 9'd1;
    end else if (qkm_cnt == 5'd16 &&
               qkm_n == N_TOKENS[8:0] - 9'd1 &&
               qkm_h != NUM_HEADS[1:0] - 2'd1) begin
        qkm_cnt <= 5'd0;
        qkm_n   <= 9'd0;
        qkm_h   <= qkm_h + 2'd1;
    end else if (qkm_cnt == 5'd16) begin
        qkm_cnt <= 5'd0;
        qkm_n   <= 9'd0;
    end else
        qkm_cnt <= qkm_cnt + 5'd1;
end

// QKM accumulator (odd cnt only; consumes macro Q sram_q_q_i issued on prior even cnt)
always @(posedge clk) begin
    if (state == S_QKM && qkm_cnt == 5'd1)
        qkm_acc <= qkm_term;
    else if (state == S_QKM && qkm_cnt[0] == 1'b1 &&
             qkm_cnt >= 5'd3 && qkm_cnt <= 5'd15)
        qkm_acc <= qkm_acc + qkm_term;
end

// QKM write latch: hold 1 cycle past S_QKM so last beat (QKM->ZR) still writes
always @(posedge clk) begin
    if (reset)
        qkm_wr_pending <= 1'b0;
    else if (state == S_QKM && qkm_cnt == 5'd16) begin
        qkm_wr_pending <= 1'b1;
        qkm_wr_addr_r  <= qkm_wr_addr;
        qkm_wr_data_r  <= qkm_clamp;
    end else if (qkm_wr_pending)
        qkm_wr_pending <= 1'b0;
end

// =========================================================================
// S_ZR: reciprocal via recip_nr
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        zr_sub     <= 2'd0;
        zr_h       <= 2'd0;
        zr_n       <= 9'd0;
        zr_start_r <= 1'b0;
        zr_x_r     <= 16'sd0;
    end else if (state != S_ZR) begin
        zr_sub     <= 2'd0;
        zr_h       <= 2'd0;
        zr_n       <= 9'd0;
        zr_start_r <= 1'b0;
    end else if (zr_sub == 2'd0)
        zr_sub <= 2'd1;
    else if (zr_sub == 2'd1)
        // CLK(clk): sram_qkm read addr issued this cycle (mux @ zr_sub==1);
        // macro Q valid next cycle -> capture at zr_sub==2.
        zr_sub <= 2'd2;
    else if (zr_sub == 2'd2) begin
        zr_x_r     <= $signed(sram_qkm_q_i);
        zr_start_r <= 1'b1;
        zr_sub     <= 2'd3;
    end else if (zr_sub == 2'd3 && recip_done &&
               zr_n != N_TOKENS[8:0] - 9'd1) begin
        zr_sub     <= 2'd0;
        zr_n       <= zr_n + 9'd1;
        zr_start_r <= 1'b0;
    end else if (zr_sub == 2'd3 && recip_done &&
               zr_n == N_TOKENS[8:0] - 9'd1 &&
               zr_h != NUM_HEADS[1:0] - 2'd1) begin
        zr_sub     <= 2'd0;
        zr_n       <= 9'd0;
        zr_h       <= zr_h + 2'd1;
        zr_start_r <= 1'b0;
    end else if (zr_sub == 2'd3 && recip_done) begin
        zr_sub     <= 2'd0;
        zr_start_r <= 1'b0;
    end else
        zr_start_r <= 1'b0;
end

// ZR write latch: hold 1 cycle past S_ZR so last beat (ZR->CORE) still writes
always @(posedge clk) begin
    if (reset)
        zr_wr_pending <= 1'b0;
    else if (state == S_ZR && zr_sub == 2'd3 && recip_done) begin
        zr_wr_pending <= 1'b1;
        zr_wr_addr_r  <= zr_flat;
        zr_wr_data_r  <= recip_y_o;
    end else if (zr_wr_pending)
        zr_wr_pending <= 1'b0;
end

// =========================================================================
// S_CORE: attention computation (q * kv * zr -> sram_ao)
// =========================================================================
// CORE datapath
always @(posedge clk) begin
    if (reset) begin
        at_phase     <= AT_Q_ADDR;
        q_load_dk    <= 3'd0;
        at_h         <= 2'd0;
        at_n         <= 9'd0;
        at_dout      <= 3'd0;
        at_acc       <= 49'sd0;
        core_zr_r    <= 16'sd0;
        at_dot_sat_r <= 16'sd0;
    end else if (state != S_CORE) begin
        at_phase  <= AT_Q_ADDR;
        q_load_dk <= 3'd0;
        at_h      <= 2'd0;
        at_n      <= 9'd0;
        at_dout   <= 3'd0;
    end else begin
        case (at_phase)
            AT_Q_ADDR: at_phase <= AT_Q_CAP;
            AT_Q_CAP: begin
                q_buf[q_load_dk] <= $signed(sram_q_q_i);
                if (q_load_dk == HEAD_DIM[2:0] - 3'd1) begin
                    q_load_dk <= 3'd0;
                    at_phase  <= AT_ZR_ADDR;
                end else begin
                    q_load_dk <= q_load_dk + 3'd1;
                    at_phase  <= AT_Q_ADDR;
                end
            end
            AT_ZR_ADDR: at_phase <= AT_ZR_CAP;
            AT_ZR_CAP: begin
                core_zr_r <= $signed(sram_qkm_q_i);
                at_dout   <= 3'd0;
                at_phase  <= AT_MAC8;
            end
            AT_MAC8: begin
                at_acc   <= at_acc_par;
                at_phase <= AT_DOT;
            end
            AT_DOT: begin
                at_dot_sat_r <= at_dot_sat;
                at_phase     <= AT_AO;
            end
            AT_AO: begin
                if (!at_dout_last) begin
                    at_dout  <= at_dout + 3'd1;
                    at_phase <= AT_MAC8;
                end else if (at_n != N_TOKENS[8:0] - 9'd1) begin
                    at_n      <= at_n + 9'd1;
                    at_dout   <= 3'd0;
                    q_load_dk <= 3'd0;
                    at_phase  <= AT_Q_ADDR;
                end else if (at_h != NUM_HEADS[1:0] - 2'd1) begin
                    at_h      <= at_h + 2'd1;
                    at_n      <= 9'd0;
                    at_dout   <= 3'd0;
                    q_load_dk <= 3'd0;
                    at_phase  <= AT_Q_ADDR;
                end
            end
            default: at_phase <= AT_Q_ADDR;
        endcase
    end
end

// =========================================================================
// S_OUT: stream from sram_q (2-phase read)
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        out_cnt   <= 14'd0;
        out_phase <= 1'b0;
    end else if (state != S_OUT) begin
        out_cnt   <= 14'd0;
        out_phase <= 1'b0;
    end else if (out_phase == 1'b0)
        out_phase <= 1'b1;
    else if (out_cnt < OUT_TOTAL[13:0] - 14'd1) begin
        out_phase <= 1'b0;
        out_cnt   <= out_cnt + 14'd1;
    end else
        out_phase <= 1'b0;
end

// =========================================================================
// y_o / y_valid / done
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        y_o      <= 16'sd0;
        y_valid  <= 1'b0;
        y_neu_o  <= 7'd0;
        px_tok_o <= 9'd0;
    end else if (state == S_OUT && out_phase == 1'b1) begin
        y_o      <= $signed(sram_q_q_i);
        y_valid  <= 1'b1;
        px_tok_o <= out_cnt[13:5];
        y_neu_o  <= {2'b0, out_cnt[4:0]};
    end else
        y_valid <= 1'b0;
end

// Done pulse
always @(posedge clk) begin
    if (reset)     done <= 1'b0;
    else if (state == S_DONE_ST) done <= 1'b1;
    else           done <= 1'b0;
end

// =========================================================================
// SRAM mux (combinational; one if/else if chain per macro)
// =========================================================================
// sram_ao: CORE write or PROJ MAC read
always @(*) begin
    sram_ao_ceb_o  = 1'b1;
    sram_ao_web_o  = 1'b1;
    sram_ao_addr_o = 14'd0;
    sram_ao_din_o  = 16'd0;
    if (state == S_CORE && at_phase == AT_AO) begin
        sram_ao_ceb_o  = 1'b0;
        sram_ao_web_o  = 1'b0;
        sram_ao_addr_o = at_ao_flat;
        sram_ao_din_o  = at_ao_sat;
    end else if (state == S_MAC && ws_phase == 1'b1 &&
               mac_cnt < IN_DIM[5:0]) begin
        // CLK(clk) combinational read: addr@T (feat=mac_cnt) -> macro Q@T+1, consumed
        // by the PROJ MAC mul at mac_cnt+1 (pairs w_rd_r[k] with x[k]).
        sram_ao_ceb_o  = 1'b0;
        sram_ao_web_o  = 1'b1;
        sram_ao_addr_o = {tok_cnt, 5'b0} + {9'd0, mac_cnt[4:0]};
    end
end

// sram_qkm: deferred QKM/ZR write or stage read
always @(*) begin
    sram_qkm_ceb_o  = 1'b1;
    sram_qkm_web_o  = 1'b1;
    sram_qkm_addr_o = 14'd0;
    sram_qkm_din_o  = 16'd0;
    if (qkm_wr_pending) begin
        sram_qkm_ceb_o  = 1'b0;
        sram_qkm_web_o  = 1'b0;
        sram_qkm_addr_o = {3'd0, qkm_wr_addr_r};
        sram_qkm_din_o  = qkm_wr_data_r;
    end else if (zr_wr_pending) begin
        sram_qkm_ceb_o  = 1'b0;
        sram_qkm_web_o  = 1'b0;
        sram_qkm_addr_o = {3'd0, zr_wr_addr_r};
        sram_qkm_din_o  = zr_wr_data_r;
    end else if (state == S_ZR && zr_sub == 2'd1) begin
        sram_qkm_ceb_o  = 1'b0;
        sram_qkm_web_o  = 1'b1;
        sram_qkm_addr_o = {3'd0, zr_flat};
    end else if (state == S_CORE && at_phase == AT_ZR_ADDR) begin
        // CLK(clk): issue read in AT_ZR_ADDR so macro Q is valid in AT_ZR_CAP.
        // addr@T -> Q@T+1 (at_h/at_n stable across ADDR->CAP).
        sram_qkm_ceb_o  = 1'b0;
        sram_qkm_web_o  = 1'b1;
        sram_qkm_addr_o = {3'd0, at_zr_flat};
    end
end

// sram_q: QKV/PROJ SAT write or stage read
always @(*) begin
    sram_q_ceb_o  = 1'b1;
    sram_q_web_o  = 1'b1;
    sram_q_addr_o = 14'd0;
    sram_q_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && ws_phase == 1'b0 &&
        group_cnt[3:2] == 2'b00) begin
        sram_q_ceb_o  = 1'b0;
        sram_q_web_o  = 1'b0;
        sram_q_addr_o = qkv_wr_flat;
        sram_q_din_o  = sat_val_r;
    end else if (state == S_SAT && sat_wr_pending && ws_phase == 1'b1) begin
        sram_q_ceb_o  = 1'b0;
        sram_q_web_o  = 1'b0;
        sram_q_addr_o = proj_wr_flat;
        sram_q_din_o  = sat_val_r;
    end else if (state == S_QKM && qkm_cnt[0] == 1'b0 && qkm_cnt <= 5'd14) begin
        sram_q_ceb_o  = 1'b0;
        sram_q_web_o  = 1'b1;
        sram_q_addr_o = qkm_q_flat;
    end else if (state == S_CORE && at_phase == AT_Q_ADDR) begin
        // CLK(clk): issue read in AT_Q_ADDR so macro Q is valid in AT_Q_CAP (q_load_dk
        // is stable across ADDR->CAP for the same dim). addr@T -> Q@T+1.
        sram_q_ceb_o  = 1'b0;
        sram_q_web_o  = 1'b1;
        sram_q_addr_o = at_q_flat;
    end else if (state == S_OUT && out_phase == 1'b0) begin
        sram_q_ceb_o  = 1'b0;
        sram_q_web_o  = 1'b1;
        sram_q_addr_o = out_cnt;
    end
end

// sram_k: QKV SAT write or KV read
always @(*) begin
    sram_k_ceb_o  = 1'b1;
    sram_k_web_o  = 1'b1;
    sram_k_addr_o = 14'd0;
    sram_k_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && ws_phase == 1'b0 &&
        group_cnt[3:2] == 2'b01) begin
        sram_k_ceb_o  = 1'b0;
        sram_k_web_o  = 1'b0;
        sram_k_addr_o = qkv_wr_flat;
        sram_k_din_o  = sat_val_r;
    end else if (state == S_KV && kv_sub == KV_RD &&
               kv_rd_cnt >= 4'd1 && kv_rd_cnt <= 4'd8) begin
        sram_k_ceb_o  = 1'b0;
        sram_k_web_o  = 1'b1;
        sram_k_addr_o = kv_sram_rd_flat;
    end
end

// sram_v: QKV SAT write or KV read
always @(*) begin
    sram_v_ceb_o  = 1'b1;
    sram_v_web_o  = 1'b1;
    sram_v_addr_o = 14'd0;
    sram_v_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && ws_phase == 1'b0 &&
        group_cnt[3:2] == 2'b10) begin
        sram_v_ceb_o  = 1'b0;
        sram_v_web_o  = 1'b0;
        sram_v_addr_o = qkv_wr_flat;
        sram_v_din_o  = sat_val_r;
    end else if (state == S_KV && kv_sub == KV_RD &&
               kv_rd_cnt >= 4'd1 && kv_rd_cnt <= 4'd8) begin
        sram_v_ceb_o  = 1'b0;
        sram_v_web_o  = 1'b1;
        sram_v_addr_o = kv_sram_rd_flat;
    end
end

endmodule
