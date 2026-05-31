// =============================================================================
// care_attention.v  (SRAM macros in sglatrack_top; QKV reads parent norm1 via norm_rd_*)
//
// CARE multi-head attention (Softmax-free, O(N), Q8.8 fixed-point).
// Bit-accurate mirror of attention_forward() in
// python/tracking/run_backbone_numpy_shared_trunk.py (Q8.8 CARE path).
//
// Pipeline (per block):
//   parent norm1 on Sram_tok1 (token-major) -> 2-phase read -> [QKV linear] -> [SPLIT...]
//      -> [K_MEAN] -> [QK_MEAN] -> [Z_RECIP via NR]
//      -> [KV outer mean] -> [ATTN: q@kv*zr]
//      -> [PROJ linear] -> y_o stream
//
// ============================================================================
// SRAM activation buffers (macros in sglatrack_top; names match head2 sram_* style):
// ============================================================================
//   q_buf  -> sram_q_*   (Sram_q  12288 x 16)
//   k_buf  -> sram_k_*   (Sram_k  12288 x 16)
//   v_buf  -> sram_v_*   (Sram_v  12288 x 16)
//                     also hosts ao_buf in time-multiplex (deviation from
//                     python/md/SRAM_suggestion.md 5.3: ao_buf was planned in
//                     Sram_q, but moved to Sram_v to avoid same-cycle read q +
//                     write ao on a single-port macro during S_ATTN. v_buf is
//                     idle from S_ATTN onward, so v + ao share Sram_v with no
//                     temporal overlap.)
//   ao_buf -> Sram_v  (S_ATTN write, S_PROJ read; non-overlapping with v role)
//
//   norm1 x            -> parent Sram_tok1 (block staging; same flatten as norm2 tmp-on-q)
//   q layout           -> sram_q only (QKV capture, SPLIT, QK_MEAN, ATTN; no x)
//   qkm_buf  [ 1280]   -> Sram_qkm (S_QK_MEAN write, S_Z_RECIP read/write zr)
//   zr_buf   [ 1280]   -> same Sram_qkm (non-overlapping states vs qkm role)
//   km_buf   [   32]   (reg; S_K_MEAN / S_QK_MEAN)
//   kv_buf   [  256]   (reg; S_KV / S_ATTN)
//
// SRAM read contract (verilog_rule.mdc 7.7, also matches existing ROM convention):
//   posedge T  : drive A, D, WEB, CEB=0 to macro
//   posedge T+1: Q valid (macro CLK = ~clk, samples at internal negedge T)
//   single-port: cannot read AND write the same posedge on the same macro
//
// Conflict-handling FSM additions (all scalar regs, NO new 2D reg per rule 5.1):
//   S_SPLIT          : 2-phase (sp_phase). Phase 0 drives Sram_q+Sram_k read at
//                       sp_ptr; sp_q_r/sp_k_r latch at posedge clk end of ADDR
//                       (full ADDR beat; macro CLK=~clk, Q valid next posedge).
//                       Phase 1 drives write from sp_q_r/sp_k_r; ptr++ on phase 1.
//                       Cycle count: HD_ELEMS * 2 = 20480.
//   S_K_MEAN         : km_phase. Phase 0 drives Sram_k read; phase 1 updates
//                       km_acc using s4_q_i. KM_ELEMS * N_TOKENS * 2 = 20480.
//   S_QK_MEAN        : qk_phase. Same pattern on Sram_q. * 2 = 20480.
//   S_KV             : kv_phase. Parallel Sram_k + Sram_v read (different macros).
//                       * 2 = 163840.
//   S_ATTN           : at_phase. Phase 0 Sram_q read at at_q_flat; phase 1 update
//                       at_acc, on at_dk == HEAD_DIM-1 also write ao to Sram_v.
//                       * 2 = 20480.
//   S_PROJ streaming : pj_sub (4-state: START/USE/ADDR/WAIT). Reads ao via Sram_v
//                       to feed lin_proj_x; ~2 cycles per beat.
//   S_QKV x read : qkv_x_phase + norm_rd_* (2-phase read parent tok1 norm1 staging).
//   S_Z_RECIP          : zr_phase on Sram_qkm (read qkm, write zr per index).
//   S_ATTN zr          : at_zr_r shadow from Sram_qkm read at at_dk==0.

//
// Golden activation files (Q8.8, one 16-bit binary per line, C-order flatten):
//   backbone_blocks_<b>_attn_after_qkv_q_bi.txt   (H,N,d) = 4*320*8 = 10240
//   backbone_blocks_<b>_attn_after_qkv_k_bi.txt   same shape
//   backbone_blocks_<b>_attn_after_qkv_v_bi.txt   same shape
//   backbone_blocks_<b>_after_attn_attn_out_bi.txt (N,C) = 320*32 = 10240
//
// ROM access: care_attention drives 13-bit wgt_addr_o (local addr).
//   S_QKV  : addr = 0..3071        (QKV weight, bias decoded from local[12:5])
//   S_PROJ : addr = 3072..4095     (PROJ weight, bias decoded from local[9:5])
//
// Sequential = `<=`, combinational = `=`. No latch inference (all outputs
// covered in case branches).
// =============================================================================

module care_attention #(
    parameter EMBED_DIM   = 32,
    parameter NUM_HEADS   = 4,
    parameter HEAD_DIM    = 8,         // EMBED_DIM / NUM_HEADS
    parameter N_TOKENS    = 320,
    parameter S_Q88       = 152,       // round(256 * HEAD_DIM^(-0.25))
    parameter RELU6_MAX   = 1536,      // 6.0 * 256
    parameter RCP_N_NUM   = 205,       // round(65536 / N_TOKENS)
    parameter RCP_N_SHIFT = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // 2-phase read of parent norm1 buffer (backbone Sram_tok1 staging)
    output reg         norm_rd_en,
    output reg  [13:0] norm_rd_flat,
    input  wire signed [15:0] norm_x,

    // ROM weight / bias (1-cycle latency from wgt_addr_o)
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [12:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    // Streaming output to next stage (after PROJ linear)
    output reg  signed [15:0] y_o,
    output reg         y_valid,

    // 1P SRAM port mux -> sglatrack_top (head2-style sram_*_ceb_o / sram_*_q_i)
    output wire        sram_q_ceb_o,
    output wire        sram_q_web_o,
    output wire [13:0] sram_q_addr_o,
    output wire [15:0] sram_q_din_o,
    input  wire [15:0] sram_q_q_i,

    output wire        sram_k_ceb_o,
    output wire        sram_k_web_o,
    output wire [13:0] sram_k_addr_o,
    output wire [15:0] sram_k_din_o,
    input  wire [15:0] sram_k_q_i,

    output wire        sram_v_ceb_o,
    output wire        sram_v_web_o,
    output wire [13:0] sram_v_addr_o,
    output wire [15:0] sram_v_din_o,
    input  wire [15:0] sram_v_q_i,

    output wire        sram_qkm_ceb_o,
    output wire        sram_qkm_web_o,
    output wire [13:0] sram_qkm_addr_o,
    output wire [15:0] sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i
);

// ---------------------------------------------------------------------------
// Derived parameters
// ---------------------------------------------------------------------------
parameter HD_ELEMS  = NUM_HEADS * N_TOKENS * HEAD_DIM;   // 10240
parameter KM_ELEMS  = NUM_HEADS * HEAD_DIM;              // 32
parameter QKM_ELEMS = NUM_HEADS * N_TOKENS;              // 1280
parameter KV_ELEMS  = NUM_HEADS * HEAD_DIM * HEAD_DIM;   // 256
parameter X_ELEMS   = N_TOKENS * EMBED_DIM;              // 10240

// ---------------------------------------------------------------------------
// FSM states (4-bit)
// ---------------------------------------------------------------------------
parameter S_IDLE    = 4'd0;
parameter S_QKV     = 4'd2;
parameter S_SPLIT   = 4'd3;
parameter S_K_MEAN  = 4'd4;
parameter S_QK_MEAN = 4'd5;
parameter S_Z_RECIP = 4'd6;
parameter S_KV      = 4'd7;
parameter S_ATTN    = 4'd8;
parameter S_PROJ    = 4'd9;
parameter S_DONE_ST = 4'd10;

// S_PROJ streaming sub-states (reads ao from Sram_v via 2-phase)
parameter PJ_START = 2'd0;
parameter PJ_ADDR  = 2'd1;
parameter PJ_USE   = 2'd2;
parameter PJ_WAIT  = 2'd3;

// ---------------------------------------------------------------------------
// Small reg scratch (km_buf / kv_buf; attention SRAM macros for q/k/v/ao/x)
// ---------------------------------------------------------------------------
reg signed [15:0] km_buf   [0:KM_ELEMS-1];
reg signed [15:0] kv_buf   [0:KV_ELEMS-1];

`ifndef SYNTHESIS
integer ca_ii;
initial begin
    for (ca_ii = 0; ca_ii < KM_ELEMS;  ca_ii = ca_ii + 1) km_buf [ca_ii] = 16'sd0;
    for (ca_ii = 0; ca_ii < KV_ELEMS;  ca_ii = ca_ii + 1) kv_buf[ca_ii] = 16'sd0;
end
`endif

// ---------------------------------------------------------------------------
// FSM state regs
// ---------------------------------------------------------------------------
reg [3:0] state, next_state;

// ---------------------------------------------------------------------------
// Counters / sub-state regs (declarations precede always blocks)
// ---------------------------------------------------------------------------
reg [8:0]  qx_tok;
reg [5:0]  qkv_stream_cnt;
reg [13:0] sp_ptr;              // S_SPLIT pointer 0..HD_ELEMS-1

reg [4:0]  km_oidx;             // S_K_MEAN outer (h*HEAD_DIM + d) 0..KM_ELEMS-1
reg [8:0]  km_n;                // S_K_MEAN inner sum 0..N_TOKENS-1
reg signed [32:0] km_acc;       // 33-bit signed (320 x 16-bit fits in 25)

reg [10:0] qk_oidx;             // S_QK_MEAN outer (h*N + n) 0..QKM_ELEMS-1
reg [2:0]  qk_d;                // S_QK_MEAN inner d 0..HEAD_DIM-1
reg signed [32:0] qk_acc;       // 33-bit signed accumulator

reg [10:0] zr_idx;              // S_Z_RECIP index 0..QKM_ELEMS-1

reg [7:0]  kv_oidx;             // S_KV outer (h*HEAD_DIM*HEAD_DIM + d1*d + d2)
reg [8:0]  kv_n;                // S_KV inner n 0..N_TOKENS-1
reg signed [47:0] kv_acc;       // 48-bit signed accumulator

reg [13:0] at_oidx;             // S_ATTN outer (h*N*d + n*d + d_out)
reg [2:0]  at_dk;               // S_ATTN inner d_k 0..HEAD_DIM-1
reg signed [48:0] at_acc;       // 49-bit signed accumulator

reg [8:0]  px_tok;              // S_PROJ: token 0..N_TOKENS-1
reg [5:0]  proj_stream_cnt;     // S_PROJ streaming phase (0..32)

// ---------------------------------------------------------------------------
// 2-phase ADDR/USE phase regs (scalar) and shadow regs (scalar 16-bit)
//   No 2D regs are introduced (compliance with SRAM_suggestion.md 5.1 rule #2).
// ---------------------------------------------------------------------------
reg               sp_phase;     // S_SPLIT  : 0 = ADDR (read), 1 = USE (write)
reg signed [15:0] sp_q_r;       // S_SPLIT  : shadow for s3_q_i (q_buf read)
reg signed [15:0] sp_k_r;       // S_SPLIT  : shadow for s4_q_i (k_buf read)
reg               km_phase;     // S_K_MEAN : 0 = ADDR, 1 = USE
reg               qk_phase;     // S_QK_MEAN: 0 = ADDR, 1 = USE
reg               kv_phase;     // S_KV     : 0 = ADDR, 1 = USE
reg               at_phase;     // S_ATTN   : 0 = ADDR, 1 = USE
reg [1:0]         pj_sub;       // S_PROJ streaming sub-FSM (PJ_*)
reg               qkv_x_phase;  // S_QKV    : 0 = ADDR read norm1 (parent tmp), 1 = USE
reg               zr_phase;     // S_Z_RECIP: 0 = ADDR read qkm, 1 = launch recip
reg signed [15:0] zr_qkm_r;     // S_Z_RECIP: latched qkm read for recip_x
reg signed [15:0] at_zr_r;      // S_ATTN   : latched zr read (at_dk==0 ADDR beat)

// ---------------------------------------------------------------------------
// QK_MEAN outer (h,n) explicit counters (N=320 not power of two)
// ---------------------------------------------------------------------------
reg [1:0] qk_h_reg;
reg [8:0] qk_n_reg;

// ---------------------------------------------------------------------------
// S_ATTN outer counters (h,n,d_out) explicit (same reason)
// ---------------------------------------------------------------------------
reg [1:0] at_h_reg;
reg [8:0] at_n_reg;
reg [2:0] at_dout_reg;

// ---------------------------------------------------------------------------
// SRAM port driver regs (drive macros in always @(*) below)
// ---------------------------------------------------------------------------
reg         s3_ceb, s3_web;
reg [13:0]  s3_addr;
reg [15:0]  s3_din;
wire [15:0] s3_q;

reg         s4_ceb, s4_web;
reg [13:0]  s4_addr;
reg [15:0]  s4_din;
wire [15:0] s4_q;

reg         s5_ceb, s5_web;
reg [13:0]  s5_addr;
reg [15:0]  s5_din;
wire [15:0] s5_q;

reg         s6_ceb, s6_web;
reg [10:0]  s6_addr;
reg [15:0]  s6_din;
wire [15:0] s6_q;

assign sram_q_ceb_o   = s3_ceb;
assign sram_q_web_o   = s3_web;
assign sram_q_addr_o  = s3_addr;
assign sram_q_din_o   = s3_din;
assign s3_q           = sram_q_q_i;

assign sram_k_ceb_o   = s4_ceb;
assign sram_k_web_o   = s4_web;
assign sram_k_addr_o  = s4_addr;
assign sram_k_din_o   = s4_din;
assign s4_q           = sram_k_q_i;

assign sram_v_ceb_o   = s5_ceb;
assign sram_v_web_o   = s5_web;
assign sram_v_addr_o  = s5_addr;
assign sram_v_din_o   = s5_din;
assign s5_q           = sram_v_q_i;

assign sram_qkm_ceb_o  = s6_ceb;
assign sram_qkm_web_o  = s6_web;
assign sram_qkm_addr_o = {3'b000, s6_addr};
assign sram_qkm_din_o  = s6_din;
assign s6_q            = sram_qkm_q_i;

// ---------------------------------------------------------------------------
// recip_nr driver regs (small)
// ---------------------------------------------------------------------------
reg                recip_start;
reg  signed [15:0] recip_x;
wire               recip_busy, recip_done;
wire signed [15:0] recip_y;

// ---------------------------------------------------------------------------
// Linear instance driver regs (S_QKV and S_PROJ)
// ---------------------------------------------------------------------------
reg                lin_qkv_start;
wire               lin_qkv_busy, lin_qkv_done;
reg  signed [15:0] lin_qkv_x;
reg                lin_qkv_xv;
wire signed [15:0] lin_qkv_y;
wire               lin_qkv_yv;
wire [6:0]         lin_qkv_neu;
wire [12:0]        lin_qkv_addr;

reg                lin_proj_start;
wire               lin_proj_busy, lin_proj_done;
reg  signed [15:0] lin_proj_x;
reg                lin_proj_xv;
wire signed [15:0] lin_proj_y;
wire               lin_proj_yv;
wire [6:0]         lin_proj_neu;
wire [12:0]        lin_proj_addr;

// ---------------------------------------------------------------------------
// Helper functions: rounding / saturation (mirror numpy fp() / sat16_*)
// ---------------------------------------------------------------------------
function signed [15:0] sat16_q88_32;
    input signed [31:0] v;
    begin
        if (v > 32'sd32767)       sat16_q88_32 = 16'sh7FFF;
        else if (v < -32'sd32768) sat16_q88_32 = 16'sh8000;
        else                       sat16_q88_32 = v[15:0];
    end
endfunction

function signed [15:0] sat16_q88_33;
    input signed [32:0] v;
    begin
        if (v > 33'sd32767)       sat16_q88_33 = 16'sh7FFF;
        else if (v < -33'sd32768) sat16_q88_33 = 16'sh8000;
        else                       sat16_q88_33 = v[15:0];
    end
endfunction

function signed [15:0] sat16_q88_48;
    input signed [47:0] v;
    begin
        if (v > 48'sd32767)       sat16_q88_48 = 16'sh7FFF;
        else if (v < -48'sd32768) sat16_q88_48 = 16'sh8000;
        else                       sat16_q88_48 = v[15:0];
    end
endfunction

function signed [15:0] sat16_q88_49;
    input signed [48:0] v;
    begin
        if (v > 49'sd32767)       sat16_q88_49 = 16'sh7FFF;
        else if (v < -49'sd32768) sat16_q88_49 = 16'sh8000;
        else                       sat16_q88_49 = v[15:0];
    end
endfunction

function signed [15:0] sat16_q88_64;
    input signed [63:0] v;
    begin
        if (v > 64'sd32767)       sat16_q88_64 = 16'sh7FFF;
        else if (v < -64'sd32768) sat16_q88_64 = 16'sh8000;
        else                       sat16_q88_64 = v[15:0];
    end
endfunction

// rnd_shr8: (v + 128) >>> 8 then sat16. Mirrors numpy rnd_shr8.
function signed [15:0] rnd_shr8_q88;
    input signed [31:0] v;
    reg signed [31:0] t;
    begin
        t = v + 32'sd128;
        rnd_shr8_q88 = sat16_q88_32(t >>> 8);
    end
endfunction

// relu6 in Q8.8: clip [0, RELU6_MAX]
function signed [15:0] relu6_q88;
    input signed [15:0] x;
    begin
        if (x[15])                       relu6_q88 = 16'sd0;
        else if ($signed(x) > RELU6_MAX) relu6_q88 = RELU6_MAX[15:0];
        else                              relu6_q88 = x;
    end
endfunction

// ---------------------------------------------------------------------------
// Linear instances
// ---------------------------------------------------------------------------
linear #(.IN_DIM(EMBED_DIM), .OUT_DIM(3*EMBED_DIM)) u_lin_qkv (
    .clk     (clk),
    .reset   (reset),
    .start   (lin_qkv_start),
    .x_i     (lin_qkv_x),
    .x_valid (lin_qkv_xv),
    .w_i     (wgt_i),
    .b_i     (bias_i),
    .w_addr_o(lin_qkv_addr),
    .busy    (lin_qkv_busy),
    .done    (lin_qkv_done),
    .y_o     (lin_qkv_y),
    .y_valid (lin_qkv_yv),
    .y_neu_o (lin_qkv_neu)
);

linear #(.IN_DIM(EMBED_DIM), .OUT_DIM(EMBED_DIM)) u_lin_proj (
    .clk     (clk),
    .reset   (reset),
    .start   (lin_proj_start),
    .x_i     (lin_proj_x),
    .x_valid (lin_proj_xv),
    .w_i     (wgt_i),
    .b_i     (bias_i),
    .w_addr_o(lin_proj_addr),
    .busy    (lin_proj_busy),
    .done    (lin_proj_done),
    .y_o     (lin_proj_y),
    .y_valid (lin_proj_yv),
    .y_neu_o (lin_proj_neu)
);

// ROM addr mux for the two linears; other states output 0
assign wgt_addr_o = (state == S_QKV)  ? lin_qkv_addr :
                    (state == S_PROJ) ? (13'd3072 + lin_proj_addr) :
                                        13'd0;

// ---------------------------------------------------------------------------
// recip_nr instance (for S_Z_RECIP)
// ---------------------------------------------------------------------------
recip_nr u_recip (
    .clk  (clk),
    .reset(reset),
    .start(recip_start),
    .x_i  (recip_x),
    .busy (recip_busy),
    .done (recip_done),
    .y_o  (recip_y)
);

// ---------------------------------------------------------------------------
// Index decoders (combinational)
//   Source of data for accumulators:
//     Data from s3_q / s4_q / s5_q (1-cycle delayed read)
//   In SRAM mode, the *_phase regs ensure compute happens on the USE cycle
//   when the macro Q is stable.
// ---------------------------------------------------------------------------
// S_K_MEAN outer (h,d)
wire [1:0] km_h = km_oidx[4:3];
wire [2:0] km_d = km_oidx[2:0];
wire [13:0] km_k_flat =
    {12'd0, km_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, km_n}) * HEAD_DIM
  + {11'd0, km_d};
wire signed [15:0] km_k_data = s4_q;
wire signed [32:0] km_acc_next =
    (km_n == 9'd0) ? $signed({{17{km_k_data[15]}}, km_k_data})
                   : km_acc + $signed({{17{km_k_data[15]}}, km_k_data});
wire signed [47:0] km_scaled =
    $signed({{15{km_acc_next[32]}}, km_acc_next}) * $signed({32'd0, RCP_N_NUM[15:0]});
wire signed [47:0] km_shr_w  = (km_scaled + 48'sd32768) >>> RCP_N_SHIFT;

// S_QK_MEAN
wire [13:0] qk_q_flat =
    {12'd0, qk_h_reg} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, qk_n_reg}) * HEAD_DIM
  + {11'd0, qk_d};
wire [4:0] qk_km_flat = {qk_h_reg, qk_d};   // h*HEAD_DIM + d (HEAD_DIM=8)
wire signed [15:0] qk_q_data = s3_q;
wire signed [31:0] qk_term =
    $signed(qk_q_data) * $signed(km_buf[qk_km_flat]);
wire signed [32:0] qk_acc_next =
    (qk_d == 3'd0) ? $signed({qk_term[31], qk_term})
                   : qk_acc + $signed({qk_term[31], qk_term});
wire signed [32:0] qk_rounded = qk_acc_next + 33'sd128;
wire signed [15:0] qk_sat = sat16_q88_33(qk_rounded >>> 8);
wire signed [15:0] qkm_wr_val = (qk_sat < 16'sd1) ? 16'sd1 : qk_sat;

// S_KV outer (h,d1,d2)
wire [1:0] kv_h  = kv_oidx[7:6];
wire [2:0] kv_d1 = kv_oidx[5:3];
wire [2:0] kv_d2 = kv_oidx[2:0];
wire [13:0] kv_k_flat =
    {12'd0, kv_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, kv_n}) * HEAD_DIM
  + {11'd0, kv_d1};
wire [13:0] kv_v_flat =
    {12'd0, kv_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, kv_n}) * HEAD_DIM
  + {11'd0, kv_d2};
wire signed [15:0] kv_k_data = s4_q;
wire signed [15:0] kv_v_data = s5_q;
wire signed [31:0] kv_term = $signed(kv_k_data) * $signed(kv_v_data);
wire signed [48:0] kv_acc_next =
    (kv_n == 9'd0) ? $signed({{17{kv_term[31]}}, kv_term})
                   : kv_acc + $signed({{17{kv_term[31]}}, kv_term});
wire signed [63:0] kv_scaled =
    $signed({{15{kv_acc_next[48]}}, kv_acc_next}) * $signed({48'd0, RCP_N_NUM[15:0]});
wire signed [63:0] kv_shr_w = (kv_scaled + 64'sd8388608) >>> (RCP_N_SHIFT + 8);

// S_ATTN
wire [13:0] at_q_flat =
    {12'd0, at_h_reg} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, at_n_reg}) * HEAD_DIM
  + {11'd0, at_dk};
wire [7:0] at_kv_flat =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM)
  + {5'd0, at_dk} * HEAD_DIM
  + {5'd0, at_dout_reg};
wire [10:0] at_zr_idx = ({2'd0, at_h_reg} * N_TOKENS) + {2'd0, at_n_reg};
wire signed [15:0] at_q_data = s3_q;
wire signed [31:0] at_term =
    $signed(at_q_data) * $signed(kv_buf[at_kv_flat]);
wire signed [48:0] at_acc_next =
    (at_dk == 3'd0) ? $signed({{17{at_term[31]}}, at_term})
                    : at_acc + $signed({{17{at_term[31]}}, at_term});
// fp #1: round dot sum once
wire signed [15:0] at_dot_sat = sat16_q88_49((at_acc_next + 49'sd128) >>> 8);
// fp #2: rnd_shr8(dot_sat * zr[h*N + n])
wire signed [15:0] at_zr_data = at_zr_r;
wire signed [31:0] at_zprod = $signed(at_dot_sat) * $signed(at_zr_data);
wire signed [15:0] at_ao_val = rnd_shr8_q88(at_zprod);
// destination flat in ao_buf
wire [13:0] at_ao_flat =
    ({5'd0, at_n_reg}) * EMBED_DIM
  + {12'd0, at_h_reg} * HEAD_DIM
  + {11'd0, at_dout_reg};

// S_QKV norm1 input flat (token-major, parent tok1 staging)
wire [13:0] qkv_x_flat =
    ({5'd0, qx_tok}) * EMBED_DIM + {9'd0, qkv_stream_cnt - 6'd1};

// S_QKV capture address (used by SRAM mux on lin_qkv_yv pulse; combinational)
wire [13:0] qkv_cap_flat_w =
    {12'd0, lin_qkv_neu[4:3]} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, qx_tok}) * HEAD_DIM
  + {11'd0, lin_qkv_neu[2:0]};

// S_PROJ streaming address (ao read from Sram_v during PJ_START/PJ_ADDR)
wire [13:0] pj_ao_addr =
    {5'd0, px_tok} * EMBED_DIM + {9'd0, proj_stream_cnt};

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
//   In SRAM mode, advance to next state requires phase==1 (USE) so that the
//   final iteration's compute has actually executed. Same termination index
//   as the legacy reg path; just gated by phase to enforce 2-cycle pacing.
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:    next_state = start ? S_QKV : S_IDLE;
        S_QKV:     next_state = (lin_qkv_done && qx_tok == N_TOKENS[8:0] - 9'd1) ? S_SPLIT : S_QKV;
        S_SPLIT:   next_state = (sp_ptr == HD_ELEMS[13:0] - 14'd1 && sp_phase == 1'b1) ? S_K_MEAN : S_SPLIT;
        S_K_MEAN:  next_state = (km_oidx == KM_ELEMS[4:0] - 5'd1 && km_n == N_TOKENS[8:0] - 9'd1 && km_phase == 1'b1) ? S_QK_MEAN : S_K_MEAN;
        S_QK_MEAN: next_state = (qk_oidx == QKM_ELEMS[10:0] - 11'd1 && qk_d == HEAD_DIM[2:0] - 3'd1 && qk_phase == 1'b1) ? S_Z_RECIP : S_QK_MEAN;
        S_Z_RECIP: next_state = (recip_done && zr_idx == QKM_ELEMS[10:0] - 11'd1) ? S_KV : S_Z_RECIP;
        S_KV:      next_state = (kv_oidx == KV_ELEMS[7:0] - 8'd1 && kv_n == N_TOKENS[8:0] - 9'd1 && kv_phase == 1'b1) ? S_ATTN : S_KV;
        S_ATTN:    next_state = (at_oidx == HD_ELEMS[13:0] - 14'd1 && at_dk == HEAD_DIM[2:0] - 3'd1 && at_phase == 1'b1) ? S_PROJ : S_ATTN;
        S_PROJ:    next_state = (lin_proj_done && px_tok == N_TOKENS[8:0] - 9'd1) ? S_DONE_ST : S_PROJ;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// Sequential capture of SRAM Q into shadow regs (S_SPLIT only).
//   Latch at posedge while sp_phase==0 (end of full ADDR beat): s3_q/s4_q hold
//   the read for sp_ptr issued during that beat (1-cycle macro latency).
//   USE phase (sp_phase==1) write din uses sp_q_r/sp_k_r, not live Q on write.
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        sp_q_r <= 16'sd0;
        sp_k_r <= 16'sd0;
    end else if (state == S_SPLIT && sp_phase == 1'b0) begin
        sp_q_r <= s3_q;
        sp_k_r <= s4_q;
    end
end

// S_Z_RECIP: latch qkm read before launching recip_nr.
always @(posedge clk) begin
    if (reset)
        zr_qkm_r <= 16'sd0;
    else if (state == S_Z_RECIP && zr_phase == 1'b0)
        zr_qkm_r <= s6_q;
end

// S_ATTN: latch zr_buf[at_zr_idx] once per (h,n,d_out) at at_dk==0 ADDR beat.
always @(posedge clk) begin
    if (reset)
        at_zr_r <= 16'sd0;
    else if (state == S_ATTN && at_phase == 1'b0 && at_dk == 3'd0)
        at_zr_r <= s6_q;
end

// ---------------------------------------------------------------------------
// FSM segment 3: main datapath
//   In SRAM mode every iteration of S_SPLIT / S_K_MEAN / S_QK_MEAN / S_KV /
//   S_ATTN takes 2 cycles: phase 0 (ADDR) drives the read addr via the mux
//   block below and does NOT advance counters; phase 1 (USE) consumes
//   s3_q / s4_q / s5_q and advances counters. Reg-array writes (where they
//   happen on phase 1 (USE), when macro Q is stable.
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    // Defaults each cycle
    done           <= 1'b0;
    y_valid        <= 1'b0;
    lin_qkv_start  <= 1'b0;
    lin_proj_start <= 1'b0;
    recip_start    <= 1'b0;
    lin_qkv_xv     <= 1'b0;
    lin_proj_xv    <= 1'b0;
    norm_rd_en     <= 1'b0;
    norm_rd_flat   <= 14'd0;

    if (reset) begin
        qx_tok          <= 9'd0;
        qkv_stream_cnt  <= 6'd0;
        sp_ptr          <= 14'd0;
        sp_phase        <= 1'b0;
        km_oidx         <= 5'd0;
        km_n            <= 9'd0;
        km_acc          <= 33'sd0;
        km_phase        <= 1'b0;
        qk_oidx         <= 11'd0;
        qk_h_reg        <= 2'd0;
        qk_n_reg        <= 9'd0;
        qk_d            <= 3'd0;
        qk_acc          <= 33'sd0;
        qk_phase        <= 1'b0;
        zr_idx          <= 11'd0;
        kv_oidx         <= 8'd0;
        kv_n            <= 9'd0;
        kv_acc          <= 48'sd0;
        kv_phase        <= 1'b0;
        at_oidx         <= 14'd0;
        at_h_reg        <= 2'd0;
        at_n_reg        <= 9'd0;
        at_dout_reg     <= 3'd0;
        at_dk           <= 3'd0;
        at_acc          <= 49'sd0;
        at_phase        <= 1'b0;
        px_tok          <= 9'd0;
        proj_stream_cnt <= 6'd0;
        pj_sub          <= PJ_START;
        qkv_x_phase     <= 1'b0;
        zr_phase        <= 1'b0;
        zr_qkm_r        <= 16'sd0;
        at_zr_r         <= 16'sd0;
        y_o             <= 16'sd0;
    end else begin
        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                qx_tok          <= 9'd0;
                qkv_stream_cnt  <= 6'd0;
                qkv_x_phase     <= 1'b0;
                sp_ptr          <= 14'd0;
                sp_phase        <= 1'b0;
                km_oidx         <= 5'd0;
                km_n            <= 9'd0;
                km_phase        <= 1'b0;
                qk_oidx         <= 11'd0;
                qk_h_reg        <= 2'd0;
                qk_n_reg        <= 9'd0;
                qk_d            <= 3'd0;
                qk_phase        <= 1'b0;
                zr_idx          <= 11'd0;
                kv_oidx         <= 8'd0;
                kv_n            <= 9'd0;
                kv_phase        <= 1'b0;
                at_oidx         <= 14'd0;
                at_h_reg        <= 2'd0;
                at_n_reg        <= 9'd0;
                at_dout_reg     <= 3'd0;
                at_dk           <= 3'd0;
                at_phase        <= 1'b0;
                px_tok          <= 9'd0;
                proj_stream_cnt <= 6'd0;
                pj_sub          <= PJ_START;
                zr_phase        <= 1'b0;
            end

            // -----------------------------------------------------------
            // S_QKV: 2-phase read norm1 from parent tok1; q/k/v -> SRAM.
            // Must finish before S_PROJ (parent overwrites tmp with attn out).
            // -----------------------------------------------------------
            S_QKV: begin
                if (qkv_stream_cnt == 6'd0) begin
                    lin_qkv_start  <= 1'b1;
                    qkv_stream_cnt <= 6'd1;
                    qkv_x_phase    <= 1'b0;
                end else if (qkv_stream_cnt <= EMBED_DIM[5:0]) begin
                    if (qkv_x_phase == 1'b0) begin
                        norm_rd_en   <= 1'b1;
                        norm_rd_flat <= qkv_x_flat;
                        qkv_x_phase  <= 1'b1;
                    end else begin
                        lin_qkv_x        <= norm_x;
                        lin_qkv_xv       <= 1'b1;
                        qkv_stream_cnt   <= qkv_stream_cnt + 6'd1;
                        qkv_x_phase      <= 1'b0;
                    end
                end

                if (lin_qkv_done) begin
                    if (qx_tok == N_TOKENS[8:0] - 9'd1) begin
                        qx_tok         <= 9'd0;
                        qkv_stream_cnt <= 6'd0;
                        sp_ptr         <= 14'd0;
                        sp_phase       <= 1'b0;
                    end else begin
                        qx_tok         <= qx_tok + 9'd1;
                        qkv_stream_cnt <= 6'd0;
                    end
                end
            end

            // -----------------------------------------------------------
            // S_SPLIT: 2-phase RW on Sram_q + Sram_k at sp_ptr.
            //   phase 0 (ADDR): SRAM mux drives read addr = sp_ptr
            //   phase 1 (USE) : SRAM mux drives write addr = sp_ptr with
            //                    relu6(rnd_shr8(sp_q_r * S_Q88)). Advance sp_ptr.
            // Phase 1 writes scaled q/k back to Sram_q / Sram_k (via mux).
            // -----------------------------------------------------------
            S_SPLIT: begin
                if (sp_phase == 1'b0) begin
                    sp_phase <= 1'b1;
                end else begin
                    if (sp_ptr < HD_ELEMS[13:0] - 14'd1) begin
                        sp_ptr   <= sp_ptr + 14'd1;
                        sp_phase <= 1'b0;
                    end
                    // On final entry: sp_phase stays 1, next_state transitions
                end
                if (next_state == S_K_MEAN) begin
                    km_oidx  <= 5'd0;
                    km_n     <= 9'd0;
                    km_acc   <= 33'sd0;
                    km_phase <= 1'b0;
                end
            end

            // -----------------------------------------------------------
            // S_K_MEAN: 2-phase read of k.
            //   phase 0 (ADDR): SRAM mux drives s4_addr = km_k_flat
            //   phase 1 (USE) : km_acc <= km_acc_next (uses km_k_data which is
            //                    s4_q in SRAM mode, k_buf[km_k_flat] in reg
            //                    mode); on last inner n, write km_buf and
            //                    advance km_oidx.
            // -----------------------------------------------------------
            S_K_MEAN: begin
                if (km_phase == 1'b0) begin
                    km_phase <= 1'b1;
                end else begin
                    km_acc <= km_acc_next;
                    if (km_n == N_TOKENS[8:0] - 9'd1) begin
                        km_buf[km_oidx] <= sat16_q88_48(km_shr_w);
                        km_n <= 9'd0;
                        if (km_oidx == KM_ELEMS[4:0] - 5'd1) begin
                            km_oidx <= 5'd0;
                        end else begin
                            km_oidx <= km_oidx + 5'd1;
                        end
                    end else begin
                        km_n <= km_n + 9'd1;
                    end
                    km_phase <= 1'b0;
                end
                if (next_state == S_QK_MEAN) begin
                    qk_oidx  <= 11'd0;
                    qk_h_reg <= 2'd0;
                    qk_n_reg <= 9'd0;
                    qk_d     <= 3'd0;
                    qk_acc   <= 33'sd0;
                    qk_phase <= 1'b0;
                end
            end

            // -----------------------------------------------------------
            // S_QK_MEAN: 2-phase read of q. On last inner d, write qkm_buf.
            // -----------------------------------------------------------
            S_QK_MEAN: begin
                if (qk_phase == 1'b0) begin
                    qk_phase <= 1'b1;
                end else begin
                    qk_acc <= qk_acc_next;
                    if (qk_d == HEAD_DIM[2:0] - 3'd1) begin
                        qk_d <= 3'd0;
                        if (qk_oidx == QKM_ELEMS[10:0] - 11'd1) begin
                            qk_oidx  <= 11'd0;
                            qk_h_reg <= 2'd0;
                            qk_n_reg <= 9'd0;
                        end else begin
                            qk_oidx <= qk_oidx + 11'd1;
                            if (qk_n_reg == N_TOKENS[8:0] - 9'd1) begin
                                qk_n_reg <= 9'd0;
                                qk_h_reg <= qk_h_reg + 2'd1;
                            end else begin
                                qk_n_reg <= qk_n_reg + 9'd1;
                            end
                        end
                    end else begin
                        qk_d <= qk_d + 3'd1;
                    end
                    qk_phase <= 1'b0;
                end
                if (next_state == S_Z_RECIP) begin
                    zr_idx   <= 11'd0;
                    zr_phase <= 1'b0;
                end
            end

            // -----------------------------------------------------------
            // S_Z_RECIP: qkm/zr on Sram_qkm (2-phase read qkm, write zr on done).
            //   Does not touch Sram_q/k/v.
            // -----------------------------------------------------------
            S_Z_RECIP: begin
                if (recip_done) begin
                    if (zr_idx == QKM_ELEMS[10:0] - 11'd1) begin
                        zr_idx <= 11'd0;
                    end else begin
                        zr_idx <= zr_idx + 11'd1;
                    end
                    zr_phase <= 1'b0;
                end else if (!recip_busy) begin
                    if (zr_phase == 1'b0)
                        zr_phase <= 1'b1;
                    else begin
                        recip_start <= 1'b1;
                        recip_x     <= zr_qkm_r;
                    end
                end
                if (next_state == S_KV) begin
                    kv_oidx  <= 8'd0;
                    kv_n     <= 9'd0;
                    kv_acc   <= 48'sd0;
                    kv_phase <= 1'b0;
                end
            end

            // -----------------------------------------------------------
            // S_KV: 2-phase parallel read of Sram_k + Sram_v (different macros,
            // no cross-SRAM conflict). On last inner n, write kv_buf (reg).
            // -----------------------------------------------------------
            S_KV: begin
                if (kv_phase == 1'b0) begin
                    kv_phase <= 1'b1;
                end else begin
                    kv_acc <= kv_acc_next;
                    if (kv_n == N_TOKENS[8:0] - 9'd1) begin
                        kv_buf[kv_oidx] <= sat16_q88_64(kv_shr_w);
                        kv_n <= 9'd0;
                        if (kv_oidx == KV_ELEMS[7:0] - 8'd1) begin
                            kv_oidx <= 8'd0;
                        end else begin
                            kv_oidx <= kv_oidx + 8'd1;
                        end
                    end else begin
                        kv_n <= kv_n + 9'd1;
                    end
                    kv_phase <= 1'b0;
                end
                if (next_state == S_ATTN) begin
                    at_oidx     <= 14'd0;
                    at_h_reg    <= 2'd0;
                    at_n_reg    <= 9'd0;
                    at_dout_reg <= 3'd0;
                    at_dk       <= 3'd0;
                    at_acc      <= 49'sd0;
                    at_phase    <= 1'b0;
                end
            end

            // -----------------------------------------------------------
            // S_ATTN: 2-phase read of q (Sram_q). On phase 1 with at_dk ==
            // HEAD_DIM-1, also write ao to Sram_v at at_ao_flat (cross-SRAM,
            // no R+W conflict on either macro since q-read and ao-write target
            // different macros).
            // Phase 1: write attention output ao to Sram_v (mux).
            // -----------------------------------------------------------
            S_ATTN: begin
                if (at_phase == 1'b0) begin
                    at_phase <= 1'b1;
                end else begin
                    at_acc <= at_acc_next;
                    if (at_dk == HEAD_DIM[2:0] - 3'd1) begin
                        at_dk <= 3'd0;
                        if (at_oidx == HD_ELEMS[13:0] - 14'd1) begin
                            at_oidx     <= 14'd0;
                            at_h_reg    <= 2'd0;
                            at_n_reg    <= 9'd0;
                            at_dout_reg <= 3'd0;
                        end else begin
                            at_oidx <= at_oidx + 14'd1;
                            if (at_dout_reg == HEAD_DIM[2:0] - 3'd1) begin
                                at_dout_reg <= 3'd0;
                                if (at_n_reg == N_TOKENS[8:0] - 9'd1) begin
                                    at_n_reg <= 9'd0;
                                    at_h_reg <= at_h_reg + 2'd1;
                                end else begin
                                    at_n_reg <= at_n_reg + 9'd1;
                                end
                            end else begin
                                at_dout_reg <= at_dout_reg + 3'd1;
                            end
                        end
                    end else begin
                        at_dk <= at_dk + 3'd1;
                    end
                    at_phase <= 1'b0;
                end
                if (next_state == S_PROJ) begin
                    px_tok          <= 9'd0;
                    proj_stream_cnt <= 6'd0;
                    pj_sub          <= PJ_START;
                end
            end

            // -----------------------------------------------------------
            // S_PROJ: per token, drive u_lin_proj with 32 beats of ao.
            //   pj_sub:
            //     PJ_START : pulse lin_proj_start; mux drives s5_addr = ao[tok,0]
            //     PJ_ADDR  : mux drives s5_addr = ao[tok, cnt]
            //     PJ_USE   : s5_q valid -> capture into lin_proj_x; xv = 1
            //     PJ_WAIT  : streaming done; wait lin_proj_done
            // -----------------------------------------------------------
            S_PROJ: begin
                case (pj_sub)
                    PJ_START: begin
                        lin_proj_start  <= 1'b1;
                        proj_stream_cnt <= 6'd0;     // first beat ao[tok, 0]
                        pj_sub          <= PJ_USE;
                    end
                    PJ_ADDR: begin
                        pj_sub <= PJ_USE;
                    end
                    PJ_USE: begin
                        lin_proj_x <= s5_q;
                        lin_proj_xv <= 1'b1;
                        if (proj_stream_cnt == EMBED_DIM[5:0] - 6'd1) begin
                            pj_sub <= PJ_WAIT;
                        end else begin
                            proj_stream_cnt <= proj_stream_cnt + 6'd1;
                            pj_sub          <= PJ_ADDR;
                        end
                    end
                    PJ_WAIT: begin
                        // wait lin_proj_done
                    end
                    default: ;
                endcase

                // Forward linear output to module y_o (one beat per neuron)
                if (lin_proj_yv) begin
                    y_o     <= lin_proj_y;
                    y_valid <= 1'b1;
                end

                if (lin_proj_done) begin
                    if (px_tok == N_TOKENS[8:0] - 9'd1) begin
                        px_tok          <= 9'd0;
                        proj_stream_cnt <= 6'd0;
                    end else begin
                        px_tok          <= px_tok + 9'd1;
                        proj_stream_cnt <= 6'd0;
                        pj_sub          <= PJ_START;
                    end
                end
            end

            // -----------------------------------------------------------
            S_DONE_ST: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// SRAM port mux (combinational). Norm1 read is parent tok1 (norm_rd_*).
// ---------------------------------------------------------------------------
//   Defaults: deselect all 3 macros (CEB = 1).
//   Per state, drive read or write as needed. At most one operation per macro
//   per cycle (single-port discipline). Phase regs ensure same-macro R and W
//   never coincide.
// ---------------------------------------------------------------------------
always @(*) begin
    // Defaults: deselect
    s3_ceb = 1'b1; s3_web = 1'b1; s3_addr = 14'd0; s3_din = 16'd0;
    s4_ceb = 1'b1; s4_web = 1'b1; s4_addr = 14'd0; s4_din = 16'd0;
    s5_ceb = 1'b1; s5_web = 1'b1; s5_addr = 14'd0; s5_din = 16'd0;
    s6_ceb = 1'b1; s6_web = 1'b1; s6_addr = 11'd0; s6_din = 16'd0;

    case (state)
        // ---- S_QKV: capture q/k/v to s3/s4/s5 (norm1 read via parent tmp) ----
        S_QKV: begin
            if (lin_qkv_yv) begin
                case (lin_qkv_neu[6:5])
                    2'b00: begin   // q -> s3
                        s3_ceb  = 1'b0;
                        s3_web  = 1'b0;
                        s3_addr = qkv_cap_flat_w;
                        s3_din  = lin_qkv_y;
                    end
                    2'b01: begin   // k -> s4
                        s4_ceb  = 1'b0;
                        s4_web  = 1'b0;
                        s4_addr = qkv_cap_flat_w;
                        s4_din  = lin_qkv_y;
                    end
                    2'b10: begin   // v -> s5
                        s5_ceb  = 1'b0;
                        s5_web  = 1'b0;
                        s5_addr = qkv_cap_flat_w;
                        s5_din  = lin_qkv_y;
                    end
                    default: ;
                endcase
            end
        end

        // ---- S_SPLIT: phase 0 read s3+s4 at sp_ptr; phase 1 write back ----
        S_SPLIT: begin
            if (sp_phase == 1'b0) begin
                s3_ceb  = 1'b0; s3_web = 1'b1; s3_addr = sp_ptr;
                s4_ceb  = 1'b0; s4_web = 1'b1; s4_addr = sp_ptr;
            end else begin
                s3_ceb  = 1'b0; s3_web = 1'b0; s3_addr = sp_ptr;
                s3_din  = relu6_q88(rnd_shr8_q88($signed(sp_q_r) * S_Q88));
                s4_ceb  = 1'b0; s4_web = 1'b0; s4_addr = sp_ptr;
                s4_din  = relu6_q88(rnd_shr8_q88($signed(sp_k_r) * S_Q88));
            end
        end

        // ---- S_K_MEAN: continuous read of s4 at km_k_flat (both phases) ----
        S_K_MEAN: begin
            s4_ceb  = 1'b0;
            s4_web  = 1'b1;
            s4_addr = km_k_flat;
        end

        // ---- S_QK_MEAN: read s3; on last d write qkm to s6 ----
        S_QK_MEAN: begin
            s3_ceb  = 1'b0;
            s3_web  = 1'b1;
            s3_addr = qk_q_flat;
            if ((qk_phase == 1'b1) && (qk_d == HEAD_DIM[2:0] - 3'd1)) begin
                s6_ceb  = 1'b0;
                s6_web  = 1'b0;
                s6_addr = qk_oidx[10:0];
                s6_din  = qkm_wr_val;
            end
        end

        // ---- S_Z_RECIP: on recip_done write zr; else phase0 read qkm ----
        S_Z_RECIP: begin
            if (recip_done) begin
                s6_ceb  = 1'b0;
                s6_web  = 1'b0;
                s6_addr = zr_idx[10:0];
                s6_din  = recip_y;
            end else if (zr_phase == 1'b0) begin
                s6_ceb  = 1'b0;
                s6_web  = 1'b1;
                s6_addr = zr_idx[10:0];
            end
        end

        // ---- S_KV: parallel read of s4 (k) and s5 (v) ----
        S_KV: begin
            s4_ceb  = 1'b0; s4_web = 1'b1; s4_addr = kv_k_flat;
            s5_ceb  = 1'b0; s5_web = 1'b1; s5_addr = kv_v_flat;
        end

        // ---- S_ATTN: read s3 (q), s6 (zr at dk==0); write s5 (ao) ----
        S_ATTN: begin
            s3_ceb  = 1'b0;
            s3_web  = 1'b1;
            s3_addr = at_q_flat;
            if ((at_phase == 1'b0) && (at_dk == 3'd0)) begin
                s6_ceb  = 1'b0;
                s6_web  = 1'b1;
                s6_addr = at_zr_idx;
            end
            if (at_phase == 1'b1 && at_dk == HEAD_DIM[2:0] - 3'd1) begin
                s5_ceb  = 1'b0;
                s5_web  = 1'b0;
                s5_addr = at_ao_flat;
                s5_din  = at_ao_val;
            end
        end

        // ---- S_PROJ: read s5 (ao) during PJ_START / PJ_ADDR ----
        S_PROJ: begin
            if (pj_sub == PJ_START || pj_sub == PJ_ADDR) begin
                s5_ceb  = 1'b0;
                s5_web  = 1'b1;
                s5_addr = pj_ao_addr;
            end
        end

        default: ;
    endcase
end

assign busy = (state != S_IDLE);

endmodule
