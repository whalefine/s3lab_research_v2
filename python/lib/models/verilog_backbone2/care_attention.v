// =============================================================================
// care_attention.v  (SRAM macros in sglatrack_top; QKV reads parent norm1 via norm_rd_*)
// -----------------------------------------------------------------------------
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
//   S_ATTN           : at_phase 4-step (timing-friendly, golden unchanged):
//                       0 ADDR read Sram_q (+ zr@dk0); 1 MAC at_acc<=acc+term;
//                       2 DOT latch at_dot_sat from at_acc; 3 AO *zr write Sram_v.
//                       dk<7: ADDR<->MAC only; dk==7: +DOT+AO. ~2.5*10240+2*10240 cyc.
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
// Sequential = `<=`, combinational = `=`. Saturation / rnd_shr8 / relu6 via wire
// only (no function). No latch inference (all outputs covered in case branches).
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
    clk,
    reset,
    start,
    norm_rd_en,
    norm_rd_flat,
    norm_x,
    wgt_i,
    bias_i,
    wgt_addr_o,
    busy,
    done,
    y_o,
    y_valid,
    sram_q_ceb_o,
    sram_q_web_o,
    sram_q_addr_o,
    sram_q_din_o,
    sram_q_q_i,
    sram_k_ceb_o,
    sram_k_web_o,
    sram_k_addr_o,
    sram_k_din_o,
    sram_k_q_i,
    sram_v_ceb_o,
    sram_v_web_o,
    sram_v_addr_o,
    sram_v_din_o,
    sram_v_q_i,
    sram_qkm_ceb_o,
    sram_qkm_web_o,
    sram_qkm_addr_o,
    sram_qkm_din_o,
    sram_qkm_q_i
);

parameter HD_ELEMS  = NUM_HEADS * N_TOKENS * HEAD_DIM;   // 10240
parameter KM_ELEMS  = NUM_HEADS * HEAD_DIM;              // 32
parameter QKM_ELEMS = NUM_HEADS * N_TOKENS;              // 1280
parameter KV_ELEMS  = NUM_HEADS * HEAD_DIM * HEAD_DIM;   // 256
parameter X_ELEMS   = N_TOKENS * EMBED_DIM;              // 10240

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

parameter PJ_START = 2'd0;
parameter PJ_ADDR  = 2'd1;
parameter PJ_USE   = 2'd2;
parameter PJ_WAIT  = 2'd3;

parameter AT_PH_ADDR = 2'd0;
parameter AT_PH_MAC  = 2'd1;
parameter AT_PH_DOT  = 2'd2;
parameter AT_PH_AO   = 2'd3;

input                       clk;
input                       reset;
input                       start;
output reg                  norm_rd_en;
output reg [13:0]           norm_rd_flat;
input  signed [15:0]        norm_x;
input  signed [15:0]        wgt_i;
input  signed [15:0]        bias_i;
output [12:0]               wgt_addr_o;
output                      busy;
output reg                  done;
output reg signed [15:0]    y_o;
output reg                  y_valid;
output                      sram_q_ceb_o;
output                      sram_q_web_o;
output [13:0]               sram_q_addr_o;
output [15:0]               sram_q_din_o;
input  [15:0]               sram_q_q_i;
output                      sram_k_ceb_o;
output                      sram_k_web_o;
output [13:0]               sram_k_addr_o;
output [15:0]               sram_k_din_o;
input  [15:0]               sram_k_q_i;
output                      sram_v_ceb_o;
output                      sram_v_web_o;
output [13:0]               sram_v_addr_o;
output [15:0]               sram_v_din_o;
input  [15:0]               sram_v_q_i;
output                      sram_qkm_ceb_o;
output                      sram_qkm_web_o;
output [13:0]               sram_qkm_addr_o;
output [15:0]               sram_qkm_din_o;
input  [15:0]               sram_qkm_q_i;

// Small reg scratch (km_buf / kv_buf; attention SRAM macros for q/k/v/ao)
reg signed [15:0]           km_buf [0:KM_ELEMS-1];
reg signed [15:0]           kv_buf [0:KV_ELEMS-1];

`ifndef SYNTHESIS
integer ca_ii;
initial begin
    for (ca_ii = 0; ca_ii < KM_ELEMS; ca_ii = ca_ii + 1)
        km_buf[ca_ii] = 16'sd0;
    for (ca_ii = 0; ca_ii < KV_ELEMS; ca_ii = ca_ii + 1)
        kv_buf[ca_ii] = 16'sd0;
end
`endif

reg [3:0]                   state, next_state;

reg [8:0]                   qx_tok;
reg [5:0]                   qkv_stream_cnt;
reg [13:0]                  sp_ptr;

reg [4:0]                   km_oidx;
reg [8:0]                   km_n;
reg signed [32:0]           km_acc;

reg [10:0]                  qk_oidx;
reg [2:0]                   qk_d;
reg signed [32:0]           qk_acc;

reg [10:0]                  zr_idx;

reg [7:0]                   kv_oidx;
reg [8:0]                   kv_n;
reg signed [47:0]           kv_acc;

reg [13:0]                  at_oidx;
reg [2:0]                   at_dk;
reg signed [48:0]           at_acc;

reg [8:0]                   px_tok;
reg [5:0]                   proj_stream_cnt;

reg                         sp_phase;
reg signed [15:0]           sp_q_r;
reg signed [15:0]           sp_k_r;
reg                         km_phase;
reg                         qk_phase;
reg                         kv_phase;
reg [1:0]                   at_phase;
reg signed [15:0]           at_dot_sat_r;
reg [1:0]                   pj_sub;
reg                         qkv_x_phase;
reg                         zr_phase;
reg signed [15:0]           zr_qkm_r;
reg signed [15:0]           at_zr_r;

reg [1:0]                   qk_h_reg;
reg [8:0]                   qk_n_reg;

reg [1:0]                   at_h_reg;
reg [8:0]                   at_n_reg;
reg [2:0]                   at_dout_reg;

reg                         s3_ceb, s3_web;
reg [13:0]                  s3_addr;
reg [15:0]                  s3_din;
wire [15:0]                 s3_q;

reg                         s4_ceb, s4_web;
reg [13:0]                  s4_addr;
reg [15:0]                  s4_din;
wire [15:0]                 s4_q;

reg                         s5_ceb, s5_web;
reg [13:0]                  s5_addr;
reg [15:0]                  s5_din;
wire [15:0]                 s5_q;

reg                         s6_ceb, s6_web;
reg [10:0]                  s6_addr;
reg [15:0]                  s6_din;
wire [15:0]                 s6_q;

reg                         recip_start;
reg signed [15:0]           recip_x;
wire                        recip_busy, recip_done;
wire signed [15:0]          recip_y;

reg                         lin_qkv_start;
wire                        lin_qkv_busy, lin_qkv_done;
reg signed [15:0]           lin_qkv_x;
reg                         lin_qkv_xv;
wire signed [15:0]          lin_qkv_y;
wire                        lin_qkv_yv;
wire [6:0]                  lin_qkv_neu;
wire [12:0]                 lin_qkv_addr;

reg                         lin_proj_start;
wire                        lin_proj_busy, lin_proj_done;
reg signed [15:0]           lin_proj_x;
reg                         lin_proj_xv;
wire signed [15:0]          lin_proj_y;
wire                        lin_proj_yv;
wire [6:0]                  lin_proj_neu;
wire [12:0]                 lin_proj_addr;

// Index decoders (combinational; USE cycle consumes s3_q / s4_q / s5_q)
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
wire signed [15:0] km_wr_val =
    (km_shr_w > 48'sd32767) ? 16'sh7FFF :
    (km_shr_w < -48'sd32768) ? 16'sh8000 : km_shr_w[15:0];

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
wire signed [32:0] qk_shr8    = qk_rounded >>> 8;
wire signed [15:0] qk_sat     =
    (qk_shr8 > 33'sd32767) ? 16'sh7FFF :
    (qk_shr8 < -33'sd32768) ? 16'sh8000 : qk_shr8[15:0];
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
wire signed [15:0] kv_wr_val =
    (kv_shr_w > 64'sd32767) ? 16'sh7FFF :
    (kv_shr_w < -64'sd32768) ? 16'sh8000 : kv_shr_w[15:0];

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
// S_ATTN MAC (phase AT_PH_MAC): one q*kv term into at_acc
wire signed [31:0] at_term =
    $signed(at_q_data) * $signed(kv_buf[at_kv_flat]);
wire signed [48:0] at_acc_next =
    (at_dk == 3'd0) ? $signed({{17{at_term[31]}}, at_term})
                    : at_acc + $signed({{17{at_term[31]}}, at_term});
// S_ATTN POST (phases AT_PH_DOT / AT_PH_AO): sequential from latched at_acc / at_dot_sat_r
// maps _care_attn_q88: dot_sat = sat16_from49((acc+128)>>8); ao = rnd_shr8(dot_sat*zr)
wire signed [48:0] at_dot_shr_from_acc = (at_acc + 49'sd128) >>> 8;
wire signed [15:0] at_dot_sat_next =
    (at_dot_shr_from_acc > 49'sd32767) ? 16'sh7FFF :
    (at_dot_shr_from_acc < -49'sd32768) ? 16'sh8000 : at_dot_shr_from_acc[15:0];
wire signed [31:0] at_zprod_raw = $signed(at_dot_sat_r) * $signed(at_zr_r);
wire signed [31:0] at_zprod_rnd   = at_zprod_raw + 32'sd128;
wire signed [31:0] at_zprod_shr_w = at_zprod_rnd >>> 8;
wire signed [15:0] at_ao_sat_comb =
    (at_zprod_shr_w > 32'sd32767) ? 16'sh7FFF :
    (at_zprod_shr_w < -32'sd32768) ? 16'sh8000 : at_zprod_shr_w[15:0];
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

// S_SPLIT: rnd_shr8(sp_* * S_Q88) then relu6
wire signed [31:0] sp_q_prod    = $signed(sp_q_r) * S_Q88;
wire signed [31:0] sp_q_rnd_t  = sp_q_prod + 32'sd128;
wire signed [31:0] sp_q_shr     = sp_q_rnd_t >>> 8;
wire signed [15:0] sp_q_sat     =
    (sp_q_shr > 32'sd32767) ? 16'sh7FFF :
    (sp_q_shr < -32'sd32768) ? 16'sh8000 : sp_q_shr[15:0];
wire signed [15:0] sp_q_relu   =
    sp_q_sat[15] ? 16'sd0 :
    ($signed(sp_q_sat) > RELU6_MAX) ? RELU6_MAX[15:0] : sp_q_sat;

wire signed [31:0] sp_k_prod    = $signed(sp_k_r) * S_Q88;
wire signed [31:0] sp_k_rnd_t  = sp_k_prod + 32'sd128;
wire signed [31:0] sp_k_shr     = sp_k_rnd_t >>> 8;
wire signed [15:0] sp_k_sat     =
    (sp_k_shr > 32'sd32767) ? 16'sh7FFF :
    (sp_k_shr < -32'sd32768) ? 16'sh8000 : sp_k_shr[15:0];
wire signed [15:0] sp_k_relu   =
    sp_k_sat[15] ? 16'sd0 :
    ($signed(sp_k_sat) > RELU6_MAX) ? RELU6_MAX[15:0] : sp_k_sat;

assign sram_q_ceb_o    = s3_ceb;
assign sram_q_web_o    = s3_web;
assign sram_q_addr_o   = s3_addr;
assign sram_q_din_o    = s3_din;
assign s3_q            = sram_q_q_i;

assign sram_k_ceb_o    = s4_ceb;
assign sram_k_web_o    = s4_web;
assign sram_k_addr_o   = s4_addr;
assign sram_k_din_o    = s4_din;
assign s4_q            = sram_k_q_i;

assign sram_v_ceb_o    = s5_ceb;
assign sram_v_web_o    = s5_web;
assign sram_v_addr_o   = s5_addr;
assign sram_v_din_o    = s5_din;
assign s5_q            = sram_v_q_i;

assign sram_qkm_ceb_o  = s6_ceb;
assign sram_qkm_web_o  = s6_web;
assign sram_qkm_addr_o = {3'b000, s6_addr};
assign sram_qkm_din_o  = s6_din;
assign s6_q            = sram_qkm_q_i;

assign wgt_addr_o = (state == S_QKV)  ? lin_qkv_addr :
                    (state == S_PROJ) ? (13'd3072 + lin_proj_addr) :
                                        13'd0;

assign busy = (state != S_IDLE);

linear #(
    .IN_DIM (EMBED_DIM),
    .OUT_DIM(3*EMBED_DIM)
) u_lin_qkv (
    .clk      (clk),
    .reset    (reset),
    .start    (lin_qkv_start),
    .x_i      (lin_qkv_x),
    .x_valid  (lin_qkv_xv),
    .w_i      (wgt_i),
    .b_i      (bias_i),
    .w_addr_o (lin_qkv_addr),
    .busy     (lin_qkv_busy),
    .done     (lin_qkv_done),
    .y_o      (lin_qkv_y),
    .y_valid  (lin_qkv_yv),
    .y_neu_o  (lin_qkv_neu)
);

linear #(
    .IN_DIM (EMBED_DIM),
    .OUT_DIM(EMBED_DIM)
) u_lin_proj (
    .clk      (clk),
    .reset    (reset),
    .start    (lin_proj_start),
    .x_i      (lin_proj_x),
    .x_valid  (lin_proj_xv),
    .w_i      (wgt_i),
    .b_i      (bias_i),
    .w_addr_o (lin_proj_addr),
    .busy     (lin_proj_busy),
    .done     (lin_proj_done),
    .y_o      (lin_proj_y),
    .y_valid  (lin_proj_yv),
    .y_neu_o  (lin_proj_neu)
);

recip_nr u_recip (
    .clk   (clk),
    .reset (reset),
    .start (recip_start),
    .x_i   (recip_x),
    .busy  (recip_busy),
    .done  (recip_done),
    .y_o   (recip_y)
);

// FSM state
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM next_state
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:    next_state = start ? S_QKV : S_IDLE;
        S_QKV:     next_state = (lin_qkv_done && qx_tok == N_TOKENS[8:0] - 9'd1) ? S_SPLIT : S_QKV;
        S_SPLIT:   next_state = (sp_ptr == HD_ELEMS[13:0] - 14'd1 && sp_phase == 1'b1) ? S_K_MEAN : S_SPLIT;
        S_K_MEAN:  next_state = (km_oidx == KM_ELEMS[4:0] - 5'd1 && km_n == N_TOKENS[8:0] - 9'd1 && km_phase == 1'b1) ? S_QK_MEAN : S_K_MEAN;
        S_QK_MEAN: next_state = (qk_oidx == QKM_ELEMS[10:0] - 11'd1 && qk_d == HEAD_DIM[2:0] - 3'd1 && qk_phase == 1'b1) ? S_Z_RECIP : S_QK_MEAN;
        S_Z_RECIP: next_state = (recip_done && zr_idx == QKM_ELEMS[10:0] - 11'd1) ? S_KV : S_Z_RECIP;
        S_KV:      next_state = (kv_oidx == KV_ELEMS[7:0] - 8'd1 && kv_n == N_TOKENS[8:0] - 9'd1 && kv_phase == 1'b1) ? S_ATTN : S_KV;
        S_ATTN:    next_state = (at_oidx == HD_ELEMS[13:0] - 14'd1 && at_dk == HEAD_DIM[2:0] - 3'd1 && at_phase == AT_PH_AO) ? S_PROJ : S_ATTN;
        S_PROJ:    next_state = (lin_proj_done && px_tok == N_TOKENS[8:0] - 9'd1) ? S_DONE_ST : S_PROJ;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// S_SPLIT shadow: latch s3_q/s4_q at end of ADDR beat
always @(posedge clk) begin
    if (reset) begin
        sp_q_r <= 16'sd0;
        sp_k_r <= 16'sd0;
    end else if (state == S_SPLIT && sp_phase == 1'b0) begin
        sp_q_r <= s3_q;
        sp_k_r <= s4_q;
    end
end

// S_Z_RECIP: latch qkm read before launching recip_nr
always @(posedge clk) begin
    if (reset)
        zr_qkm_r <= 16'sd0;
    else if (state == S_Z_RECIP && zr_phase == 1'b0)
        zr_qkm_r <= s6_q;
end

// S_ATTN: latch zr at at_dk==0 ADDR beat
always @(posedge clk) begin
    if (reset)
        at_zr_r <= 16'sd0;
    else if (state == S_ATTN && at_phase == AT_PH_ADDR && at_dk == 3'd0)
        at_zr_r <= s6_q;
end

// S_ATTN POST: latch dot_sat after MAC completes (at_acc holds full dot)
always @(posedge clk) begin
    if (reset)
        at_dot_sat_r <= 16'sd0;
    else if (state == S_ATTN && at_phase == AT_PH_DOT)
        at_dot_sat_r <= at_dot_sat_next;
end

// done, y_valid, sub-block starts, 2-phase datapath
always @(posedge clk) begin
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
        at_phase        <= AT_PH_ADDR;
        at_dot_sat_r    <= 16'sd0;
        px_tok          <= 9'd0;
        proj_stream_cnt <= 6'd0;
        pj_sub          <= PJ_START;
        qkv_x_phase     <= 1'b0;
        zr_phase        <= 1'b0;
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
                at_phase        <= AT_PH_ADDR;
                at_dot_sat_r    <= 16'sd0;
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
                        km_buf[km_oidx] <= km_wr_val;
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
                        kv_buf[kv_oidx] <= kv_wr_val;
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
                    at_phase    <= AT_PH_ADDR;
                end
            end

            // -----------------------------------------------------------
            // S_ATTN: 4-phase (ADDR/MAC/DOT/AO). MAC only updates at_acc; ao write
            // on AT_PH_AO uses at_dot_sat_r + at_zr_r (shorter comb than old 1-beat).
            // -----------------------------------------------------------
            S_ATTN: begin
                if (at_phase == AT_PH_ADDR) begin
                    at_phase <= AT_PH_MAC;
                end else if (at_phase == AT_PH_MAC) begin
                    at_acc <= at_acc_next;
                    if (at_dk == HEAD_DIM[2:0] - 3'd1)
                        at_phase <= AT_PH_DOT;
                    else begin
                        at_dk    <= at_dk + 3'd1;
                        at_phase <= AT_PH_ADDR;
                    end
                end else if (at_phase == AT_PH_DOT) begin
                    at_phase <= AT_PH_AO;
                end else begin
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
                    at_phase <= AT_PH_ADDR;
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

// SRAM port mux (norm1 read via parent norm_rd_*)
always @(*) begin
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
                s3_din  = sp_q_relu;
                s4_ceb  = 1'b0; s4_web = 1'b0; s4_addr = sp_ptr;
                s4_din  = sp_k_relu;
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
            if ((at_phase == AT_PH_ADDR) && (at_dk == 3'd0)) begin
                s6_ceb  = 1'b0;
                s6_web  = 1'b1;
                s6_addr = at_zr_idx;
            end
            if (at_phase == AT_PH_AO) begin
                s5_ceb  = 1'b0;
                s5_web  = 1'b0;
                s5_addr = at_ao_flat;
                s5_din  = at_ao_sat_comb;
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

endmodule
