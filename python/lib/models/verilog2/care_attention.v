// =============================================================================
// care_attention.v
//
// CARE multi-head attention (Softmax-free, O(N), Q8.8 fixed-point).
// Bit-accurate mirror of attention_forward() in
// python/tracking/run_backbone_numpy_shared_trunk.py (Q8.8 CARE path).
//
// Pipeline (per block):
//   norm1 stream -> [LOAD_X] -> [QKV linear] -> [SPLIT: scale + ReLU6 on q,k]
//      -> [K_MEAN] -> [QK_MEAN] -> [Z_RECIP via NR]
//      -> [KV outer mean] -> [ATTN: q@kv*zr]
//      -> [PROJ linear] -> y_o stream
//
// Golden activation files (Q8.8, one 16-bit binary per line, C-order flatten):
//   backbone_blocks_<b>_attn_after_qkv_q_bi.txt   (H,N,d) = 4*320*8 = 10240
//   backbone_blocks_<b>_attn_after_qkv_k_bi.txt   same shape
//   backbone_blocks_<b>_attn_after_qkv_v_bi.txt   same shape
//   backbone_blocks_<b>_after_attn_attn_out_bi.txt (N,C) = 320*32 = 10240
//
// ROM access: care_attention drives 13-bit wgt_addr_o (local addr).
//   S_QKV  : addr = 0..3071        (QKV weight, bias decoded from local[12:5])
//   S_PROJ : addr = 3072..4095      (PROJ weight, bias decoded from local[9:5])
// transformer_block prepends wtype=3'b010 to form the 16-bit wgt_addr_o feeding
// backbone_top's ROM decoding (matches existing bb_wgt_mux / bb_bias_mux logic).
//
// Buffer sizes (simulation reg arrays; APR must replace with SRAM macros):
//   x_in_buf,q,k,v,ao  : 10240 x 16-bit each
//   km                  :    32 x 16
//   qkm, zr             :  1280 x 16
//   kv                  :   256 x 16
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

    // Streaming input from norm1 (N_TOKENS * EMBED_DIM = 10240 beats, Q8.8)
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // ROM weight / bias (1-cycle latency from wgt_addr_o)
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [12:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    // Streaming output to next stage (after PROJ linear)
    output reg  signed [15:0] y_o,
    output reg         y_valid
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
parameter S_LOAD_X  = 4'd1;
parameter S_QKV     = 4'd2;
parameter S_SPLIT   = 4'd3;
parameter S_K_MEAN  = 4'd4;
parameter S_QK_MEAN = 4'd5;
parameter S_Z_RECIP = 4'd6;
parameter S_KV      = 4'd7;
parameter S_ATTN    = 4'd8;
parameter S_PROJ    = 4'd9;
parameter S_DONE_ST = 4'd10;

reg [3:0] state, next_state;
reg [3:0] prev_state;

// ---------------------------------------------------------------------------
// Storage (sim-only reg arrays; replace with SRAM for synth/APR)
// ---------------------------------------------------------------------------
reg signed [15:0] x_in_buf [0:X_ELEMS-1];
reg signed [15:0] q_buf    [0:HD_ELEMS-1];
reg signed [15:0] k_buf    [0:HD_ELEMS-1];
reg signed [15:0] v_buf    [0:HD_ELEMS-1];
reg signed [15:0] km_buf   [0:KM_ELEMS-1];
reg signed [15:0] qkm_buf  [0:QKM_ELEMS-1];
reg signed [15:0] zr_buf   [0:QKM_ELEMS-1];
reg signed [15:0] kv_buf   [0:KV_ELEMS-1];
reg signed [15:0] ao_buf   [0:X_ELEMS-1];

`ifndef SYNTHESIS
integer ca_ii;
initial begin
    for (ca_ii = 0; ca_ii < X_ELEMS;  ca_ii = ca_ii + 1) begin
        x_in_buf[ca_ii] = 16'sd0;
        ao_buf  [ca_ii] = 16'sd0;
    end
    for (ca_ii = 0; ca_ii < HD_ELEMS; ca_ii = ca_ii + 1) begin
        q_buf[ca_ii] = 16'sd0;
        k_buf[ca_ii] = 16'sd0;
        v_buf[ca_ii] = 16'sd0;
    end
    for (ca_ii = 0; ca_ii < KM_ELEMS;  ca_ii = ca_ii + 1) km_buf [ca_ii] = 16'sd0;
    for (ca_ii = 0; ca_ii < QKM_ELEMS; ca_ii = ca_ii + 1) begin
        qkm_buf[ca_ii] = 16'sd0;
        zr_buf [ca_ii] = 16'sd0;
    end
    for (ca_ii = 0; ca_ii < KV_ELEMS;  ca_ii = ca_ii + 1) kv_buf[ca_ii] = 16'sd0;
end
`endif

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
// Linear instance (QKV: IN=32, OUT=96)
// ---------------------------------------------------------------------------
reg                lin_qkv_start;
wire               lin_qkv_busy, lin_qkv_done;
reg  signed [15:0] lin_qkv_x;
reg                lin_qkv_xv;
wire signed [15:0] lin_qkv_y;
wire               lin_qkv_yv;
wire [6:0]         lin_qkv_neu;
wire [12:0]        lin_qkv_addr;

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

// ---------------------------------------------------------------------------
// Linear instance (PROJ: IN=32, OUT=32)
// ---------------------------------------------------------------------------
reg                lin_proj_start;
wire               lin_proj_busy, lin_proj_done;
reg  signed [15:0] lin_proj_x;
reg                lin_proj_xv;
wire signed [15:0] lin_proj_y;
wire               lin_proj_yv;
wire [6:0]         lin_proj_neu;
wire [12:0]        lin_proj_addr;

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
reg                recip_start;
reg  signed [15:0] recip_x;
wire               recip_busy, recip_done;
wire signed [15:0] recip_y;

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
// Counters / sub-state regs (all declarations here; no reg after assign/always)
// ---------------------------------------------------------------------------
reg [13:0] load_ptr;
reg [8:0]  qx_tok;
reg [5:0]  qkv_stream_cnt;
reg [1:0]  qkv_grp;
reg [4:0]  neu_in_grp;
reg [1:0]  cap_h;
reg [2:0]  cap_d;
reg [13:0] cap_flat;

reg [13:0] sp_ptr;              // S_SPLIT pointer 0..HD_ELEMS-1

reg [4:0]  km_oidx;             // S_K_MEAN outer (h*HEAD_DIM + d) 0..KM_ELEMS-1
reg [8:0]  km_n;                // S_K_MEAN inner sum 0..N_TOKENS-1
reg signed [32:0] km_acc;       // 33-bit signed (320 x 16-bit fits in 25)

reg [10:0] qk_oidx;             // S_QK_MEAN outer (h*N + n) 0..QKM_ELEMS-1
reg [2:0]  qk_d;                // S_QK_MEAN inner d 0..HEAD_DIM-1
reg signed [32:0] qk_acc;       // 33-bit signed accumulator

reg [10:0] zr_idx;              // S_Z_RECIP index 0..QKM_ELEMS-1

reg [7:0]  kv_oidx;             // S_KV outer (h*HEAD_DIM*HEAD_DIM + d1*d + d2) 0..KV_ELEMS-1
reg [8:0]  kv_n;                // S_KV inner n 0..N_TOKENS-1
reg signed [47:0] kv_acc;       // 48-bit signed accumulator

reg [13:0] at_oidx;             // S_ATTN outer (h*N*d + n*d + d_out) 0..HD_ELEMS-1
reg [2:0]  at_dk;               // S_ATTN inner d_k 0..HEAD_DIM-1
reg signed [48:0] at_acc;       // 49-bit signed accumulator

reg [8:0]  px_tok;              // S_PROJ: token 0..N_TOKENS-1
reg [5:0]  proj_stream_cnt;     // S_PROJ streaming phase (0..32)

// ---------------------------------------------------------------------------
// Index decoders (combinational)
// ---------------------------------------------------------------------------
// S_K_MEAN outer (h,d): h = km_oidx[4:3], d = km_oidx[2:0]
wire [1:0] km_h = km_oidx[4:3];
wire [2:0] km_d = km_oidx[2:0];
// flat index into k_buf for (km_h, km_n, km_d): h*N*d + n*d + d
wire [13:0] km_k_flat =
    {12'd0, km_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, km_n}) * HEAD_DIM
  + {11'd0, km_d};
wire signed [32:0] km_acc_next =
    (km_n == 9'd0) ? $signed({{17{k_buf[km_k_flat][15]}}, k_buf[km_k_flat]})
                   : km_acc + $signed({{17{k_buf[km_k_flat][15]}}, k_buf[km_k_flat]});
wire signed [47:0] km_scaled =
    $signed({{15{km_acc_next[32]}}, km_acc_next}) * $signed({32'd0, RCP_N_NUM[15:0]});
wire signed [47:0] km_shr_w  = (km_scaled + 48'sd32768) >>> RCP_N_SHIFT;

// S_QK_MEAN outer (h,n): N_TOKENS=320 is not a power of two, so we maintain
// qk_h_reg and qk_n_reg as explicit counters that step together with qk_oidx.
reg [1:0] qk_h_reg;
reg [8:0] qk_n_reg;
wire [13:0] qk_q_flat =
    {12'd0, qk_h_reg} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, qk_n_reg}) * HEAD_DIM
  + {11'd0, qk_d};
wire [4:0] qk_km_flat = {qk_h_reg, qk_d};   // h*HEAD_DIM + d (HEAD_DIM=8)
wire signed [31:0] qk_term =
    $signed(q_buf[qk_q_flat]) * $signed(km_buf[qk_km_flat]);
wire signed [32:0] qk_acc_next =
    (qk_d == 3'd0) ? $signed({qk_term[31], qk_term})
                   : qk_acc + $signed({qk_term[31], qk_term});
wire signed [32:0] qk_rounded = qk_acc_next + 33'sd128;
wire signed [15:0] qk_sat = sat16_q88_33(qk_rounded >>> 8);

// S_KV outer (h,d1,d2): kv_h = kv_oidx[7:6], d1=kv_oidx[5:3], d2=kv_oidx[2:0]
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
wire signed [31:0] kv_term =
    $signed(k_buf[kv_k_flat]) * $signed(v_buf[kv_v_flat]);
wire signed [48:0] kv_acc_next =
    (kv_n == 9'd0) ? $signed({{17{kv_term[31]}}, kv_term})
                   : kv_acc + $signed({{17{kv_term[31]}}, kv_term});
wire signed [63:0] kv_scaled =
    $signed({{15{kv_acc_next[48]}}, kv_acc_next}) * $signed({48'd0, RCP_N_NUM[15:0]});
// k*v is Q16.16 per term; mean/N then >>>8 -> Q8.8 (total >>>24 with RCP_N_SHIFT=16)
wire signed [63:0] kv_shr_w = (kv_scaled + 64'sd8388608) >>> (RCP_N_SHIFT + 8);

// S_ATTN outer: at_oidx = h*N*d + n*d + d_out (flat over q_buf layout)
// Decompose using explicit regs (since N=320 not power-of-2):
reg [1:0] at_h_reg;
reg [8:0] at_n_reg;
reg [2:0] at_dout_reg;
wire [13:0] at_q_flat =
    {12'd0, at_h_reg} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, at_n_reg}) * HEAD_DIM
  + {11'd0, at_dk};
// kv_buf layout: flat = h*64 + d_k*8 + d_out (see S_KV kv_d1/d2); q@kv uses kv[d_k,d_out]
wire [7:0] at_kv_flat =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM)
  + {5'd0, at_dk} * HEAD_DIM
  + {5'd0, at_dout_reg};
wire signed [31:0] at_term =
    $signed(q_buf[at_q_flat]) * $signed(kv_buf[at_kv_flat]);
wire signed [48:0] at_acc_next =
    (at_dk == 3'd0) ? $signed({{17{at_term[31]}}, at_term})
                    : at_acc + $signed({{17{at_term[31]}}, at_term});
// fp #1: round dot sum once
wire signed [15:0] at_dot_sat = sat16_q88_49((at_acc_next + 49'sd128) >>> 8);
// fp #2: rnd_shr8(dot_sat * zr_buf[h*N + n])
wire [10:0] at_zr_idx = ({2'd0, at_h_reg} * N_TOKENS) + {2'd0, at_n_reg};
wire signed [31:0] at_zprod =
    $signed(at_dot_sat) * $signed(zr_buf[at_zr_idx]);
wire signed [15:0] at_ao_val = rnd_shr8_q88(at_zprod);
// destination flat in ao_buf: ao_buf[n*EMBED_DIM + h*HEAD_DIM + d_out]
wire [13:0] at_ao_flat =
    ({5'd0, at_n_reg}) * EMBED_DIM
  + {12'd0, at_h_reg} * HEAD_DIM
  + {11'd0, at_dout_reg};

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// Sample previous state for "first cycle of new state" detection
always @(posedge clk) begin
    if (reset) prev_state <= S_IDLE;
    else       prev_state <= state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:    next_state = start ? S_LOAD_X : S_IDLE;
        S_LOAD_X:  next_state = (load_ptr == X_ELEMS[13:0] - 14'd1 && x_valid) ? S_QKV : S_LOAD_X;
        S_QKV:     next_state = (lin_qkv_done && qx_tok == N_TOKENS[8:0] - 9'd1) ? S_SPLIT : S_QKV;
        S_SPLIT:   next_state = (sp_ptr == HD_ELEMS[13:0] - 14'd1) ? S_K_MEAN : S_SPLIT;
        S_K_MEAN:  next_state = (km_oidx == KM_ELEMS[4:0] - 5'd1 && km_n == N_TOKENS[8:0] - 9'd1) ? S_QK_MEAN : S_K_MEAN;
        S_QK_MEAN: next_state = (qk_oidx == QKM_ELEMS[10:0] - 11'd1 && qk_d == HEAD_DIM[2:0] - 3'd1) ? S_Z_RECIP : S_QK_MEAN;
        S_Z_RECIP: next_state = (recip_done && zr_idx == QKM_ELEMS[10:0] - 11'd1) ? S_KV : S_Z_RECIP;
        S_KV:      next_state = (kv_oidx == KV_ELEMS[7:0] - 8'd1 && kv_n == N_TOKENS[8:0] - 9'd1) ? S_ATTN : S_KV;
        S_ATTN:    next_state = (at_oidx == HD_ELEMS[13:0] - 14'd1 && at_dk == HEAD_DIM[2:0] - 3'd1) ? S_PROJ : S_ATTN;
        S_PROJ:    next_state = (lin_proj_done && px_tok == N_TOKENS[8:0] - 9'd1) ? S_DONE_ST : S_PROJ;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: main datapath
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

    if (reset) begin
        load_ptr        <= 14'd0;
        qx_tok          <= 9'd0;
        qkv_stream_cnt  <= 6'd0;
        sp_ptr          <= 14'd0;
        km_oidx         <= 5'd0;
        km_n            <= 9'd0;
        km_acc          <= 33'sd0;
        qk_oidx         <= 11'd0;
        qk_h_reg        <= 2'd0;
        qk_n_reg        <= 9'd0;
        qk_d            <= 3'd0;
        qk_acc          <= 33'sd0;
        zr_idx          <= 11'd0;
        kv_oidx         <= 8'd0;
        kv_n            <= 9'd0;
        kv_acc          <= 48'sd0;
        at_oidx         <= 14'd0;
        at_h_reg        <= 2'd0;
        at_n_reg        <= 9'd0;
        at_dout_reg     <= 3'd0;
        at_dk           <= 3'd0;
        at_acc          <= 49'sd0;
        px_tok          <= 9'd0;
        proj_stream_cnt <= 6'd0;
        y_o             <= 16'sd0;
    end else begin
        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                load_ptr        <= 14'd0;
                qx_tok          <= 9'd0;
                qkv_stream_cnt  <= 6'd0;
                sp_ptr          <= 14'd0;
                km_oidx         <= 5'd0;
                km_n            <= 9'd0;
                qk_oidx         <= 11'd0;
                qk_h_reg        <= 2'd0;
                qk_n_reg        <= 9'd0;
                qk_d            <= 3'd0;
                zr_idx          <= 11'd0;
                kv_oidx         <= 8'd0;
                kv_n            <= 9'd0;
                at_oidx         <= 14'd0;
                at_h_reg        <= 2'd0;
                at_n_reg        <= 9'd0;
                at_dout_reg     <= 3'd0;
                at_dk           <= 3'd0;
                px_tok          <= 9'd0;
                proj_stream_cnt <= 6'd0;
            end

            // -----------------------------------------------------------
            // S_LOAD_X: capture norm1 stream into x_in_buf
            // -----------------------------------------------------------
            S_LOAD_X: begin
                if (x_valid && load_ptr < X_ELEMS[13:0]) begin
                    x_in_buf[load_ptr] <= x_i;
                    load_ptr <= load_ptr + 14'd1;
                end
                if (next_state == S_QKV) begin
                    qx_tok          <= 9'd0;
                    qkv_stream_cnt  <= 6'd0;
                end
            end

            // -----------------------------------------------------------
            // S_QKV: for each token, run u_lin_qkv (IN=32, OUT=96).
            //  Phase 0       : pulse lin_start (1 cycle).
            //  Phase 1..32   : stream x_in_buf[tok*32 + (phase-1)] with xv=1.
            //  Phase >=33    : wait for lin_done; advance token.
            // y_valid pulses from linear are captured into q/k/v_buf based on
            // y_neu_o (0..95): neu[6:5]=00 -> q, 01 -> k, 10 -> v.
            // -----------------------------------------------------------
            S_QKV: begin
                if (qkv_stream_cnt == 6'd0) begin
                    lin_qkv_start  <= 1'b1;
                    qkv_stream_cnt <= 6'd1;
                end else if (qkv_stream_cnt <= EMBED_DIM[5:0]) begin
                    // Stream EMBED_DIM = 32 beats. Phase 1..32 send x[0..31].
                    lin_qkv_x  <= x_in_buf[{5'd0, qx_tok} * EMBED_DIM +
                                           {9'd0, qkv_stream_cnt - 6'd1}];
                    lin_qkv_xv <= 1'b1;
                    qkv_stream_cnt <= qkv_stream_cnt + 6'd1;
                end
                // else: streaming done; wait for lin_qkv_done

                if (lin_qkv_done) begin
                    if (qx_tok == N_TOKENS[8:0] - 9'd1) begin
                        // Last token done. Reset counters for next state.
                        qx_tok         <= 9'd0;
                        qkv_stream_cnt <= 6'd0;
                        sp_ptr         <= 14'd0;
                    end else begin
                        qx_tok         <= qx_tok + 9'd1;
                        qkv_stream_cnt <= 6'd0;
                    end
                end
            end

            // -----------------------------------------------------------
            // S_SPLIT: in-place q_buf[i] <= relu6(rnd_shr8(q*S_Q88))
            //          and same for k_buf. v_buf untouched.
            // -----------------------------------------------------------
            S_SPLIT: begin
                if (sp_ptr < HD_ELEMS[13:0]) begin
                    q_buf[sp_ptr] <= relu6_q88(rnd_shr8_q88(
                        $signed(q_buf[sp_ptr]) * S_Q88));
                    k_buf[sp_ptr] <= relu6_q88(rnd_shr8_q88(
                        $signed(k_buf[sp_ptr]) * S_Q88));
                    sp_ptr <= sp_ptr + 14'd1;
                end
                if (next_state == S_K_MEAN) begin
                    km_oidx <= 5'd0;
                    km_n    <= 9'd0;
                    km_acc  <= 33'sd0;
                end
            end

            // -----------------------------------------------------------
            // S_K_MEAN: km[h,d] = sat48((sum_n k[h,n,d] * RCP + 32768) >> 16)
            // -----------------------------------------------------------
            S_K_MEAN: begin
                km_acc <= km_acc_next;
                if (km_n == N_TOKENS[8:0] - 9'd1) begin
                    // Last inner term -- write km and advance outer.
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
                if (next_state == S_QK_MEAN) begin
                    qk_oidx  <= 11'd0;
                    qk_h_reg <= 2'd0;
                    qk_n_reg <= 9'd0;
                    qk_d     <= 3'd0;
                    qk_acc   <= 33'sd0;
                end
            end

            // -----------------------------------------------------------
            // S_QK_MEAN: qkm[h,n] = sat33((sum_d q[h,n,d]*km[h,d] + 128) >> 8)
            // Also stage qkm_eps = max(qkm, 1) into qkm_buf for recip use.
            // -----------------------------------------------------------
            S_QK_MEAN: begin
                qk_acc <= qk_acc_next;
                if (qk_d == HEAD_DIM[2:0] - 3'd1) begin
                    qkm_buf[qk_oidx] <= (qk_sat < 16'sd1) ? 16'sd1 : qk_sat;
                    qk_d <= 3'd0;
                    if (qk_oidx == QKM_ELEMS[10:0] - 11'd1) begin
                        qk_oidx  <= 11'd0;
                        qk_h_reg <= 2'd0;
                        qk_n_reg <= 9'd0;
                    end else begin
                        qk_oidx <= qk_oidx + 11'd1;
                        // qk_h_reg, qk_n_reg counters for outer (h, n)
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
                if (next_state == S_Z_RECIP) begin
                    zr_idx <= 11'd0;
                end
            end

            // -----------------------------------------------------------
            // S_Z_RECIP: serialize recip_nr over qkm_buf (1280 values)
            // -----------------------------------------------------------
            S_Z_RECIP: begin
                if (recip_done) begin
                    zr_buf[zr_idx] <= recip_y;
                    if (zr_idx == QKM_ELEMS[10:0] - 11'd1) begin
                        zr_idx <= 11'd0;
                    end else begin
                        zr_idx <= zr_idx + 11'd1;
                    end
                end else if (!recip_busy) begin
                    recip_start <= 1'b1;
                    recip_x     <= qkm_buf[zr_idx];
                end
                if (next_state == S_KV) begin
                    kv_oidx <= 8'd0;
                    kv_n    <= 9'd0;
                    kv_acc  <= 48'sd0;
                end
            end

            // -----------------------------------------------------------
            // S_KV: kv[h,d1,d2] = sat48((sum_n k[h,n,d1]*v[h,n,d2]*RCP + 32768) >> 16)
            // -----------------------------------------------------------
            S_KV: begin
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
                if (next_state == S_ATTN) begin
                    at_oidx     <= 14'd0;
                    at_h_reg    <= 2'd0;
                    at_n_reg    <= 9'd0;
                    at_dout_reg <= 3'd0;
                    at_dk       <= 3'd0;
                    at_acc      <= 49'sd0;
                end
            end

            // -----------------------------------------------------------
            // S_ATTN: ao[n,h*d+d_out] = rnd_shr8(sat49((sum_{d_k} q[d_k]*kv[d_k,d_out]+128)>>8)*zr)
            // -----------------------------------------------------------
            S_ATTN: begin
                at_acc <= at_acc_next;
                if (at_dk == HEAD_DIM[2:0] - 3'd1) begin
                    ao_buf[at_ao_flat] <= at_ao_val;
                    at_dk <= 3'd0;
                    if (at_oidx == HD_ELEMS[13:0] - 14'd1) begin
                        at_oidx     <= 14'd0;
                        at_h_reg    <= 2'd0;
                        at_n_reg    <= 9'd0;
                        at_dout_reg <= 3'd0;
                    end else begin
                        at_oidx <= at_oidx + 14'd1;
                        // Advance (h, n, d_out) in order matching at_oidx C-order:
                        //   outer: h, mid: n, inner: d_out (within at_oidx counter)
                        // Layout: at_oidx = h*N*d + n*d + d_out
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
                if (next_state == S_PROJ) begin
                    px_tok          <= 9'd0;
                    proj_stream_cnt <= 6'd0;
                end
            end

            // -----------------------------------------------------------
            // S_PROJ: drive u_lin_proj per token; forward y to module output.
            // -----------------------------------------------------------
            S_PROJ: begin
                if (proj_stream_cnt == 6'd0) begin
                    lin_proj_start  <= 1'b1;
                    proj_stream_cnt <= 6'd1;
                end else if (proj_stream_cnt <= EMBED_DIM[5:0]) begin
                    lin_proj_x  <= ao_buf[{5'd0, px_tok} * EMBED_DIM +
                                          {9'd0, proj_stream_cnt - 6'd1}];
                    lin_proj_xv <= 1'b1;
                    proj_stream_cnt <= proj_stream_cnt + 6'd1;
                end

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
// Capture u_lin_qkv outputs into q_buf / k_buf / v_buf (separate always so
// the index decode is local).  neu = 0..95: 0..31 -> q, 32..63 -> k, 64..95 -> v.
// Within each group of 32 neurons: h = neu[4:3] (HEAD_DIM=8), d = neu[2:0].
// Flat layout (h, n, d) row-major: idx = h * (N*d) + n*d + d.
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (state == S_QKV && lin_qkv_yv) begin
        qkv_grp    = lin_qkv_neu[6:5];      // 00=q,01=k,10=v
        neu_in_grp = lin_qkv_neu[4:0];      // 0..31 inside group
        cap_h      = neu_in_grp[4:3];
        cap_d      = neu_in_grp[2:0];
        cap_flat   = {12'd0, cap_h} * (N_TOKENS * HEAD_DIM)
                   + ({5'd0, qx_tok}) * HEAD_DIM
                   + {11'd0, cap_d};
        case (qkv_grp)
            2'b00: q_buf[cap_flat] <= lin_qkv_y;
            2'b01: k_buf[cap_flat] <= lin_qkv_y;
            2'b10: v_buf[cap_flat] <= lin_qkv_y;
            default: ;  // neu[6:5]==11 cannot happen (OUT_DIM=96)
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
