// =============================================================================
// care_attention.v
//
// CARE Multi-Head Attention，對齊 run_backbone_numpy.attention_forward()。
//
// 演算法（NUM_HEADS=12, HEAD_DIM=64, N_TOKENS=320）：
//   1. QKV = linear(x, qkv_w, qkv_b)         [N, 3×H×D] = [320, 2304]
//   2. q,k,v = split & reshape → [H,N,D] = [12,320,64]; q,k ×= S; relu6; fp
//   3. k_mean = sum(k, axis=N) × rcp_n        [H, 1, D] = [12, 1, 64]
//   4. qk_mean = q @ k_mean.T                 [H, N, 1] = [12, 320, 1]
//   5. z_recip = recip_nr(max(qk_mean, 1/256)) [H, N, 1]
//   6. kv = k.T @ v × rcp_n                   [H, D, D] = [12, 64, 64]
//   7. attn = (q @ kv) × z_recip              [H, N, D] → reshape [N, C=768]
//   8. out = linear(attn, proj_w, proj_b) + fp (called by transformer_block)
//
// 此模組包含 FSM controller + large intermediate buffers（適合 testbench 模擬）。
// 實際 ASIC/FPGA 部署需將 intermediate buffers 對應至外部 SRAM。
//
// 常數：
//   S = 0.354 in Q8.8 ≈ 16'h005A (round(0.354×256)=91)
//   2.0/relu6 upper = 6.0 in Q8.8 = 16'h0600 = 1536
//   rcp_n = round(1/N × 2^16)/2^16 = 2/65536 ≈ 0.0000305 (round(65536/320)=205)
//
// 外部介面：
//   - x 由外部 controller 串流輸入（N×EMBED_DIM 個 Q8.8 值）
//   - QKV / proj weight & bias 由外部 ROM 提供（以 wgt_addr_o 定址）
//   - 輸出 attn_out 串流（N×EMBED_DIM 個 Q8.8 值）
// =============================================================================

module care_attention #(
    parameter EMBED_DIM  = 768,
    parameter NUM_HEADS  = 12,
    parameter HEAD_DIM   = 64,
    parameter N_TOKENS   = 320,
    // S = round(sqrt(1/HEAD_DIM) × 256) = round(0.354 × 256) = 91
    parameter S_Q88      = 91,
    // ReLU6 upper: 6.0 in Q8.8 = 1536
    parameter RELU6_MAX  = 1536,
    // rcp_n: round(2^16 / N_TOKENS) = round(65536/320) = 205
    parameter RCP_N_NUM  = 205,
    parameter RCP_N_SHIFT= 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // x input stream: N_TOKENS × EMBED_DIM values
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // Weight/bias from external ROM
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [13:0] wgt_addr_o,  // 0..EMBED_DIM×3×EMBED_DIM-1 (for QKV then proj)

    // Status
    output wire        busy,
    output reg         done,

    // attn_out stream: N_TOKENS × EMBED_DIM values (after proj, before residual add)
    output reg  signed [15:0] y_o,
    output reg         y_valid
);

// ------------------------------------------------------------------
// Intermediate buffers (testbench-friendly; map to SRAM in synthesis)
// ------------------------------------------------------------------
// Q: [NUM_HEADS, N_TOKENS, HEAD_DIM] = [12, 320, 64] = 245760 × 16b
// K: same
// V: same
// KV: [NUM_HEADS, HEAD_DIM, HEAD_DIM] = [12, 64, 64] = 49152 × 16b
// K_MEAN: [NUM_HEADS, HEAD_DIM] = [12, 64] = 768 × 16b
// QK_MEAN: [NUM_HEADS, N_TOKENS] = [12, 320] = 3840 × 16b
// Z_RECIP: same as QK_MEAN
// ATTN_OUT_FLAT: [N_TOKENS, EMBED_DIM] = 245760 × 16b (before proj)

reg signed [15:0] q_buf  [0:NUM_HEADS*N_TOKENS*HEAD_DIM-1]; // [H][N][D]
reg signed [15:0] k_buf  [0:NUM_HEADS*N_TOKENS*HEAD_DIM-1];
reg signed [15:0] v_buf  [0:NUM_HEADS*N_TOKENS*HEAD_DIM-1];
reg signed [15:0] kv_buf [0:NUM_HEADS*HEAD_DIM*HEAD_DIM-1]; // [H][D_k][D_v]
reg signed [15:0] km_buf [0:NUM_HEADS*HEAD_DIM-1];           // k_mean [H][D]
reg signed [15:0] qkm_buf[0:NUM_HEADS*N_TOKENS-1];           // qk_mean [H][N]
reg signed [15:0] zr_buf [0:NUM_HEADS*N_TOKENS-1];           // z_recip [H][N]
reg signed [15:0] ao_buf [0:N_TOKENS*EMBED_DIM-1];           // attn_out flat

// ------------------------------------------------------------------
// FSM state encoding
// ------------------------------------------------------------------
parameter S_IDLE      = 4'd0;
parameter S_LOAD_X    = 4'd1;  // load x, compute QKV via linear (serial)
parameter S_SPLIT     = 4'd2;  // scale by S, relu6, write q_buf/k_buf/v_buf
parameter S_K_MEAN    = 4'd3;  // k_mean = sum(k) × rcp_n per head×dim
parameter S_QK_MEAN   = 4'd4;  // qk_mean[h][n] = dot(q[h][n], k_mean[h])
parameter S_Z_RECIP   = 4'd5;  // z_recip = recip_nr(max(qk_mean, 1))
parameter S_KV        = 4'd6;  // kv[h][d1][d2] = dot(k[h][:,d1], v[h][:,d2])/N
parameter S_ATTN      = 4'd7;  // attn[h][n][d] = dot(q[h][n], kv[h][d]) × z_recip
parameter S_PROJ      = 4'd8;  // proj linear → y_o stream
parameter S_DONE      = 4'd9;

reg [3:0] state, next_state;

// Counters & indices
reg [13:0] cnt;      // general purpose counter
reg  [3:0] h_idx;   // head index [0..NUM_HEADS-1]
reg  [8:0] n_idx;   // token index [0..N_TOKENS-1]
reg  [5:0] d_idx;   // dim index [0..HEAD_DIM-1]
reg  [5:0] d2_idx;  // second dim index for kv

// QKV accumulator (shared with other linear ops)
reg signed [47:0] mac_acc;

// k_mean accumulator
reg signed [31:0] km_acc;

// QK_MEAN dot product accumulator
reg signed [47:0] qkm_acc;

// KV accumulator
reg signed [47:0] kv_acc;

// ATTN dot product accumulator
reg signed [47:0] attn_acc;

// recip_nr control
reg  recip_start;
reg  [15:0] recip_x;
wire recip_busy, recip_done;
wire signed [15:0] recip_y;

recip_nr u_recip (
    .clk   (clk),
    .reset (reset),
    .start (recip_start),
    .x_i   (recip_x),
    .busy  (recip_busy),
    .done  (recip_done),
    .y_o   (recip_y)
);

// ------------------------------------------------------------------
// Index helper functions (via wire expressions)
// ------------------------------------------------------------------
// q_buf index: h*N*D + n*D + d
// kv_buf index: h*D*D + d1*D + d2

// Helper wires for current element addresses
wire [17:0] q_rd_addr  = h_idx*N_TOKENS*HEAD_DIM + n_idx*HEAD_DIM + d_idx;
wire [17:0] k_rd_addr  = h_idx*N_TOKENS*HEAD_DIM + n_idx*HEAD_DIM + d_idx;
wire [17:0] v_rd_addr  = h_idx*N_TOKENS*HEAD_DIM + n_idx*HEAD_DIM + d2_idx;
wire [15:0] km_rd_addr = h_idx*HEAD_DIM + d_idx;
wire [15:0] kv_wr_addr = h_idx*HEAD_DIM*HEAD_DIM + d_idx*HEAD_DIM + d2_idx;
wire [15:0] kv_rd_addr = h_idx*HEAD_DIM*HEAD_DIM + d_idx*HEAD_DIM + d2_idx;

// QKV flat address for qkv_buf (loaded serially from linear)
// Row-major: [n][h][d] for each of q/k/v
// In practice, QKV projection output is [N, 3*H*D]; split at positions H*D, 2*H*D
// q at col [0..H*D-1], k at [H*D..2*H*D-1], v at [2*H*D..3*H*D-1]

// Current QKV linear output being computed (cnt tracks position)
wire [17:0] q_wr_idx = (cnt / HEAD_DIM) * HEAD_DIM * NUM_HEADS * N_TOKENS;  // simplified

// ReLU6 Q8.8: clamp [0, RELU6_MAX]
function signed [15:0] relu6_q88;
    input signed [15:0] x;
    begin
        if (x[15])                            relu6_q88 = 16'sd0;
        else if ($signed(x) > RELU6_MAX)      relu6_q88 = RELU6_MAX[15:0];
        else                                   relu6_q88 = x;
    end
endfunction

// fp() — round then saturate to Q8.8 (already in Q8.8 representation, just saturate)
// For add operations: saturate 32-bit result to 16-bit Q8.8
function signed [15:0] sat_q88;
    input signed [31:0] x;
    begin
        if (x > 32'sh7FFF)      sat_q88 = 16'sh7FFF;
        else if (x < -32'sh8000) sat_q88 = -16'sh8000;
        else                      sat_q88 = x[15:0];
    end
endfunction

// Scale by S: (x_int × S_Q88) >> 8 in Q8.8
wire signed [31:0] qkv_scaled_prod = $signed(wgt_i) * S_Q88;
wire signed [15:0] qkv_scaled      = qkv_scaled_prod[23:8];

// ------------------------------------------------------------------
// FSM segment 1: state register
// ------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// ------------------------------------------------------------------
// FSM segment 2: next-state logic
// ------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:    next_state = start      ? S_LOAD_X : S_IDLE;
        S_LOAD_X:  next_state = (cnt == N_TOKENS*EMBED_DIM*3 - 1) ? S_SPLIT   : S_LOAD_X;
        S_SPLIT:   next_state = (cnt == NUM_HEADS*N_TOKENS*HEAD_DIM - 1) ? S_K_MEAN : S_SPLIT;
        S_K_MEAN:  next_state = (cnt == NUM_HEADS*HEAD_DIM - 1)          ? S_QK_MEAN : S_K_MEAN;
        S_QK_MEAN: next_state = (cnt == NUM_HEADS*N_TOKENS - 1)          ? S_Z_RECIP : S_QK_MEAN;
        S_Z_RECIP: next_state = (cnt == NUM_HEADS*N_TOKENS - 1)          ? S_KV      : S_Z_RECIP;
        S_KV:      next_state = (cnt == NUM_HEADS*HEAD_DIM*HEAD_DIM - 1) ? S_ATTN    : S_KV;
        S_ATTN:    next_state = (cnt == NUM_HEADS*N_TOKENS*HEAD_DIM - 1) ? S_PROJ    : S_ATTN;
        S_PROJ:    next_state = (cnt == N_TOKENS*EMBED_DIM - 1)          ? S_DONE    : S_PROJ;
        S_DONE:    next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// ------------------------------------------------------------------
// FSM segment 3: output logic and datapath
// NOTE: Full implementation of each phase requires careful address
// arithmetic. Below shows the key datapath operations; addr/cnt
// management is abbreviated with comments for each phase.
// ------------------------------------------------------------------
always @(posedge clk) begin
    done         <= 1'b0;
    y_valid      <= 1'b0;
    recip_start  <= 1'b0;

    if (reset) begin
        cnt   <= 14'd0;
        h_idx <= 4'd0;
        n_idx <= 9'd0;
        d_idx <= 6'd0;
        d2_idx <= 6'd0;
        mac_acc <= 48'sd0;
        km_acc  <= 32'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                cnt <= 14'd0;
            end

            // Phase 1: QKV projection
            // External controller provides (x_i, wgt_i) for QKV weight matrix.
            // Here we just track the counter; actual MAC is handled by linear_q88
            // (external instantiation drives this module).
            // For this RTL skeleton, we assume QKV values arrive pre-computed
            // as a stream into q_buf/k_buf/v_buf.
            // [PLACEHOLDER: in full implementation, instantiate 3 linear_q88 units
            //  or multiplex one linear_q88 for Q, K, V rows sequentially.]
            S_LOAD_X: begin
                // Accept QKV output values and distribute to q/k/v buffers
                // cnt maps to [token_n][qkv_head_dim]; q: cols [0..H*D-1], etc.
                if (x_valid) begin
                    // Simplified: all values go into q_buf (full impl needs split)
                    // q_buf[cnt] <= x_i;  (full impl: route to q/k/v based on col)
                    cnt <= cnt + 1'b1;
                end
            end

            // Phase 2: Scale by S, apply ReLU6, write q_buf/k_buf
            S_SPLIT: begin
                // Scale q,k: q_buf[i] = relu6(q_buf[i] * S_Q88 >> 8)
                // v unchanged
                // [Implementation: iterate over all H*N*D elements]
                cnt <= cnt + 1'b1;
            end

            // Phase 3: K_MEAN = sum(k) over N, then × rcp_n
            // For each (h, d): km_buf[h*D+d] = (Σ_{n} k[h][n][d]) × 205 >> 16
            S_K_MEAN: begin
                // [Implementation: nested h/d loop, inner n loop for sum]
                cnt <= cnt + 1'b1;
            end

            // Phase 4: QK_MEAN[h][n] = Σ_d q[h][n][d] × km[h][d]
            S_QK_MEAN: begin
                cnt <= cnt + 1'b1;
            end

            // Phase 5: z_recip = recip_nr(max(qk_mean, 1))
            S_Z_RECIP: begin
                // Launch recip_nr for each (h,n); wait for done before next
                if (!recip_busy) begin
                    recip_start <= 1'b1;
                    recip_x     <= (qkm_buf[cnt] < 16'd1) ? 16'd1 : qkm_buf[cnt];
                end
                if (recip_done) begin
                    zr_buf[cnt] <= recip_y;
                    cnt <= cnt + 1'b1;
                end
            end

            // Phase 6: kv[h][d1][d2] = Σ_n k[h][n][d1] × v[h][n][d2] × rcp_n
            S_KV: begin
                cnt <= cnt + 1'b1;
            end

            // Phase 7: attn_out[h][n][d] = (Σ_d2 q[h][n][d2] × kv[h][d2][d]) × z_recip[h][n]
            S_ATTN: begin
                cnt <= cnt + 1'b1;
            end

            // Phase 8: proj linear + output stream
            S_PROJ: begin
                // [Implementation: linear(attn_flat, proj_w, proj_b), stream y_o]
                y_valid <= 1'b1;
                y_o     <= ao_buf[cnt];  // placeholder output
                cnt     <= cnt + 1'b1;
            end

            S_DONE: begin
                done <= 1'b1;
                cnt  <= 14'd0;
            end

            default: ;
        endcase
    end
end

assign busy       = (state != S_IDLE);
assign wgt_addr_o = cnt[13:0];

endmodule
