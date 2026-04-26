// =============================================================================
// transformer_block.v
//
// Transformer Block Controller，對齊 run_backbone_numpy.block_forward()。
//
// 架構：
//   x_norm1 = layer_norm(x, norm1_w, norm1_b)   [N, C]
//   attn_out = care_attention(x_norm1, ...)      [N, C]
//   x = fp(x + attn_out)                         [N, C]   residual 1
//   x_norm2 = layer_norm(x, norm2_w, norm2_b)   [N, C]
//   h = relu(linear(x_norm2, fc1_w, fc1_b))     [N, MLP_DIM] (mlp fc1)
//   mlp_out = linear(h, fc2_w, fc2_b)           [N, C]   (mlp fc2)
//   x = fp(x + mlp_out)                          [N, C]   residual 2
//
// 此模組為結構性 controller，實例化：
//   - 2 個 layer_norm（norm1, norm2）
//   - 1 個 care_attention
//   - 1 個 mlp_block（fc1）
//   - 1 個 mlp_block（fc2）
//
// Token buffer（N × C = 320 × 768 = 245760 × 16 bit ≈ 480 KB）：
//   x_buf 儲存目前 token 向量；在殘差加法後寫回。
//   實際 ASIC/FPGA 部署需對應外部 SRAM。
//
// 外部 controller（backbone_top）提供 block_idx，決定 weight ROM 選擇。
// =============================================================================

module transformer_block #(
    parameter EMBED_DIM = 768,
    parameter MLP_DIM   = 3072,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // Input token stream: N_TOKENS × EMBED_DIM values
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // Weight / bias for norm1, norm2, qkv, proj, fc1, fc2
    // All provided by external ROM; addr muxed by block controller
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [15:0] wgt_addr_o,   // weight address to external ROM

    // Block index (selects which block's weights to use)
    input  wire [3:0]  block_idx,

    // Status
    output wire        busy,
    output reg         done,

    // Output token stream: N_TOKENS × EMBED_DIM values
    output reg  signed [15:0] y_o,
    output reg         y_valid
);

// ------------------------------------------------------------------
// Token buffer: x_buf [N_TOKENS][EMBED_DIM]
// ------------------------------------------------------------------
reg signed [15:0] x_buf [0:N_TOKENS*EMBED_DIM-1];

// ------------------------------------------------------------------
// FSM state encoding
// ------------------------------------------------------------------
parameter S_IDLE     = 4'd0;
parameter S_LOAD_X   = 4'd1;   // load input tokens into x_buf
parameter S_NORM1    = 4'd2;   // run layer_norm (norm1) on all N tokens
parameter S_ATTN     = 4'd3;   // run care_attention
parameter S_RESID1   = 4'd4;   // x = fp(x + attn_out)
parameter S_NORM2    = 4'd5;   // run layer_norm (norm2) on all N tokens
parameter S_MLP_FC1  = 4'd6;   // run fc1 linear + relu
parameter S_MLP_FC2  = 4'd7;   // run fc2 linear
parameter S_RESID2   = 4'd8;   // x = fp(x + mlp_out)
parameter S_OUT      = 4'd9;   // stream x_buf out as y_o
parameter S_DONE     = 4'd10;

reg [3:0] state, next_state;
reg [8:0] tok_cnt;   // token counter [0..N_TOKENS-1]
reg [9:0] feat_cnt;  // feature counter [0..EMBED_DIM-1]
reg [17:0] buf_addr; // x_buf address = tok_cnt*EMBED_DIM + feat_cnt

// Intermediate buffer for norm/attn outputs (reuse x_buf slot after residual)
reg signed [15:0] tmp_buf [0:N_TOKENS*EMBED_DIM-1];

// Residual add with saturation
wire signed [31:0] resid_sum = $signed(x_buf[buf_addr]) + $signed(tmp_buf[buf_addr]);
wire signed [15:0] resid_sat = (resid_sum > 32'sh7FFF) ? 16'sh7FFF :
                                (resid_sum < -32'sh8000) ? -16'sh8000 :
                                resid_sum[15:0];

// Submodule control signals
reg  ln1_start, ln2_start, attn_start, fc1_start, fc2_start;
wire ln1_busy, ln1_done, ln2_busy, ln2_done;
wire attn_busy, attn_done;
wire fc1_busy, fc1_done, fc2_busy, fc2_done;

// Submodule data
wire signed [15:0] ln1_y, ln2_y;
wire signed [15:0] attn_y, fc1_y, fc2_y;
wire ln1_yv, ln2_yv, attn_yv, fc1_done_w, fc2_done_w;
wire [9:0] ln1_addr, ln2_addr;

// layer_norm instances (norm1, norm2)
layer_norm #(.FEAT_DIM(EMBED_DIM)) u_norm1 (
    .clk(clk), .reset(reset), .start(ln1_start),
    .x_i(x_i), .x_valid(x_valid),
    .w_i(wgt_i), .b_i(bias_i),
    .feat_addr_o(ln1_addr),
    .busy(ln1_busy), .done(ln1_done),
    .y_o(ln1_y), .y_valid(ln1_yv)
);

layer_norm #(.FEAT_DIM(EMBED_DIM)) u_norm2 (
    .clk(clk), .reset(reset), .start(ln2_start),
    .x_i(x_i), .x_valid(x_valid),
    .w_i(wgt_i), .b_i(bias_i),
    .feat_addr_o(ln2_addr),
    .busy(ln2_busy), .done(ln2_done),
    .y_o(ln2_y), .y_valid(ln2_yv)
);

// care_attention
care_attention #(
    .EMBED_DIM(EMBED_DIM)
) u_attn (
    .clk(clk), .reset(reset), .start(attn_start),
    .x_i(x_i), .x_valid(x_valid),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_addr_o(wgt_addr_o[13:0]),
    .busy(attn_busy), .done(attn_done),
    .y_o(attn_y), .y_valid(attn_yv)
);

// mlp_block fc1 (RELU=1) and fc2 (RELU=0)
mlp_block #(.CIN(EMBED_DIM), .RELU(1)) u_fc1 (
    .clk(clk), .reset(reset), .start(fc1_start),
    .a_valid(x_valid), .a_i(x_i), .w_i(wgt_i), .b_i(bias_i),
    .busy(fc1_busy), .done(fc1_done), .y_o(fc1_y)
);

mlp_block #(.CIN(MLP_DIM), .RELU(0)) u_fc2 (
    .clk(clk), .reset(reset), .start(fc2_start),
    .a_valid(x_valid), .a_i(x_i), .w_i(wgt_i), .b_i(bias_i),
    .busy(fc2_busy), .done(fc2_done), .y_o(fc2_y)
);

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
        S_IDLE:    next_state = start     ? S_LOAD_X  : S_IDLE;
        S_LOAD_X:  next_state = (buf_addr == N_TOKENS*EMBED_DIM-1 && x_valid)
                                           ? S_NORM1   : S_LOAD_X;
        S_NORM1:   next_state = (ln1_done && tok_cnt == N_TOKENS-1)
                                           ? S_ATTN    : S_NORM1;
        S_ATTN:    next_state = attn_done  ? S_RESID1  : S_ATTN;
        S_RESID1:  next_state = (buf_addr == N_TOKENS*EMBED_DIM-1)
                                           ? S_NORM2   : S_RESID1;
        S_NORM2:   next_state = (ln2_done && tok_cnt == N_TOKENS-1)
                                           ? S_MLP_FC1 : S_NORM2;
        S_MLP_FC1: next_state = (fc1_done && tok_cnt == N_TOKENS*MLP_DIM/EMBED_DIM-1)
                                           ? S_MLP_FC2 : S_MLP_FC1;
        S_MLP_FC2: next_state = (fc2_done && tok_cnt == N_TOKENS*EMBED_DIM/EMBED_DIM-1)
                                           ? S_RESID2  : S_MLP_FC2;
        S_RESID2:  next_state = (buf_addr == N_TOKENS*EMBED_DIM-1)
                                           ? S_OUT     : S_RESID2;
        S_OUT:     next_state = (buf_addr == N_TOKENS*EMBED_DIM-1)
                                           ? S_DONE    : S_OUT;
        S_DONE:    next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// ------------------------------------------------------------------
// FSM segment 3: output logic
// ------------------------------------------------------------------
always @(posedge clk) begin
    done    <= 1'b0;
    y_valid <= 1'b0;
    ln1_start  <= 1'b0;
    ln2_start  <= 1'b0;
    attn_start <= 1'b0;
    fc1_start  <= 1'b0;
    fc2_start  <= 1'b0;

    if (reset) begin
        tok_cnt  <= 9'd0;
        feat_cnt <= 10'd0;
        buf_addr <= 18'd0;
    end else begin
        case (state)
            S_IDLE: begin
                buf_addr <= 18'd0;
                tok_cnt  <= 9'd0;
            end

            S_LOAD_X: begin
                if (x_valid) begin
                    x_buf[buf_addr] <= x_i;
                    buf_addr <= buf_addr + 1'b1;
                end
            end

            S_NORM1: begin
                // Run layer_norm for each token; norm output → tmp_buf
                if (ln1_yv) begin
                    tmp_buf[tok_cnt*EMBED_DIM + feat_cnt] <= ln1_y;
                    feat_cnt <= feat_cnt + 1'b1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 10'd0;
                        tok_cnt  <= tok_cnt + 1'b1;
                        if (tok_cnt < N_TOKENS-1)
                            ln1_start <= 1'b1;
                    end
                end else if (!ln1_busy && tok_cnt == 9'd0)
                    ln1_start <= 1'b1;
            end

            S_ATTN: begin
                // care_attention streams in tmp_buf, streams out to tmp_buf2/attn region
                if (!attn_busy)
                    attn_start <= 1'b1;
                if (attn_yv) begin
                    tmp_buf[buf_addr] <= attn_y;
                    buf_addr <= buf_addr + 1'b1;
                end
            end

            S_RESID1: begin
                // x = fp(x + attn_out); x_buf += tmp_buf, saturated
                x_buf[buf_addr] <= resid_sat;
                buf_addr <= buf_addr + 1'b1;
            end

            S_NORM2: begin
                if (ln2_yv) begin
                    tmp_buf[tok_cnt*EMBED_DIM + feat_cnt] <= ln2_y;
                    feat_cnt <= feat_cnt + 1'b1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 10'd0;
                        tok_cnt  <= tok_cnt + 1'b1;
                        if (tok_cnt < N_TOKENS-1)
                            ln2_start <= 1'b1;
                    end
                end else if (!ln2_busy && tok_cnt == 9'd0)
                    ln2_start <= 1'b1;
            end

            S_MLP_FC1: begin
                // fc1: [N, EMBED_DIM] → [N, MLP_DIM]; ReLU applied inside mlp_block
                if (fc1_done) begin
                    tmp_buf[tok_cnt] <= fc1_y;
                    tok_cnt <= tok_cnt + 1'b1;
                end else if (!fc1_busy)
                    fc1_start <= 1'b1;
            end

            S_MLP_FC2: begin
                // fc2: [N, MLP_DIM] → [N, EMBED_DIM]
                if (fc2_done) begin
                    tmp_buf[tok_cnt] <= fc2_y;
                    tok_cnt <= tok_cnt + 1'b1;
                end else if (!fc2_busy)
                    fc2_start <= 1'b1;
            end

            S_RESID2: begin
                x_buf[buf_addr] <= resid_sat;
                buf_addr <= buf_addr + 1'b1;
            end

            S_OUT: begin
                y_o     <= x_buf[buf_addr];
                y_valid <= 1'b1;
                buf_addr <= buf_addr + 1'b1;
            end

            S_DONE: begin
                done     <= 1'b1;
                buf_addr <= 18'd0;
                tok_cnt  <= 9'd0;
                feat_cnt <= 10'd0;
            end

            default: ;
        endcase
    end
end

assign busy         = (state != S_IDLE);
assign wgt_addr_o   = {block_idx, 12'd0};  // upper bits select block weight bank

endmodule
