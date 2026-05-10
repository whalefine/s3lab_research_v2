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
//   h = relu(linear(x_norm2, fc1_w, fc1_b))     [N, MLP_DIM]
//   mlp_out = linear(h, fc2_w, fc2_b)           [N, C]
//   x = fp(x + mlp_out)                          [N, C]   residual 2
//
// wgt_addr_o encoding:
//   [15:13] = weight type (3-bit)
//     3'b000 = norm1   3'b001 = norm2
//     3'b010 = attn    3'b100 = fc1    3'b101 = fc2
//   [12:0]  = local address within type
//     norm: feat index [4:0]
//     attn: care_attention internal counter [12:0]
//     fc1:  {1'b0, neuron[6:0], feat[4:0]}  (neuron*32+feat)
//     fc2:  {1'b0, neuron[4:0], feat[6:0]}  (neuron*128+feat)
// =============================================================================

module transformer_block #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // Input token stream: N_TOKENS × EMBED_DIM values
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // Weight / bias from external ROM (backbone_top muxes based on wgt_addr_o type)
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [15:0] wgt_addr_o,

    // Block index (passed through to backbone_top ROM address computation)
    input  wire [3:0]  block_idx,

    // Status
    output wire        busy,
    output reg         done,

    // Output token stream: N_TOKENS × EMBED_DIM values
    output reg  signed [15:0] y_o,
    output reg         y_valid
);

// ---------------------------------------------------------------------------
// Derived parameter: reciprocal for layer_norm (round(2^16/EMBED_DIM))
// ---------------------------------------------------------------------------
parameter LN_RCP = 65536 / EMBED_DIM;   // 2048 for EMBED_DIM=32

// ---------------------------------------------------------------------------
// Token buffers
// ---------------------------------------------------------------------------
// x_buf: current token residual stream [N_TOKENS][EMBED_DIM]
reg signed [15:0] x_buf   [0:N_TOKENS*EMBED_DIM-1];
// tmp_buf: norm/attn/mlp intermediate [N_TOKENS][EMBED_DIM]
reg signed [15:0] tmp_buf [0:N_TOKENS*EMBED_DIM-1];
// h_buf: fc1 output [N_TOKENS][MLP_DIM]
reg signed [15:0] h_buf   [0:N_TOKENS*MLP_DIM-1];

// ---------------------------------------------------------------------------
// FSM state encoding
// ---------------------------------------------------------------------------
parameter S_IDLE     = 4'd0;
parameter S_LOAD_X   = 4'd1;
parameter S_NORM1    = 4'd2;
parameter S_ATTN     = 4'd3;
parameter S_RESID1   = 4'd4;
parameter S_NORM2    = 4'd5;
parameter S_MLP_FC1  = 4'd6;
parameter S_MLP_FC2  = 4'd7;
parameter S_RESID2   = 4'd8;
parameter S_OUT      = 4'd9;
parameter S_DONE     = 4'd10;

reg [3:0] state, next_state;

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------
reg [13:0] buf_addr;  // general buffer addr (LOAD_X, ATTN, RESID1/2, OUT)
reg [8:0]  tok_cnt;   // token index (0..N_TOKENS) for NORM1, NORM2
reg [4:0]  feat_cnt;  // feature index for norm output capture (0..EMBED_DIM-1)

// Replay streaming
reg [4:0]  rp_feat;   // feature index for x_buf/tmp_buf replay (EMBED_DIM cycles)
reg [7:0]  mlp_feat;  // feature index for h_buf replay (MLP_DIM cycles, S_MLP_FC2)
reg        rp_stream; // replay valid gate
reg        rp_stream_r; // 1-cycle delayed rp_stream (for mlp_block ROM latency)

// MLP iteration counters
reg [8:0]  mlp_tok;   // token index for MLP (0..N_TOKENS-1)
reg [7:0]  mlp_neu;   // output neuron index (0..MLP_DIM-1 fc1; 0..EMBED_DIM-1 fc2)

// ---------------------------------------------------------------------------
// Submodule control / data
// ---------------------------------------------------------------------------
reg  ln1_start, ln2_start, attn_start, fc1_start, fc2_start;
wire ln1_busy, ln1_done;
wire ln2_busy, ln2_done;
wire attn_busy, attn_done;
wire fc1_busy, fc1_done;
wire fc2_busy, fc2_done;

wire signed [15:0] ln1_y, ln2_y, attn_y, fc1_y, fc2_y;
wire ln1_yv, ln2_yv, attn_yv;
wire [9:0]  ln1_addr, ln2_addr;
wire [13:0] attn_wgt_addr;

// ---------------------------------------------------------------------------
// Replay data wires (combinational reads from buffers)
// ---------------------------------------------------------------------------
// x_buf replay → u_norm1 and u_norm2 (one token at a time)
wire [13:0] xbuf_rp_addr = tok_cnt * EMBED_DIM + {9'b0, rp_feat};
wire signed [15:0] xbuf_rp_data = x_buf[xbuf_rp_addr];

// tmp_buf replay → u_attn (sequential, full stream)
wire signed [15:0] tmp_rp_data  = tmp_buf[buf_addr];

// tmp_buf replay → u_fc1 (per token, EMBED_DIM values)
wire [13:0] tmp_fc1_addr = mlp_tok * EMBED_DIM + {9'b0, rp_feat};
wire signed [15:0] tmp_fc1_data = tmp_buf[tmp_fc1_addr];

// h_buf replay → u_fc2 (per token, MLP_DIM values)
wire [15:0] h_fc2_addr  = mlp_tok * MLP_DIM + {8'b0, mlp_feat};
wire signed [15:0] h_fc2_data  = h_buf[h_fc2_addr];

// ---------------------------------------------------------------------------
// Submodule x_valid / a_valid gating
// ---------------------------------------------------------------------------
wire ln1_xv  = rp_stream && (state == S_NORM1);
wire ln2_xv  = rp_stream && (state == S_NORM2);
wire attn_xv = rp_stream && (state == S_ATTN);
// 1-cycle delay for mlp_block: compensates falling-edge ROM read latency
wire fc1_av  = rp_stream_r && (state == S_MLP_FC1);
wire fc2_av  = rp_stream_r && (state == S_MLP_FC2);

// ---------------------------------------------------------------------------
// Residual saturating add (used in S_RESID1 and S_RESID2)
// ---------------------------------------------------------------------------
wire signed [16:0] resid_sum = $signed({x_buf[buf_addr][15], x_buf[buf_addr]}) +
                                $signed({tmp_buf[buf_addr][15], tmp_buf[buf_addr]});
// Overflow: bit16 != bit15
wire resid_ovf = resid_sum[16] ^ resid_sum[15];
wire signed [15:0] resid_sat = resid_ovf ? (resid_sum[16] ? 16'sh8000 : 16'sh7FFF)
                                          : resid_sum[15:0];

// ---------------------------------------------------------------------------
// Submodule instantiations
// ---------------------------------------------------------------------------
layer_norm #(
    .FEAT_DIM (EMBED_DIM),
    .RCP_NUM  (LN_RCP),
    .RCP_SHIFT(16)
) u_norm1 (
    .clk(clk), .reset(reset), .start(ln1_start),
    .x_i(xbuf_rp_data), .x_valid(ln1_xv),
    .w_i(wgt_i), .b_i(bias_i),
    .feat_addr_o(ln1_addr),
    .busy(ln1_busy), .done(ln1_done),
    .y_o(ln1_y), .y_valid(ln1_yv)
);

layer_norm #(
    .FEAT_DIM (EMBED_DIM),
    .RCP_NUM  (LN_RCP),
    .RCP_SHIFT(16)
) u_norm2 (
    .clk(clk), .reset(reset), .start(ln2_start),
    .x_i(xbuf_rp_data), .x_valid(ln2_xv),
    .w_i(wgt_i), .b_i(bias_i),
    .feat_addr_o(ln2_addr),
    .busy(ln2_busy), .done(ln2_done),
    .y_o(ln2_y), .y_valid(ln2_yv)
);

care_attention #(
    .EMBED_DIM(EMBED_DIM),
    .N_TOKENS (N_TOKENS)
) u_attn (
    .clk(clk), .reset(reset), .start(attn_start),
    .x_i(tmp_rp_data), .x_valid(attn_xv),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_addr_o(attn_wgt_addr),
    .busy(attn_busy), .done(attn_done),
    .y_o(attn_y), .y_valid(attn_yv)
);

mlp_block #(.CIN(EMBED_DIM), .RELU(1)) u_fc1 (
    .clk(clk), .reset(reset), .start(fc1_start),
    .a_valid(fc1_av), .a_i(tmp_fc1_data),
    .w_i(wgt_i), .b_i(bias_i),
    .busy(fc1_busy), .done(fc1_done), .y_o(fc1_y)
);

mlp_block #(.CIN(MLP_DIM), .RELU(0)) u_fc2 (
    .clk(clk), .reset(reset), .start(fc2_start),
    .a_valid(fc2_av), .a_i(h_fc2_data),
    .w_i(wgt_i), .b_i(bias_i),
    .busy(fc2_busy), .done(fc2_done), .y_o(fc2_y)
);

// ---------------------------------------------------------------------------
// wgt_addr_o: mux based on FSM state
// ---------------------------------------------------------------------------
assign wgt_addr_o =
    (state == S_NORM1)   ? {3'b000, 3'b0, ln1_addr[9:0]}              :
    (state == S_NORM2)   ? {3'b001, 3'b0, ln2_addr[9:0]}              :
    (state == S_ATTN)    ? {3'b010, 1'b0, attn_wgt_addr[12:0]}        :
    (state == S_MLP_FC1) ? {3'b100, 1'b0, mlp_neu[6:0], rp_feat[4:0]}:
    (state == S_MLP_FC2) ? {3'b101, 1'b0, mlp_neu[4:0], mlp_feat[6:0]}:
    16'b0;

// ---------------------------------------------------------------------------
// 1-cycle delayed rp_stream (for mlp_block ROM latency compensation)
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) rp_stream_r <= 1'b0;
    else       rp_stream_r <= rp_stream;
end

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:
            next_state = start ? S_LOAD_X : S_IDLE;
        S_LOAD_X:
            // last token, last feature, valid → transition
            next_state = (buf_addr == N_TOKENS*EMBED_DIM-1 && x_valid)
                         ? S_NORM1 : S_LOAD_X;
        S_NORM1:
            // tok_cnt reaches N_TOKENS after last increment; final ln1_done
            next_state = (ln1_done && tok_cnt == N_TOKENS)
                         ? S_ATTN : S_NORM1;
        S_ATTN:
            next_state = attn_done ? S_RESID1 : S_ATTN;
        S_RESID1:
            next_state = (buf_addr == N_TOKENS*EMBED_DIM-1) ? S_NORM2 : S_RESID1;
        S_NORM2:
            next_state = (ln2_done && tok_cnt == N_TOKENS)
                         ? S_MLP_FC1 : S_NORM2;
        S_MLP_FC1:
            // all N_TOKENS × MLP_DIM neurons done
            next_state = (fc1_done && mlp_tok == N_TOKENS-1 && mlp_neu == MLP_DIM-1)
                         ? S_MLP_FC2 : S_MLP_FC1;
        S_MLP_FC2:
            next_state = (fc2_done && mlp_tok == N_TOKENS-1 && mlp_neu == EMBED_DIM-1)
                         ? S_RESID2 : S_MLP_FC2;
        S_RESID2:
            next_state = (buf_addr == N_TOKENS*EMBED_DIM-1) ? S_OUT : S_RESID2;
        S_OUT:
            next_state = (buf_addr == N_TOKENS*EMBED_DIM-1) ? S_DONE : S_OUT;
        S_DONE:
            next_state = S_IDLE;
        default:
            next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: output / datapath logic
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    done     <= 1'b0;
    y_valid  <= 1'b0;
    ln1_start  <= 1'b0;
    ln2_start  <= 1'b0;
    attn_start <= 1'b0;
    fc1_start  <= 1'b0;
    fc2_start  <= 1'b0;

    if (reset) begin
        buf_addr  <= 14'd0;
        tok_cnt   <= 9'd0;
        feat_cnt  <= 5'd0;
        rp_feat   <= 5'd0;
        mlp_feat  <= 8'd0;
        rp_stream <= 1'b0;
        mlp_tok   <= 9'd0;
        mlp_neu   <= 8'd0;
    end else begin
        case (state)

            // ---------------------------------------------------------------
            S_IDLE: begin
                buf_addr  <= 14'd0;
                tok_cnt   <= 9'd0;
                feat_cnt  <= 5'd0;
                rp_feat   <= 5'd0;
                mlp_feat  <= 8'd0;
                rp_stream <= 1'b0;
                mlp_tok   <= 9'd0;
                mlp_neu   <= 8'd0;
            end

            // ---------------------------------------------------------------
            // Load x_i stream → x_buf
            // ---------------------------------------------------------------
            S_LOAD_X: begin
                if (x_valid) begin
                    x_buf[buf_addr] <= x_i;
                    buf_addr <= buf_addr + 14'd1;
                end
            end

            // ---------------------------------------------------------------
            // norm1: for each token, replay x_buf → u_norm1; capture → tmp_buf
            // ---------------------------------------------------------------
            S_NORM1: begin
                // Advance replay counter each streaming cycle
                if (rp_stream) begin
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else begin
                        rp_feat <= rp_feat + 5'd1;
                    end
                end

                // Capture ln1 output → tmp_buf
                if (ln1_yv) begin
                    tmp_buf[tok_cnt * EMBED_DIM + {9'b0, feat_cnt}] <= ln1_y;
                    feat_cnt <= feat_cnt + 5'd1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 5'd0;
                        tok_cnt  <= tok_cnt + 9'd1;
                        // Start next token's norm (not the last)
                        if (tok_cnt < N_TOKENS-1) begin
                            ln1_start <= 1'b1;
                            rp_stream <= 1'b1;
                            rp_feat   <= 5'd0;
                        end
                    end
                end else if (!ln1_busy && !rp_stream && tok_cnt == 9'd0) begin
                    // First token: start
                    ln1_start <= 1'b1;
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
                end

                // Reset buf_addr to 0 for S_ATTN (buf_addr was left at
                // N_TOKENS*EMBED_DIM after S_LOAD_X and is unused in S_NORM1)
                if (ln1_done && tok_cnt == N_TOKENS)
                    buf_addr <= 14'd0;
            end

            // ---------------------------------------------------------------
            // attn: stream tmp_buf → u_attn; capture attn_y → tmp_buf
            // ---------------------------------------------------------------
            S_ATTN: begin
                // Phase 1: stream tmp_buf to u_attn
                if (rp_stream) begin
                    buf_addr <= buf_addr + 14'd1;
                    if (buf_addr == N_TOKENS*EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        buf_addr  <= 14'd0;  // reset for output capture
                    end
                end

                // Start attn and begin streaming
                if (!attn_busy && !rp_stream && buf_addr == 14'd0 && !attn_done) begin
                    attn_start <= 1'b1;
                    rp_stream  <= 1'b1;
                end

                // Phase 2: capture attn output → tmp_buf (after loading phase done)
                if (attn_yv) begin
                    tmp_buf[buf_addr] <= attn_y;
                    buf_addr <= buf_addr + 14'd1;
                end

                // Prepare buf_addr=0 for S_RESID1 on exit
                if (attn_done) begin
                    buf_addr <= 14'd0;
                end
            end

            // ---------------------------------------------------------------
            // resid1: x_buf += tmp_buf (saturated)
            // ---------------------------------------------------------------
            S_RESID1: begin
                x_buf[buf_addr] <= resid_sat;
                if (buf_addr == N_TOKENS*EMBED_DIM-1) begin
                    buf_addr <= 14'd0;
                    tok_cnt  <= 9'd0;    // reset for S_NORM2
                    feat_cnt <= 5'd0;
                    rp_feat  <= 5'd0;
                end else begin
                    buf_addr <= buf_addr + 14'd1;
                end
            end

            // ---------------------------------------------------------------
            // norm2: replay x_buf → u_norm2; capture → tmp_buf
            // ---------------------------------------------------------------
            S_NORM2: begin
                if (rp_stream) begin
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else begin
                        rp_feat <= rp_feat + 5'd1;
                    end
                end

                if (ln2_yv) begin
                    tmp_buf[tok_cnt * EMBED_DIM + {9'b0, feat_cnt}] <= ln2_y;
                    feat_cnt <= feat_cnt + 5'd1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 5'd0;
                        tok_cnt  <= tok_cnt + 9'd1;
                        if (tok_cnt < N_TOKENS-1) begin
                            ln2_start <= 1'b1;
                            rp_stream <= 1'b1;
                            rp_feat   <= 5'd0;
                        end
                    end
                end else if (!ln2_busy && !rp_stream && tok_cnt == 9'd0) begin
                    ln2_start <= 1'b1;
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
                end

                // Prepare counters for S_MLP_FC1
                if (ln2_done && tok_cnt == N_TOKENS) begin
                    mlp_tok  <= 9'd0;
                    mlp_neu  <= 8'd0;
                    rp_feat  <= 5'd0;
                end
            end

            // ---------------------------------------------------------------
            // MLP fc1: for each (token, neuron), replay tmp_buf → u_fc1;
            //          fc1_y → h_buf
            // ---------------------------------------------------------------
            S_MLP_FC1: begin
                // Advance replay counter (EMBED_DIM cycles per neuron)
                if (rp_stream) begin
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else begin
                        rp_feat <= rp_feat + 5'd1;
                    end
                end

                // When fc1 finishes one neuron: store result, start next
                if (fc1_done) begin
                    h_buf[mlp_tok * MLP_DIM + {8'b0, mlp_neu}] <= fc1_y;

                    if (mlp_neu == MLP_DIM-1) begin
                        mlp_neu <= 8'd0;
                        if (mlp_tok < N_TOKENS-1) begin
                            mlp_tok  <= mlp_tok + 9'd1;
                            fc1_start <= 1'b1;
                            rp_stream <= 1'b1;
                            rp_feat   <= 5'd0;
                        end
                        // else: transition to S_MLP_FC2 handled by next_state
                    end else begin
                        mlp_neu  <= mlp_neu + 8'd1;
                        fc1_start <= 1'b1;
                        rp_stream <= 1'b1;
                        rp_feat   <= 5'd0;
                    end
                end else if (!fc1_busy && !rp_stream && mlp_tok == 9'd0
                             && mlp_neu == 8'd0) begin
                    // Start first neuron
                    fc1_start <= 1'b1;
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
                end

                // Prepare for S_MLP_FC2
                if (fc1_done && mlp_tok == N_TOKENS-1 && mlp_neu == MLP_DIM-1) begin
                    mlp_tok  <= 9'd0;
                    mlp_neu  <= 8'd0;
                    mlp_feat <= 8'd0;
                    rp_stream <= 1'b0;
                end
            end

            // ---------------------------------------------------------------
            // MLP fc2: for each (token, neuron), replay h_buf → u_fc2;
            //          fc2_y → tmp_buf
            // ---------------------------------------------------------------
            S_MLP_FC2: begin
                // Advance h_buf replay counter (MLP_DIM cycles per neuron)
                if (rp_stream) begin
                    if (mlp_feat == MLP_DIM-1) begin
                        rp_stream <= 1'b0;
                        mlp_feat  <= 8'd0;
                    end else begin
                        mlp_feat <= mlp_feat + 8'd1;
                    end
                end

                if (fc2_done) begin
                    tmp_buf[mlp_tok * EMBED_DIM + {9'b0, mlp_neu[4:0]}] <= fc2_y;

                    if (mlp_neu == EMBED_DIM-1) begin
                        mlp_neu <= 8'd0;
                        if (mlp_tok < N_TOKENS-1) begin
                            mlp_tok  <= mlp_tok + 9'd1;
                            fc2_start <= 1'b1;
                            rp_stream <= 1'b1;
                            mlp_feat  <= 8'd0;
                        end
                        // else: transition to S_RESID2 handled by next_state
                    end else begin
                        mlp_neu  <= mlp_neu + 8'd1;
                        fc2_start <= 1'b1;
                        rp_stream <= 1'b1;
                        mlp_feat  <= 8'd0;
                    end
                end else if (!fc2_busy && !rp_stream && mlp_tok == 9'd0
                             && mlp_neu == 8'd0) begin
                    fc2_start <= 1'b1;
                    rp_stream <= 1'b1;
                    mlp_feat  <= 8'd0;
                end

                // Prepare for S_RESID2
                if (fc2_done && mlp_tok == N_TOKENS-1 && mlp_neu == EMBED_DIM-1) begin
                    buf_addr  <= 14'd0;
                    rp_stream <= 1'b0;
                end
            end

            // ---------------------------------------------------------------
            // resid2: x_buf += tmp_buf (saturated)
            // ---------------------------------------------------------------
            S_RESID2: begin
                x_buf[buf_addr] <= resid_sat;
                if (buf_addr == N_TOKENS*EMBED_DIM-1)
                    buf_addr <= 14'd0;
                else
                    buf_addr <= buf_addr + 14'd1;
            end

            // ---------------------------------------------------------------
            // out: stream x_buf → y_o / y_valid
            // ---------------------------------------------------------------
            S_OUT: begin
                y_o     <= x_buf[buf_addr];
                y_valid <= 1'b1;
                buf_addr <= buf_addr + 14'd1;
            end

            // ---------------------------------------------------------------
            S_DONE: begin
                done     <= 1'b1;
                buf_addr <= 14'd0;
                tok_cnt  <= 9'd0;
                feat_cnt <= 5'd0;
            end

            default: ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
assign busy = (state != S_IDLE);

endmodule
