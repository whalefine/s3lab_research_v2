// =============================================================================
// transformer_block.v  (verilog_backbone2 — full block: norm1+attn+res1+norm2+mlp+res2)
//
// Pipeline (matches block_forward in run_backbone_numpy_shared_trunk.py L526-560):
//   external x_i  -> [S_LOAD_X] x_buf
//                 -> [S_NORM1]  tmp_buf  = norm1(x_buf)
//                 -> [S_ATTN_FEED + S_ATTN_WAIT]
//                       u_attn streams tmp_buf in, emits attn output back into
//                       tmp_buf (reused; norm1 output no longer needed)
//                 -> [S_RES1]   x_buf    = residual(x_buf, tmp_buf)  // in-place
//                 -> [S_NORM2]  tmp_buf  = norm2(x_buf)
//                 -> [S_MLP_FEED + S_MLP_WAIT]
//                       u_mlp streams tmp_buf in, emits mlp output into tmp_buf
//                 -> [S_RES2]   y_o      = residual(x_buf, tmp_buf)  // streamed
//                 -> [S_DONE]
//
// u_norm1 is reused for norm2 (same Q8.8 LayerNorm math); wgt_addr_o uses
// wtype=3'b000 for norm1 and 3'b001 for norm2 (matches backbone_top.v ROM mux).
//
// u_res is reused for residual1 and residual2 (same Q8.8 add+sat); operands are
// always x_buf[ptr] and tmp_buf[ptr]; only the destination of u_res.y_o differs.
//
// ROM weight types routed through wgt_addr_o:
//   3'b000 -> norm1   (during S_NORM1)
//   3'b001 -> norm2   (during S_NORM2)
//   3'b010 -> attn    (during S_ATTN_FEED / S_ATTN_WAIT; qkv or proj inside)
//   3'b100 -> mlp_fc1 } emitted directly by u_mlp during S_MLP_FEED / S_MLP_WAIT
//   3'b101 -> mlp_fc2 }
//
// Buffers (sim reg arrays — APR needs SRAM macros):
//   x_buf   : 10240 × 16 — input  -> after S_RES1 holds residual1 sum
//   tmp_buf : 10240 × 16 — norm1  -> attn -> norm2 -> mlp (reused each phase)
// =============================================================================

module transformer_block #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [15:0] wgt_addr_o,

    input  wire [3:0]  block_idx,

    output wire        busy,
    output reg         done,

    output reg  signed [15:0] y_o,
    output reg         y_valid
);

parameter LN_RCP   = 65536 / EMBED_DIM;
parameter TOK_FLAT = N_TOKENS * EMBED_DIM;   // 10240

reg signed [15:0] x_buf   [0:TOK_FLAT-1];
reg signed [15:0] tmp_buf [0:TOK_FLAT-1];

// 4-bit FSM (11 states)
parameter S_IDLE      = 4'd0;
parameter S_LOAD_X    = 4'd1;
parameter S_NORM1     = 4'd2;
parameter S_ATTN_FEED = 4'd3;
parameter S_ATTN_WAIT = 4'd4;
parameter S_RES1      = 4'd5;
parameter S_NORM2     = 4'd6;
parameter S_MLP_FEED  = 4'd7;
parameter S_MLP_WAIT  = 4'd8;
parameter S_RES2      = 4'd9;
parameter S_DONE      = 4'd10;

reg [3:0] state, next_state;

// Shared norm streaming regs (used by both S_NORM1 and S_NORM2)
reg [13:0] buf_addr;
reg [8:0]  tok_cnt;
reg [4:0]  feat_cnt;
reg [4:0]  rp_feat;
reg        rp_stream;
reg        ln1_start_r;

reg  ln1_start;
wire ln1_busy, ln1_done;
wire signed [15:0] ln1_y;
wire signed [15:0] ln1_y_sat;
wire ln1_yv;
wire [9:0]  ln1_addr;

// Attention sub-block regs / wires
reg                attn_start;
reg  signed [15:0] attn_x;
reg                attn_xv;
wire signed [15:0] attn_y;
wire               attn_yv;
wire               attn_busy, attn_done;
wire [12:0]        attn_wgt_addr;

// MLP sub-block regs / wires
reg                mlp_start;
reg  signed [15:0] mlp_x;
reg                mlp_xv;
wire signed [15:0] mlp_y;
wire               mlp_yv;
wire               mlp_busy, mlp_done;
wire [15:0]        mlp_wgt_addr;

// Residual sub-block regs / wires
reg  signed [15:0] res_a, res_b;
reg                res_v;
wire signed [15:0] res_y;
wire               res_v_o;

// Generic streaming pointers reused across attn/mlp feed and residual phases
reg [13:0] feed_ptr;
reg        feed_active;
reg [13:0] cap_ptr;      // capture-into-tmp_buf pointer for attn / mlp WAIT phases
reg [13:0] res_rp;       // residual read pointer
reg [13:0] res_wp;       // residual write pointer (lags res_rp by 2 cycles)

// ---------------------------------------------------------------------------
// Norm1/Norm2 shared streaming addr — reads x_buf for both phases.
// (S_NORM1: x_buf = input, S_NORM2: x_buf = residual1 result)
// ---------------------------------------------------------------------------
wire [13:0] xbuf_rp_addr = tok_cnt * EMBED_DIM + {9'b0, rp_feat};
wire signed [15:0] xbuf_rp_data = x_buf[xbuf_rp_addr];

wire in_norm_phase = (state == S_NORM1) || (state == S_NORM2);
wire ln1_xv = rp_stream && in_norm_phase;

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
    .y_o(ln1_y), .y_valid(ln1_yv),
    .y_sat_o(ln1_y_sat)
);

care_attention #(
    .EMBED_DIM(EMBED_DIM),
    .NUM_HEADS(4),
    .HEAD_DIM (EMBED_DIM/4),
    .N_TOKENS (N_TOKENS)
) u_attn (
    .clk      (clk),
    .reset    (reset),
    .start    (attn_start),
    .x_i      (attn_x),
    .x_valid  (attn_xv),
    .wgt_i    (wgt_i),
    .bias_i   (bias_i),
    .wgt_addr_o(attn_wgt_addr),
    .busy     (attn_busy),
    .done     (attn_done),
    .y_o      (attn_y),
    .y_valid  (attn_yv)
);

mlp #(
    .EMBED_DIM(EMBED_DIM),
    .MLP_DIM  (MLP_DIM),
    .N_TOKENS (N_TOKENS)
) u_mlp (
    .clk      (clk),
    .reset    (reset),
    .start    (mlp_start),
    .x_i      (mlp_x),
    .x_valid  (mlp_xv),
    .wgt_i    (wgt_i),
    .bias_i   (bias_i),
    .wgt_addr_o(mlp_wgt_addr),
    .busy     (mlp_busy),
    .done     (mlp_done),
    .y_o      (mlp_y),
    .y_valid  (mlp_yv)
);

residual #(.WIDTH(16)) u_res (
    .clk  (clk),
    .reset(reset),
    .a_i  (res_a),
    .b_i  (res_b),
    .v_i  (res_v),
    .y_o  (res_y),
    .v_o  (res_v_o)
);

// ---------------------------------------------------------------------------
// wgt_addr_o mux: select wtype + local addr based on current phase
// ---------------------------------------------------------------------------
assign wgt_addr_o =
    (state == S_NORM1)                              ? {3'b000, 3'b0, ln1_addr[9:0]} :
    (state == S_NORM2)                              ? {3'b001, 3'b0, ln1_addr[9:0]} :
    (state == S_ATTN_FEED || state == S_ATTN_WAIT)  ? {3'b010, attn_wgt_addr}        :
    (state == S_MLP_FEED  || state == S_MLP_WAIT)   ? mlp_wgt_addr                   :
                                                       16'b0;

// ---------------------------------------------------------------------------
// ln1_start handshake helper (1-cycle delay so rp_stream rises after layer_norm
// has started its internal load).
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        ln1_start_r <= 1'b0;
    else
        ln1_start_r <= ln1_start;
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
        S_IDLE:      next_state = start ? S_LOAD_X : S_IDLE;

        S_LOAD_X:    next_state = ((buf_addr == TOK_FLAT[13:0]) ||
                                   (buf_addr == TOK_FLAT[13:0] - 14'd1 && x_valid))
                                   ? S_NORM1 : S_LOAD_X;

        // Norm1 / Norm2 share the same tok_cnt + ln1_done indicator.
        S_NORM1:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_ATTN_FEED : S_NORM1;
        S_NORM2:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_MLP_FEED  : S_NORM2;

        S_ATTN_FEED: next_state = (feed_ptr == TOK_FLAT[13:0] - 14'd1) ? S_ATTN_WAIT : S_ATTN_FEED;
        S_ATTN_WAIT: next_state = attn_done ? S_RES1 : S_ATTN_WAIT;

        // S_RES1 finishes when all TOK_FLAT residuals have been written back.
        S_RES1:      next_state = (res_wp == TOK_FLAT[13:0]) ? S_NORM2 : S_RES1;

        S_MLP_FEED:  next_state = (feed_ptr == TOK_FLAT[13:0] - 14'd1) ? S_MLP_WAIT : S_MLP_FEED;
        S_MLP_WAIT:  next_state = mlp_done ? S_RES2 : S_MLP_WAIT;

        S_RES2:      next_state = (res_wp == TOK_FLAT[13:0]) ? S_DONE : S_RES2;

        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: datapath
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    // Default signal drivers (overridden inside state branches when active)
    done       <= 1'b0;
    y_valid    <= 1'b0;
    ln1_start  <= 1'b0;
    attn_start <= 1'b0;
    attn_xv    <= 1'b0;
    mlp_start  <= 1'b0;
    mlp_xv     <= 1'b0;
    res_v      <= 1'b0;

    if (reset) begin
        buf_addr    <= 14'd0;
        tok_cnt     <= 9'd0;
        feat_cnt    <= 5'd0;
        rp_feat     <= 5'd0;
        rp_stream   <= 1'b0;
        feed_ptr    <= 14'd0;
        feed_active <= 1'b0;
        cap_ptr     <= 14'd0;
        res_rp      <= 14'd0;
        res_wp      <= 14'd0;
        attn_x      <= 16'sd0;
        mlp_x       <= 16'sd0;
        res_a       <= 16'sd0;
        res_b       <= 16'sd0;
        y_o         <= 16'sd0;
    end else begin
        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                buf_addr    <= 14'd0;
                tok_cnt     <= 9'd0;
                feat_cnt    <= 5'd0;
                rp_feat     <= 5'd0;
                rp_stream   <= 1'b0;
                feed_ptr    <= 14'd0;
                feed_active <= 1'b0;
                cap_ptr     <= 14'd0;
                res_rp      <= 14'd0;
                res_wp      <= 14'd0;
            end

            // -----------------------------------------------------------
            // S_LOAD_X: external x_i -> x_buf
            // -----------------------------------------------------------
            S_LOAD_X: begin
                if (x_valid && (buf_addr < TOK_FLAT[13:0])) begin
                    x_buf[buf_addr] <= x_i;
                    buf_addr <= buf_addr + 14'd1;
                end
            end

            // -----------------------------------------------------------
            // S_NORM1: stream x_buf -> u_norm1; capture y_sat -> tmp_buf
            // -----------------------------------------------------------
            S_NORM1: begin
                if (rp_stream) begin
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else begin
                        rp_feat <= rp_feat + 5'd1;
                    end
                end

                if (ln1_start_r) begin
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
                end

                if (ln1_yv) begin
                    tmp_buf[tok_cnt * EMBED_DIM + ln1_addr[4:0]] <= ln1_y_sat;
                    feat_cnt <= feat_cnt + 5'd1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 5'd0;
                        tok_cnt  <= tok_cnt + 9'd1;
                    end
                end

                if (ln1_done && tok_cnt < N_TOKENS)
                    ln1_start <= 1'b1;
                else if (!ln1_busy && !rp_stream && tok_cnt == 9'd0 && !ln1_start)
                    ln1_start <= 1'b1;

                if (ln1_done && tok_cnt == N_TOKENS) begin
                    // Norm1 done — prepare attention feed pointers.
                    feed_ptr    <= 14'd0;
                    feed_active <= 1'b0;
                    cap_ptr     <= 14'd0;
                    // Clear norm counters so S_NORM2 can reuse them.
                    tok_cnt     <= 9'd0;
                    feat_cnt    <= 5'd0;
                    rp_feat     <= 5'd0;
                end
            end

            // -----------------------------------------------------------
            // S_ATTN_FEED: stream tmp_buf -> u_attn.x_i (one beat per cycle)
            // -----------------------------------------------------------
            S_ATTN_FEED: begin
                if (!feed_active) begin
                    attn_start  <= 1'b1;
                    feed_active <= 1'b1;
                    feed_ptr    <= 14'd0;
                end else begin
                    attn_x  <= tmp_buf[feed_ptr];
                    attn_xv <= 1'b1;
                    if (feed_ptr < TOK_FLAT[13:0] - 14'd1)
                        feed_ptr <= feed_ptr + 14'd1;
                end
            end

            // -----------------------------------------------------------
            // S_ATTN_WAIT: capture u_attn.y_o into tmp_buf (reuse buffer;
            //   norm1 output is no longer needed past S_ATTN_FEED).
            // -----------------------------------------------------------
            S_ATTN_WAIT: begin
                feed_active <= 1'b0;
                if (attn_yv) begin
                    tmp_buf[cap_ptr] <= attn_y;
                    if (cap_ptr < TOK_FLAT[13:0] - 14'd1)
                        cap_ptr <= cap_ptr + 14'd1;
                end
                if (attn_done) begin
                    // Prepare residual-1 pointers.
                    res_rp <= 14'd0;
                    res_wp <= 14'd0;
                    cap_ptr <= 14'd0;
                end
            end

            // -----------------------------------------------------------
            // S_RES1: u_res(a=x_buf[res_rp], b=tmp_buf[res_rp]).
            //   u_res has 1-cycle latency; we feed at res_rp, write back at
            //   res_wp (= res_rp lagged). res_rp walks 0..TOK_FLAT;
            //   res_wp walks 0..TOK_FLAT in sync with u_res.v_o.
            //   x_buf is updated in-place (read addr ≠ write addr).
            // -----------------------------------------------------------
            S_RES1: begin
                if (res_rp < TOK_FLAT[13:0]) begin
                    res_a  <= x_buf[res_rp];
                    res_b  <= tmp_buf[res_rp];
                    res_v  <= 1'b1;
                    res_rp <= res_rp + 14'd1;
                end
                if (res_v_o && res_wp < TOK_FLAT[13:0]) begin
                    x_buf[res_wp] <= res_y;
                    res_wp <= res_wp + 14'd1;
                end
                if (next_state == S_NORM2) begin
                    // Norm2 will reuse u_norm1; reset its streaming regs.
                    tok_cnt   <= 9'd0;
                    feat_cnt  <= 5'd0;
                    rp_feat   <= 5'd0;
                    rp_stream <= 1'b0;
                end
            end

            // -----------------------------------------------------------
            // S_NORM2: identical streaming as S_NORM1 but with wtype=001
            //   (different ROM). x_buf now holds residual1 output.
            // -----------------------------------------------------------
            S_NORM2: begin
                if (rp_stream) begin
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else begin
                        rp_feat <= rp_feat + 5'd1;
                    end
                end

                if (ln1_start_r) begin
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
                end

                if (ln1_yv) begin
                    tmp_buf[tok_cnt * EMBED_DIM + ln1_addr[4:0]] <= ln1_y_sat;
                    feat_cnt <= feat_cnt + 5'd1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 5'd0;
                        tok_cnt  <= tok_cnt + 9'd1;
                    end
                end

                if (ln1_done && tok_cnt < N_TOKENS)
                    ln1_start <= 1'b1;
                else if (!ln1_busy && !rp_stream && tok_cnt == 9'd0 && !ln1_start)
                    ln1_start <= 1'b1;

                if (ln1_done && tok_cnt == N_TOKENS) begin
                    // Norm2 done — prepare MLP feed pointers.
                    feed_ptr    <= 14'd0;
                    feed_active <= 1'b0;
                    cap_ptr     <= 14'd0;
                end
            end

            // -----------------------------------------------------------
            // S_MLP_FEED: stream tmp_buf -> u_mlp.x_i
            // -----------------------------------------------------------
            S_MLP_FEED: begin
                if (!feed_active) begin
                    mlp_start   <= 1'b1;
                    feed_active <= 1'b1;
                    feed_ptr    <= 14'd0;
                end else begin
                    mlp_x  <= tmp_buf[feed_ptr];
                    mlp_xv <= 1'b1;
                    if (feed_ptr < TOK_FLAT[13:0] - 14'd1)
                        feed_ptr <= feed_ptr + 14'd1;
                end
            end

            // -----------------------------------------------------------
            // S_MLP_WAIT: capture u_mlp.y_o into tmp_buf (reuse; norm2 output
            //   was consumed by S_MLP_FEED).
            // -----------------------------------------------------------
            S_MLP_WAIT: begin
                feed_active <= 1'b0;
                if (mlp_yv) begin
                    tmp_buf[cap_ptr] <= mlp_y;
                    if (cap_ptr < TOK_FLAT[13:0] - 14'd1)
                        cap_ptr <= cap_ptr + 14'd1;
                end
                if (mlp_done) begin
                    // Prepare residual-2 pointers.
                    res_rp <= 14'd0;
                    res_wp <= 14'd0;
                    cap_ptr <= 14'd0;
                end
            end

            // -----------------------------------------------------------
            // S_RES2: u_res(a=x_buf[res_rp], b=tmp_buf[res_rp]); stream
            //   u_res.y_o as module y_o (final block output).
            // -----------------------------------------------------------
            S_RES2: begin
                if (res_rp < TOK_FLAT[13:0]) begin
                    res_a  <= x_buf[res_rp];
                    res_b  <= tmp_buf[res_rp];
                    res_v  <= 1'b1;
                    res_rp <= res_rp + 14'd1;
                end
                if (res_v_o && res_wp < TOK_FLAT[13:0]) begin
                    y_o     <= res_y;
                    y_valid <= 1'b1;
                    res_wp  <= res_wp + 14'd1;
                end
            end

            // -----------------------------------------------------------
            S_DONE: begin
                done    <= 1'b1;
                tok_cnt <= 9'd0;
                feat_cnt<= 5'd0;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
