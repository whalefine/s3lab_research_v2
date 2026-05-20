// =============================================================================
// transformer_block.v  (verilog_backbone2 -- norm1 + CARE attention slice)
//
// Scope: external token stream -> norm1 -> care_attention -> y_o stream.
// No norm2 / MLP / residuals in this revision (next-stage scope).
//
// FSM:
//   S_IDLE       -> wait for start
//   S_LOAD_X     -> capture external x_i into x_buf (N_TOKENS*EMBED_DIM beats)
//   S_NORM1      -> stream x_buf through u_norm1; write y_sat into tmp_buf
//   S_ATTN_FEED  -> pulse u_attn.start, stream tmp_buf into u_attn.x_i
//   S_ATTN_WAIT  -> forward u_attn.y_o -> y_o until u_attn.done
//   S_DONE       -> 1-cycle done pulse, back to S_IDLE
//
// wgt_addr_o[15:13] (wtype):
//   3'b000 (norm1)     in S_NORM1
//   3'b010 (attn)      in S_ATTN_FEED / S_ATTN_WAIT (qkv if local<3072 else proj)
//   3'b000 (don't care, ROM is gated) otherwise
//
// Golden:
//   norm1 :   Activation/backbone_blocks_<b>_after_norm1_out_bi.txt
//   attn  :   Activation/backbone_blocks_<b>_after_attn_attn_out_bi.txt
//   q/k/v :   Activation/backbone_blocks_<b>_attn_after_qkv_{q,k,v}_bi.txt
//             (captured pre-SPLIT inside u_attn.q_buf/k_buf/v_buf)
//
// Debug defines:
//   +define+DUMP_NORM1_DEBUG  -> TB stops after block0 norm1 finishes
//   +define+DUMP_ATTN_DEBUG   -> TB compares q/k/v + attn_out vs golden
// =============================================================================

module transformer_block #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,   // unused; kept for backbone_top port compatibility
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

parameter S_IDLE      = 4'd0;
parameter S_LOAD_X    = 4'd1;
parameter S_NORM1     = 4'd2;
parameter S_ATTN_FEED = 4'd3;
parameter S_ATTN_WAIT = 4'd4;
parameter S_DONE      = 4'd5;

reg [3:0] state, next_state;

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

// Attention sub-block signals
reg                attn_start;
reg  signed [15:0] attn_x;
reg                attn_xv;
wire signed [15:0] attn_y;
wire               attn_yv;
wire               attn_busy, attn_done;
wire [12:0]        attn_wgt_addr;

// Streaming pointer feeding tmp_buf into u_attn
reg [13:0] feed_ptr;
reg        feed_active;

`ifdef DUMP_TB_NODES
integer dump_norm1_f;
initial begin
    dump_norm1_f = $fopen("rtl_backbone_blocks_0_after_norm1_out_bi.txt", "w");
end
`endif

wire [13:0] xbuf_rp_addr = tok_cnt * EMBED_DIM + {9'b0, rp_feat};
wire signed [15:0] xbuf_rp_data = x_buf[xbuf_rp_addr];

wire ln1_xv = rp_stream && (state == S_NORM1);

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

// Route weight address: norm1 vs attention
assign wgt_addr_o = (state == S_NORM1)                                ? {3'b000, 3'b0, ln1_addr[9:0]} :
                    (state == S_ATTN_FEED || state == S_ATTN_WAIT)    ? {3'b010, attn_wgt_addr}        :
                                                                          16'b0;

// Sequential register for ln1_start (1-cycle delayed for handshake)
always @(posedge clk) begin
    if (reset)
        ln1_start_r <= 1'b0;
    else
        ln1_start_r <= ln1_start;
end

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:      next_state = start ? S_LOAD_X : S_IDLE;
        S_LOAD_X:    next_state = ((buf_addr == TOK_FLAT[13:0]) ||
                                   (buf_addr == TOK_FLAT[13:0] - 14'd1 && x_valid))
                                   ? S_NORM1 : S_LOAD_X;
        S_NORM1:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_ATTN_FEED : S_NORM1;
        S_ATTN_FEED: next_state = (feed_ptr == TOK_FLAT[13:0] - 14'd1) ? S_ATTN_WAIT : S_ATTN_FEED;
        S_ATTN_WAIT: next_state = attn_done ? S_DONE : S_ATTN_WAIT;
        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

// FSM segment 3: datapath
always @(posedge clk) begin
    done       <= 1'b0;
    y_valid    <= 1'b0;
    ln1_start  <= 1'b0;
    attn_start <= 1'b0;
    attn_xv    <= 1'b0;

    if (reset) begin
        buf_addr    <= 14'd0;
        tok_cnt     <= 9'd0;
        feat_cnt    <= 5'd0;
        rp_feat     <= 5'd0;
        rp_stream   <= 1'b0;
        feed_ptr    <= 14'd0;
        feed_active <= 1'b0;
        attn_x      <= 16'sd0;
        y_o         <= 16'sd0;
    end else begin
        case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                buf_addr    <= 14'd0;
                tok_cnt     <= 9'd0;
                feat_cnt    <= 5'd0;
                rp_feat     <= 5'd0;
                rp_stream   <= 1'b0;
                feed_ptr    <= 14'd0;
                feed_active <= 1'b0;
            end

            // ----------------------------------------------------------
            // S_LOAD_X: external x_i -> x_buf
            // ----------------------------------------------------------
            S_LOAD_X: begin
                if (x_valid && (buf_addr < TOK_FLAT[13:0])) begin
                    x_buf[buf_addr] <= x_i;
                    buf_addr <= buf_addr + 14'd1;
                end
            end

            // ----------------------------------------------------------
            // S_NORM1: stream x_buf through u_norm1, capture into tmp_buf.
            //   Same handshake as previous revision; no functional change.
            // ----------------------------------------------------------
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
`ifdef DUMP_TB_NODES
                    if (block_idx == 4'd0) begin
                        $fwrite(dump_norm1_f, "%016b\n", ln1_y_sat[15:0]);
                        if (tok_cnt == N_TOKENS-1 && feat_cnt == EMBED_DIM-1)
                            $fclose(dump_norm1_f);
                    end
`endif
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
                    // Norm1 finished -- reset pointers for attention feed.
                    feed_ptr    <= 14'd0;
                    feed_active <= 1'b0;
                end
            end

            // ----------------------------------------------------------
            // S_ATTN_FEED: pulse attn_start on first cycle, then stream
            //   tmp_buf[0..TOK_FLAT-1] into u_attn.x_i with attn_xv=1.
            //   u_attn captures into its x_in_buf during its S_LOAD_X.
            // ----------------------------------------------------------
            S_ATTN_FEED: begin
                if (!feed_active) begin
                    // First cycle in this state: pulse start; begin streaming
                    attn_start  <= 1'b1;
                    feed_active <= 1'b1;
                    feed_ptr    <= 14'd0;
                end else begin
                    // Streaming beats
                    attn_x   <= tmp_buf[feed_ptr];
                    attn_xv  <= 1'b1;
                    if (feed_ptr < TOK_FLAT[13:0] - 14'd1)
                        feed_ptr <= feed_ptr + 14'd1;
                    // else: hold at last index; FSM transitions to S_ATTN_WAIT
                end
            end

            // ----------------------------------------------------------
            // S_ATTN_WAIT: u_attn runs internal QKV/SPLIT/.../PROJ.
            //   Forward u_attn.y_o -> y_o (one beat per PROJ neuron).
            //   When attn_done pulses -> S_DONE.
            // ----------------------------------------------------------
            S_ATTN_WAIT: begin
                feed_active <= 1'b0;
                if (attn_yv) begin
                    y_o     <= attn_y;
                    y_valid <= 1'b1;
                end
            end

            // ----------------------------------------------------------
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
