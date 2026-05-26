// =============================================================================
// transformer_block.v  (verilog_backbone2 — full block: norm1+attn+res1+norm2+mlp+res2)
//
// Pipeline (matches block_forward in run_backbone_numpy_shared_trunk.py L526-560):
//   external x_i  -> [S_LOAD_X] x_buf
//                 -> [S_NORM1]  tmp_buf  = norm1(x_buf); x_snap -> u_attn Sram_x
//                 -> [S_ATTN_FEED + S_ATTN_WAIT]
//                       S_ATTN_FEED: pulse attn_start (x preloaded during norm1)
//                       S_ATTN_WAIT: capture attn y into tmp_buf
//                 -> [S_RES1]   x_buf    = residual(x_buf, tmp_buf)  // in-place
//                 -> [S_NORM2]  tmp_buf  = norm2(x_buf)
//                 -> [S_MLP_FEED + S_MLP_WAIT]
//                       S_MLP_FEED: pulse mlp_start; FC1 reads norm2 via 2-phase tmp
//                       S_MLP_WAIT: capture mlp y into tmp
//                 -> [S_RES2]   y_o      = residual(x_buf, tmp_buf)  // streamed
//                 -> [S_DONE]
//
// Buffers:
//   USE_REG_BUF  : x_buf / tmp_buf reg arrays (10240 x 16 each)
//   USE_SRAM_BUF : Sram_tok1 u_sram_x + u_sram_tmp inside this module (no top ports)
//
// SRAM read contract (CLK = ~clk, both macros):
//   posedge T:   drive A, CEB=0, WEB=1
//   posedge T+1: Q valid for A@T; consume only after ADDR/USE phase align
//
// Golden (block output): Activation/backbone_blocks_<n>_after_block_out_bi.txt
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

`ifdef USE_REG_BUF
reg signed [15:0] x_buf   [0:TOK_FLAT-1];
reg signed [15:0] tmp_buf [0:TOK_FLAT-1];
`else
// ---- Internal activation SRAM (12288x16 macro; use 10240 entries) ----
reg        sx_ceb;
reg        sx_web;
reg [13:0] sx_addr;
reg [15:0] sx_din;
wire [15:0] sx_q;

reg        st_ceb;
reg        st_web;
reg [13:0] st_addr;
reg [15:0] st_din;
wire [15:0] st_q;

// Compile (reuse same macro module as backbone): Sram_tok1 12288 16 16 s
Sram_tok1 u_sram_x (
    .SLP   (1'b0), .DSLP  (1'b0), .SD    (1'b0), .PUDELAY(),
    .CLK   (~clk), .CEB   (sx_ceb), .WEB   (sx_web), .BIST  (1'b0),
    .CEBM  (), .WEBM  (),
    .A     (sx_addr), .D     (sx_din), .BWEB  (16'b0),
    .AM    (), .DM    (), .BWEBM (16'b0),
    .RTSEL (2'b01), .WTSEL (2'b00), .Q     (sx_q)
);

Sram_tok1 u_sram_tmp (
    .SLP   (1'b0), .DSLP  (1'b0), .SD    (1'b0), .PUDELAY(),
    .CLK   (~clk), .CEB   (st_ceb), .WEB   (st_web), .BIST  (1'b0),
    .CEBM  (), .WEBM  (),
    .A     (st_addr), .D     (st_din), .BWEB  (16'b0),
    .AM    (), .DM    (), .BWEBM (16'b0),
    .RTSEL (2'b01), .WTSEL (2'b00), .Q     (st_q)
);

// 2-phase / multi-phase helpers (SRAM only)
reg        x_norm_phase;   // norm1/2 x read: 0=ADDR, 1=USE -> u_norm1
reg [1:0]  res_subphase;   // S_RES1/S_RES2: 0=RD, 1=FEED, 2=WR or OUT

reg [13:0]       tmp_wr_flat_lat;
reg signed [15:0] tmp_wr_din_lat;
reg              tmp_wr_do;
`endif

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
wire        ln1_out_beat;

`ifndef USE_REG_BUF
wire [13:0] tmp_cap_flat = tok_cnt * EMBED_DIM + {9'b0, ln1_addr[4:0]};
`endif

// Attention sub-block regs / wires
reg                attn_start;
wire               x_snap_wr;
wire [13:0]        x_snap_flat;
wire signed [15:0] x_snap_din;
wire signed [15:0] attn_y;
wire               attn_yv;
wire               attn_busy, attn_done;
wire [12:0]        attn_wgt_addr;

// MLP sub-block (norm2 read: norm_rd_* -> tmp / u_sram_tmp, 2-phase)
reg                mlp_start;
wire               mlp_norm_rd_en;
wire [13:0]        mlp_norm_rd_flat;
wire signed [15:0] mlp_norm_x;
wire signed [15:0] mlp_y;
wire               mlp_yv;
wire               mlp_busy, mlp_done;
wire [15:0]        mlp_wgt_addr;

`ifdef USE_REG_BUF
assign mlp_norm_x = tmp_buf[mlp_norm_rd_flat];
`else
assign mlp_norm_x = st_q;
`endif

// Residual sub-block regs / wires
reg  signed [15:0] res_a, res_b;
reg                res_v;
wire signed [15:0] res_y;
wire               res_v_o;

// Generic streaming pointers reused across capture and residual phases
reg [13:0] cap_ptr;
reg [13:0] res_rp;
reg [13:0] res_wp;

wire in_norm_phase = (state == S_NORM1) || (state == S_NORM2);
wire [13:0] xbuf_rp_addr = tok_cnt * EMBED_DIM + {9'b0, rp_feat};

`ifdef USE_REG_BUF
wire signed [15:0] xbuf_rp_data = x_buf[xbuf_rp_addr];
wire               x_norm_use    = 1'b1;
`else
wire signed [15:0] xbuf_rp_data = sx_q;
wire               x_norm_use    = x_norm_phase;
`endif

wire ln1_xv = rp_stream && in_norm_phase && x_norm_use;

`ifdef USE_REG_BUF
assign x_snap_wr   = (state == S_NORM1) && ln1_yv;
assign x_snap_flat = tok_cnt * EMBED_DIM + {9'b0, ln1_addr[4:0]};
assign x_snap_din  = ln1_y_sat;
`else
assign x_snap_wr   = (state == S_NORM1) && tmp_wr_do;
assign x_snap_flat = tmp_wr_flat_lat;
assign x_snap_din  = tmp_wr_din_lat;
`endif

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
    .y_sat_o(ln1_y_sat),
    .out_beat_o(ln1_out_beat)
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
    .x_i      (16'sd0),
    .x_valid  (1'b0),
    .x_snap_wr   (x_snap_wr),
    .x_snap_flat (x_snap_flat),
    .x_snap_din  (x_snap_din),
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
    .clk          (clk),
    .reset        (reset),
    .start        (mlp_start),
    .norm_rd_en   (mlp_norm_rd_en),
    .norm_rd_flat (mlp_norm_rd_flat),
    .norm_x       (mlp_norm_x),
    .wgt_i        (wgt_i),
    .bias_i       (bias_i),
    .wgt_addr_o   (mlp_wgt_addr),
    .busy         (mlp_busy),
    .done         (mlp_done),
    .y_o          (mlp_y),
    .y_valid      (mlp_yv)
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

assign wgt_addr_o =
    (state == S_NORM1)                              ? {3'b000, 3'b0, ln1_addr[9:0]} :
    (state == S_NORM2)                              ? {3'b001, 3'b0, ln1_addr[9:0]} :
    (state == S_ATTN_FEED || state == S_ATTN_WAIT)  ? {3'b010, attn_wgt_addr}        :
    (state == S_MLP_FEED  || state == S_MLP_WAIT)   ? mlp_wgt_addr                   :
                                                       16'b0;

`ifndef USE_REG_BUF
// ---- SRAM port mux (x_buf / tmp_buf); one R or W per macro per cycle ----
always @(*) begin
    sx_ceb  = 1'b1;
    sx_web  = 1'b1;
    sx_addr = 14'd0;
    sx_din  = 16'd0;

    st_ceb  = 1'b1;
    st_web  = 1'b1;
    st_addr = 14'd0;
    st_din  = 16'd0;

  // x: load write
    if (state == S_LOAD_X && x_valid && (buf_addr < TOK_FLAT[13:0])) begin
        sx_ceb  = 1'b0;
        sx_web  = 1'b0;
        sx_addr = buf_addr;
        sx_din  = x_i[15:0];
    end
  // x: norm read (ADDR phase)
    else if (in_norm_phase && rp_stream && (x_norm_phase == 1'b0)) begin
        sx_ceb  = 1'b0;
        sx_web  = 1'b1;
        sx_addr = xbuf_rp_addr;
    end
  // x: S_RES1 read / write (never same cycle)
    else if (state == S_RES1) begin
        if (res_subphase == 2'd0) begin
            sx_ceb  = 1'b0;
            sx_web  = 1'b1;
            sx_addr = res_rp;
        end else if ((res_subphase == 2'd2) && res_v_o && (res_wp < TOK_FLAT[13:0])) begin
            sx_ceb  = 1'b0;
            sx_web  = 1'b0;
            sx_addr = res_wp;
            sx_din  = res_y[15:0];
        end
    end
  // x: S_RES2 read (ADDR phase)
    else if ((state == S_RES2) && (res_subphase == 2'd0) && (res_rp < TOK_FLAT[13:0])) begin
        sx_ceb  = 1'b0;
        sx_web  = 1'b1;
        sx_addr = res_rp;
    end

  // tmp: norm capture write (latched 1 cycle after out_beat_o)
    if (tmp_wr_do) begin
        st_ceb  = 1'b0;
        st_web  = 1'b0;
        st_addr = tmp_wr_flat_lat;
        st_din  = tmp_wr_din_lat[15:0];
    end
  // tmp: mlp FC1 norm2 read (ADDR); USE next cycle via st_q (no overlap mlp_yv write)
    else if (mlp_norm_rd_en) begin
        st_ceb  = 1'b0;
        st_web  = 1'b1;
        st_addr = mlp_norm_rd_flat;
    end
  // tmp: attn/mlp capture write
    else if ((state == S_ATTN_WAIT) && attn_yv) begin
        st_ceb  = 1'b0;
        st_web  = 1'b0;
        st_addr = cap_ptr;
        st_din  = attn_y[15:0];
    end else if ((state == S_MLP_WAIT) && mlp_yv) begin
        st_ceb  = 1'b0;
        st_web  = 1'b0;
        st_addr = cap_ptr;
        st_din  = mlp_y[15:0];
    end
  // tmp: residual read (with x read in S_RES1 phase 0; with x in S_RES2 phase 0)
    else if (((state == S_RES1) && (res_subphase == 2'd0)) ||
             ((state == S_RES2) && (res_subphase == 2'd0) && (res_rp < TOK_FLAT[13:0]))) begin
        st_ceb  = 1'b0;
        st_web  = 1'b1;
        st_addr = res_rp;
    end
end

// norm y_sat_o posedge latch -> tmp SRAM (layer_norm out_beat_o; see verilog_rule §7.7.3.1)
always @(posedge clk) begin
    if (reset) begin
        tmp_wr_flat_lat <= 14'd0;
        tmp_wr_din_lat  <= 16'd0;
        tmp_wr_do       <= 1'b0;
    end else begin
        tmp_wr_do <= 1'b0;
        if (in_norm_phase && ln1_out_beat) begin
            tmp_wr_flat_lat <= tmp_cap_flat;
            tmp_wr_din_lat  <= ln1_y_sat;
            tmp_wr_do       <= 1'b1;
            feat_cnt        <= feat_cnt + 5'd1;
            if (feat_cnt == EMBED_DIM-1) begin
                feat_cnt <= 5'd0;
                tok_cnt  <= tok_cnt + 9'd1;
            end
        end
    end
end
`endif

always @(posedge clk) begin
    if (reset)
        ln1_start_r <= 1'b0;
    else
        ln1_start_r <= ln1_start;
end

always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

always @(*) begin
    case (state)
        S_IDLE:      next_state = start ? S_LOAD_X : S_IDLE;

        S_LOAD_X:    next_state = ((buf_addr == TOK_FLAT[13:0]) ||
                                   (buf_addr == TOK_FLAT[13:0] - 14'd1 && x_valid))
                                   ? S_NORM1 : S_LOAD_X;

        S_NORM1:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_ATTN_FEED : S_NORM1;
        S_NORM2:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_MLP_FEED  : S_NORM2;

        S_ATTN_FEED: next_state = S_ATTN_WAIT;
        S_ATTN_WAIT: next_state = attn_done ? S_RES1 : S_ATTN_WAIT;

        S_RES1:      next_state = (res_wp == TOK_FLAT[13:0]) ? S_NORM2 : S_RES1;

        S_MLP_FEED:  next_state = S_MLP_WAIT;
        S_MLP_WAIT:  next_state = mlp_done ? S_RES2 : S_MLP_WAIT;

        S_RES2:      next_state = (res_wp == TOK_FLAT[13:0]) ? S_DONE : S_RES2;

        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

always @(posedge clk) begin
    done       <= 1'b0;
    y_valid    <= 1'b0;
    ln1_start  <= 1'b0;
    attn_start <= 1'b0;
    mlp_start  <= 1'b0;
    res_v      <= 1'b0;

    if (reset) begin
        buf_addr    <= 14'd0;
        tok_cnt     <= 9'd0;
        feat_cnt    <= 5'd0;
        rp_feat     <= 5'd0;
        rp_stream   <= 1'b0;
        cap_ptr     <= 14'd0;
        res_rp      <= 14'd0;
        res_wp      <= 14'd0;
        res_a       <= 16'sd0;
        res_b       <= 16'sd0;
        y_o         <= 16'sd0;
`ifndef USE_REG_BUF
        x_norm_phase <= 1'b0;
        res_subphase <= 2'd0;
`endif
    end else begin
        case (state)
            S_IDLE: begin
                buf_addr    <= 14'd0;
                tok_cnt     <= 9'd0;
                feat_cnt    <= 5'd0;
                rp_feat     <= 5'd0;
                rp_stream   <= 1'b0;
                cap_ptr     <= 14'd0;
                res_rp      <= 14'd0;
                res_wp      <= 14'd0;
`ifndef USE_REG_BUF
                x_norm_phase <= 1'b0;
                res_subphase <= 2'd0;
`endif
            end

            S_LOAD_X: begin
`ifdef USE_REG_BUF
                if (x_valid && (buf_addr < TOK_FLAT[13:0])) begin
                    x_buf[buf_addr] <= x_i;
                    buf_addr <= buf_addr + 14'd1;
                end
`else
                if (x_valid && (buf_addr < TOK_FLAT[13:0]))
                    buf_addr <= buf_addr + 14'd1;
`endif
            end

            S_NORM1: begin
                if (rp_stream) begin
`ifndef USE_REG_BUF
                    if (x_norm_phase == 1'b0)
                        x_norm_phase <= 1'b1;
                    else begin
                        x_norm_phase <= 1'b0;
                        if (rp_feat == EMBED_DIM-1) begin
                            rp_stream <= 1'b0;
                            rp_feat   <= 5'd0;
                        end else
                            rp_feat <= rp_feat + 5'd1;
                    end
`else
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else
                        rp_feat <= rp_feat + 5'd1;
`endif
                end

                if (ln1_start_r) begin
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
`ifndef USE_REG_BUF
                    x_norm_phase <= 1'b0;
`endif
                end

`ifdef USE_REG_BUF
                if (ln1_yv) begin
                    tmp_buf[tok_cnt * EMBED_DIM + ln1_addr[4:0]] <= ln1_y_sat;
                    feat_cnt <= feat_cnt + 5'd1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 5'd0;
                        tok_cnt  <= tok_cnt + 9'd1;
                    end
                end
`endif

                if (ln1_done && tok_cnt < N_TOKENS)
                    ln1_start <= 1'b1;
                else if (!ln1_busy && !rp_stream && tok_cnt == 9'd0 && !ln1_start)
                    ln1_start <= 1'b1;

                if (ln1_done && tok_cnt == N_TOKENS) begin
                    cap_ptr     <= 14'd0;
                    tok_cnt     <= 9'd0;
                    feat_cnt    <= 5'd0;
                    rp_feat     <= 5'd0;
                end
            end

            S_ATTN_FEED: begin
                attn_start <= 1'b1;
                cap_ptr    <= 14'd0;
            end

            S_ATTN_WAIT: begin
`ifdef USE_REG_BUF
                if (attn_yv) begin
                    tmp_buf[cap_ptr] <= attn_y;
                    if (cap_ptr < TOK_FLAT[13:0] - 14'd1)
                        cap_ptr <= cap_ptr + 14'd1;
                end
`else
                if (attn_yv && (cap_ptr < TOK_FLAT[13:0] - 14'd1))
                    cap_ptr <= cap_ptr + 14'd1;
`endif
                if (attn_done) begin
                    res_rp  <= 14'd0;
                    res_wp  <= 14'd0;
                    cap_ptr <= 14'd0;
`ifndef USE_REG_BUF
                    res_subphase <= 2'd0;
`endif
                end
            end

            S_RES1: begin
`ifdef USE_REG_BUF
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
`else
                case (res_subphase)
                    2'd0: res_subphase <= 2'd1;
                    2'd1: begin
                        res_a  <= sx_q;
                        res_b  <= st_q;
                        res_v  <= 1'b1;
                        res_subphase <= 2'd2;
                    end
                    2'd2: begin
                        if (res_v_o && (res_wp < TOK_FLAT[13:0])) begin
                            res_wp <= res_wp + 14'd1;
                            if (res_rp < TOK_FLAT[13:0] - 14'd1)
                                res_rp <= res_rp + 14'd1;
                            res_subphase <= 2'd0;
                        end
                    end
                    default: res_subphase <= 2'd0;
                endcase
`endif
                if (next_state == S_NORM2) begin
                    tok_cnt   <= 9'd0;
                    feat_cnt  <= 5'd0;
                    rp_feat   <= 5'd0;
                    rp_stream <= 1'b0;
`ifndef USE_REG_BUF
                    x_norm_phase <= 1'b0;
`endif
                end
            end

            S_NORM2: begin
                if (rp_stream) begin
`ifndef USE_REG_BUF
                    if (x_norm_phase == 1'b0)
                        x_norm_phase <= 1'b1;
                    else begin
                        x_norm_phase <= 1'b0;
                        if (rp_feat == EMBED_DIM-1) begin
                            rp_stream <= 1'b0;
                            rp_feat   <= 5'd0;
                        end else
                            rp_feat <= rp_feat + 5'd1;
                    end
`else
                    if (rp_feat == EMBED_DIM-1) begin
                        rp_stream <= 1'b0;
                        rp_feat   <= 5'd0;
                    end else
                        rp_feat <= rp_feat + 5'd1;
`endif
                end

                if (ln1_start_r) begin
                    rp_stream <= 1'b1;
                    rp_feat   <= 5'd0;
`ifndef USE_REG_BUF
                    x_norm_phase <= 1'b0;
`endif
                end

`ifdef USE_REG_BUF
                if (ln1_yv) begin
                    tmp_buf[tok_cnt * EMBED_DIM + ln1_addr[4:0]] <= ln1_y_sat;
                    feat_cnt <= feat_cnt + 5'd1;
                    if (feat_cnt == EMBED_DIM-1) begin
                        feat_cnt <= 5'd0;
                        tok_cnt  <= tok_cnt + 9'd1;
                    end
                end
`endif

                if (ln1_done && tok_cnt < N_TOKENS)
                    ln1_start <= 1'b1;
                else if (!ln1_busy && !rp_stream && tok_cnt == 9'd0 && !ln1_start)
                    ln1_start <= 1'b1;

                if (ln1_done && tok_cnt == N_TOKENS) begin
                    cap_ptr     <= 14'd0;
                end
            end

            S_MLP_FEED: begin
                mlp_start <= 1'b1;
                cap_ptr   <= 14'd0;
            end

            S_MLP_WAIT: begin
`ifdef USE_REG_BUF
                if (mlp_yv) begin
                    tmp_buf[cap_ptr] <= mlp_y;
                    if (cap_ptr < TOK_FLAT[13:0] - 14'd1)
                        cap_ptr <= cap_ptr + 14'd1;
                end
`else
                if (mlp_yv && (cap_ptr < TOK_FLAT[13:0] - 14'd1))
                    cap_ptr <= cap_ptr + 14'd1;
`endif
                if (mlp_done) begin
                    res_rp <= 14'd0;
                    res_wp <= 14'd0;
                    cap_ptr <= 14'd0;
`ifndef USE_REG_BUF
                    res_subphase <= 2'd0;
`endif
                end
            end

            S_RES2: begin
`ifdef USE_REG_BUF
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
`else
                case (res_subphase)
                    2'd0: res_subphase <= 2'd1;
                    2'd1: begin
                        res_a  <= sx_q;
                        res_b  <= st_q;
                        res_v  <= 1'b1;
                        res_subphase <= 2'd2;
                    end
                    2'd2: begin
                        if (res_v_o && (res_wp < TOK_FLAT[13:0])) begin
                            y_o     <= res_y;
                            y_valid <= 1'b1;
                            res_wp  <= res_wp + 14'd1;
                            if (res_rp < TOK_FLAT[13:0] - 14'd1)
                                res_rp <= res_rp + 14'd1;
                            res_subphase <= 2'd0;
                        end
                    end
                    default: res_subphase <= 2'd0;
                endcase
`endif
            end

            S_DONE: begin
                done     <= 1'b1;
                tok_cnt  <= 9'd0;
                feat_cnt <= 5'd0;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
