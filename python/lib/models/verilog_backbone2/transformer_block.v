// =============================================================================
// transformer_block.v  (verilog_backbone2 — full block: norm1+attn+res1+norm2+mlp+res2)
// -----------------------------------------------------------------------------
// Pipeline (matches block_forward in run_backbone_numpy_shared_trunk.py L526-560):
//   external x_i  -> [S_LOAD_X] x_buf
//                 -> [S_NORM1]  norm1 -> tok1 (parent); attn QKV 2-phase read tok1
//                 -> [S_ATTN_FEED + S_ATTN_WAIT]
//                       S_ATTN_WAIT: capture attn into tmp-on-q (Sram_q macro)
//                 -> [S_RES1]   x_buf    = residual(x_buf, tmp_buf)  // in-place
//                 -> [S_NORM2]  tmp_buf  = norm2(x_buf)
//                 -> [S_MLP_FEED + S_MLP_WAIT]
//                       S_MLP_FEED: pulse mlp_start; FC1 reads norm2 via 2-phase tmp
//                       S_MLP_WAIT: capture mlp y into tmp
//                 -> [S_RES2]   y_o      = residual(x_buf, tmp_buf)  // streamed
//                 -> [S_DONE]
//
// Buffers: Sram_tok2 x_buf; tmp-on-q (Sram_q, time-shared with care q/k/v capture).
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
    clk,
    reset,
    start,
    x_i,
    x_valid,
    wgt_i,
    bias_i,
    wgt_addr_o,
    busy,
    x_ready,
    done,
    y_o,
    y_valid,
    sram_tok2_ceb_o,
    sram_tok2_web_o,
    sram_tok2_addr_o,
    sram_tok2_din_o,
    sram_tok2_q_i,
    norm1_stg_wr_do,
    norm1_stg_wr_flat,
    norm1_stg_wr_din,
    norm1_stg_rd_en,
    norm1_stg_rd_flat,
    norm1_stg_x,
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

parameter LN_RCP   = 65536 / EMBED_DIM ;
parameter TOK_FLAT = N_TOKENS * EMBED_DIM ;   // 10240

input                       clk ;
input                       reset ;
input                       start ;
input  signed [15:0]        x_i ;
input                       x_valid ;
input  signed [15:0]        wgt_i ;
input  signed [15:0]        bias_i ;
output [15:0]               wgt_addr_o ;
output                      busy ;
output                      x_ready ;
output reg                  done ;
output reg  signed [15:0]   y_o ;
output reg                  y_valid ;
output                      sram_tok2_ceb_o ;
output                      sram_tok2_web_o ;
output [13:0]               sram_tok2_addr_o ;
output [15:0]               sram_tok2_din_o ;
input  [15:0]               sram_tok2_q_i ;
output                      norm1_stg_wr_do ;
output [13:0]               norm1_stg_wr_flat ;
output [15:0]               norm1_stg_wr_din ;
output                      norm1_stg_rd_en ;
output [13:0]               norm1_stg_rd_flat ;
input  signed [15:0]        norm1_stg_x ;
output                      sram_q_ceb_o ;
output                      sram_q_web_o ;
output [13:0]               sram_q_addr_o ;
output [15:0]               sram_q_din_o ;
input  [15:0]               sram_q_q_i ;
output                      sram_k_ceb_o ;
output                      sram_k_web_o ;
output [13:0]               sram_k_addr_o ;
output [15:0]               sram_k_din_o ;
input  [15:0]               sram_k_q_i ;
output                      sram_v_ceb_o ;
output                      sram_v_web_o ;
output [13:0]               sram_v_addr_o ;
output [15:0]               sram_v_din_o ;
input  [15:0]               sram_v_q_i ;
output                      sram_qkm_ceb_o ;
output                      sram_qkm_web_o ;
output [13:0]               sram_qkm_addr_o ;
output [15:0]               sram_qkm_din_o ;
input  [15:0]               sram_qkm_q_i ;

parameter S_IDLE      = 4'd0 ;
parameter S_LOAD_X    = 4'd1 ;
parameter S_NORM1     = 4'd2 ;
parameter S_ATTN_FEED = 4'd3 ;
parameter S_ATTN_WAIT = 4'd4 ;
parameter S_RES1      = 4'd5 ;
parameter S_NORM2     = 4'd6 ;
parameter S_MLP_FEED  = 4'd7 ;
parameter S_MLP_WAIT  = 4'd8 ;
parameter S_RES2      = 4'd9 ;
parameter S_DONE      = 4'd10 ;

// ---- Activation SRAM control (macros in sglatrack_top) ----
reg                         st2_ceb ;
reg                         st2_web ;
reg [13:0]                  st2_addr ;
reg [15:0]                  st2_din ;
wire [15:0]                 st2_q ;

reg                         tq_ceb ;
reg                         tq_web ;
reg [13:0]                  tq_addr ;
reg [15:0]                  tq_din ;
wire [15:0]                 tq_q ;

wire                        ca_q_ceb ;
wire                        ca_q_web ;
wire [13:0]                 ca_q_addr ;
wire [15:0]                 ca_q_din ;

// 2-phase / multi-phase helpers (SRAM only)
reg                         x_norm_phase ;   // norm1/2 x read: 0=ADDR, 1=USE -> u_norm1
reg [1:0]                   res_subphase ;   // S_RES1/S_RES2: 0=RD, 1=FEED, 2=WR or OUT

reg [13:0]                  norm1_stg_wr_flat_lat ;
reg signed [15:0]           norm1_stg_wr_din_lat ;
reg                         norm1_stg_wr_do_lat ;

reg [13:0]                  tmp_wr_flat_lat ;
reg signed [15:0]           tmp_wr_din_lat ;
reg                         tmp_wr_do ;

reg [3:0]                   state, next_state ;

// Shared norm streaming regs (used by both S_NORM1 and S_NORM2)
reg [13:0]                  buf_addr ;
reg [8:0]                   tok_cnt ;
reg [4:0]                   rp_feat ;
reg                         rp_stream ;
reg                         ln1_start_r ;

reg                         ln1_start ;
wire                        ln1_busy, ln1_done ;
wire signed [15:0]          ln1_y_sat ;
wire [9:0]                  ln1_addr ;
wire                        ln1_out_beat ;

// Attention sub-block regs / wires
reg                         attn_start ;
wire signed [15:0]          attn_norm_x ;
wire signed [15:0]          attn_y ;
wire                        attn_yv ;
wire                        attn_busy, attn_done ;
wire [12:0]                 attn_wgt_addr ;

// MLP sub-block (norm2 read: norm_rd_* -> tmp-on-q / Sram_q, 2-phase)
reg                         mlp_start ;
wire                        mlp_norm_rd_en ;
wire [13:0]                 mlp_norm_rd_flat ;
wire signed [15:0]          mlp_norm_x ;
wire signed [15:0]          mlp_y ;
wire                        mlp_yv ;
wire                        mlp_busy, mlp_done ;
wire [15:0]                 mlp_wgt_addr ;

// Residual sub-block regs / wires
reg signed [15:0]           res_a, res_b ;
reg                         res_v ;
wire signed [15:0]          res_y ;
wire                        res_v_o ;

// Generic streaming pointers reused across capture and residual phases
reg [13:0]                  cap_ptr ;
reg [13:0]                  res_rp ;
reg [13:0]                  res_wp ;

wire [13:0]                 tmp_cap_flat ;
wire                        tb_q_mux_sel ;
wire                        in_norm_phase ;
wire [13:0]                 xbuf_rp_addr ;
wire signed [15:0]          xbuf_rp_data ;
wire                        x_norm_use ;
wire                        ln1_xv ;

assign tmp_cap_flat = tok_cnt * EMBED_DIM + {9'b0, ln1_addr[4:0]} ;

assign sram_tok2_ceb_o  = st2_ceb ;
assign sram_tok2_web_o  = st2_web ;
assign sram_tok2_addr_o = st2_addr ;
assign sram_tok2_din_o  = st2_din ;
assign st2_q            = sram_tok2_q_i ;

assign tq_q             = sram_q_q_i ;

assign mlp_norm_x       = tq_q ;
assign attn_norm_x      = norm1_stg_x ;

assign norm1_stg_wr_do   = norm1_stg_wr_do_lat ;
assign norm1_stg_wr_flat = norm1_stg_wr_flat_lat ;
assign norm1_stg_wr_din  = norm1_stg_wr_din_lat ;

assign tb_q_mux_sel =
    tmp_wr_do ||
    ((state == S_ATTN_WAIT) && attn_yv) ||
    ((state == S_MLP_WAIT) && mlp_yv) ||
    mlp_norm_rd_en ||
    ((state == S_RES1) && (res_subphase == 2'd0)) ||
    ((state == S_RES2) && (res_subphase == 2'd0) && (res_rp < TOK_FLAT[13:0])) ;

assign in_norm_phase  = (state == S_NORM1) || (state == S_NORM2) ;
assign xbuf_rp_addr  = tok_cnt * EMBED_DIM + {9'b0, rp_feat} ;
assign xbuf_rp_data  = st2_q ;
assign x_norm_use    = x_norm_phase ;
assign ln1_xv        = rp_stream && in_norm_phase && x_norm_use ;

assign sram_q_ceb_o  = tb_q_mux_sel ? tq_ceb  : ca_q_ceb ;
assign sram_q_web_o  = tb_q_mux_sel ? tq_web  : ca_q_web ;
assign sram_q_addr_o = tb_q_mux_sel ? tq_addr : ca_q_addr ;
assign sram_q_din_o  = tb_q_mux_sel ? tq_din  : ca_q_din ;

assign wgt_addr_o =
    (state == S_NORM1)                              ? {3'b000, 3'b0, ln1_addr[9:0]} :
    (state == S_NORM2)                              ? {3'b001, 3'b0, ln1_addr[9:0]} :
    (state == S_ATTN_FEED || state == S_ATTN_WAIT)  ? {3'b010, attn_wgt_addr}        :
    (state == S_MLP_FEED  || state == S_MLP_WAIT)   ? mlp_wgt_addr                   :
                                                       16'b0 ;

assign busy    = (state != S_IDLE) ;
assign x_ready = (state == S_LOAD_X) ;

layer_norm #(
    .FEAT_DIM  (EMBED_DIM),
    .RCP_NUM   (LN_RCP),
    .RCP_SHIFT (16)
) u_norm1 (
    .clk         (clk),
    .reset       (reset),
    .start       (ln1_start),
    .x_i         (xbuf_rp_data),
    .x_valid     (ln1_xv),
    .w_i         (wgt_i),
    .b_i         (bias_i),
    .feat_addr_o (ln1_addr),
    .busy        (ln1_busy),
    .done        (ln1_done),
    .y_o         (),
    .y_valid     (),
    .y_sat_o     (ln1_y_sat),
    .out_beat_o  (ln1_out_beat)
);

care_attention #(
    .EMBED_DIM (EMBED_DIM),
    .NUM_HEADS (4),
    .HEAD_DIM  (EMBED_DIM/4),
    .N_TOKENS  (N_TOKENS)
) u_attn (
    .clk              (clk),
    .reset            (reset),
    .start            (attn_start),
    .norm_rd_en       (norm1_stg_rd_en),
    .norm_rd_flat     (norm1_stg_rd_flat),
    .norm_x           (attn_norm_x),
    .wgt_i            (wgt_i),
    .bias_i           (bias_i),
    .wgt_addr_o       (attn_wgt_addr),
    .busy             (attn_busy),
    .done             (attn_done),
    .y_o              (attn_y),
    .y_valid          (attn_yv),
    .sram_q_ceb_o     (ca_q_ceb),
    .sram_q_web_o     (ca_q_web),
    .sram_q_addr_o    (ca_q_addr),
    .sram_q_din_o     (ca_q_din),
    .sram_q_q_i       (sram_q_q_i),
    .sram_k_ceb_o     (sram_k_ceb_o),
    .sram_k_web_o     (sram_k_web_o),
    .sram_k_addr_o    (sram_k_addr_o),
    .sram_k_din_o     (sram_k_din_o),
    .sram_k_q_i       (sram_k_q_i),
    .sram_v_ceb_o     (sram_v_ceb_o),
    .sram_v_web_o     (sram_v_web_o),
    .sram_v_addr_o    (sram_v_addr_o),
    .sram_v_din_o     (sram_v_din_o),
    .sram_v_q_i       (sram_v_q_i),
    .sram_qkm_ceb_o   (sram_qkm_ceb_o),
    .sram_qkm_web_o   (sram_qkm_web_o),
    .sram_qkm_addr_o  (sram_qkm_addr_o),
    .sram_qkm_din_o   (sram_qkm_din_o),
    .sram_qkm_q_i     (sram_qkm_q_i)
);

mlp #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (MLP_DIM),
    .N_TOKENS  (N_TOKENS)
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

residual #(
    .WIDTH (16)
) u_res (
    .clk   (clk),
    .reset (reset),
    .a_i   (res_a),
    .b_i   (res_b),
    .v_i   (res_v),
    .y_o   (res_y),
    .v_o   (res_v_o)
);

// SRAM port mux: x_buf; tmp-on-q merged with care q at sram_q_*
always @(*) begin
    st2_ceb  = 1'b1 ;
    st2_web  = 1'b1 ;
    st2_addr = 14'd0 ;
    st2_din  = 16'd0 ;

    tq_ceb  = 1'b1 ;
    tq_web  = 1'b1 ;
    tq_addr = 14'd0 ;
    tq_din  = 16'd0 ;

    // x: load write
    if (state == S_LOAD_X && x_valid && (buf_addr < TOK_FLAT[13:0])) begin
        st2_ceb  = 1'b0 ;
        st2_web  = 1'b0 ;
        st2_addr = buf_addr ;
        st2_din  = x_i[15:0] ;
    end
    // x: norm read (ADDR + USE phases — keep CEB=0 so gate-level SRAM Q stays valid)
    else if (in_norm_phase && rp_stream) begin
        st2_ceb  = 1'b0 ;
        st2_web  = 1'b1 ;
        st2_addr = xbuf_rp_addr ;
    end
    // x: S_RES1 read / write (never same cycle)
    else if (state == S_RES1) begin
        if (res_subphase == 2'd0) begin
            st2_ceb  = 1'b0 ;
            st2_web  = 1'b1 ;
            st2_addr = res_rp ;
        end else if ((res_subphase == 2'd2) && res_v_o && (res_wp < TOK_FLAT[13:0])) begin
            st2_ceb  = 1'b0 ;
            st2_web  = 1'b0 ;
            st2_addr = res_wp ;
            st2_din  = res_y[15:0] ;
        end
    end
    // x: S_RES2 read (ADDR phase)
    else if ((state == S_RES2) && (res_subphase == 2'd0) && (res_rp < TOK_FLAT[13:0])) begin
        st2_ceb  = 1'b0 ;
        st2_web  = 1'b1 ;
        st2_addr = res_rp ;
    end

    // tmp-on-q: norm2 capture write (latched 1 cycle after out_beat_o)
    if (tmp_wr_do) begin
        tq_ceb  = 1'b0 ;
        tq_web  = 1'b0 ;
        tq_addr = tmp_wr_flat_lat ;
        tq_din  = tmp_wr_din_lat[15:0] ;
    end
    // tmp-on-q: mlp FC1 norm2 read (ADDR); USE next cycle via tq_q
    else if (mlp_norm_rd_en) begin
        tq_ceb  = 1'b0 ;
        tq_web  = 1'b1 ;
        tq_addr = mlp_norm_rd_flat ;
    end
    // tmp-on-q: attn/mlp capture write
    else if ((state == S_ATTN_WAIT) && attn_yv) begin
        tq_ceb  = 1'b0 ;
        tq_web  = 1'b0 ;
        tq_addr = cap_ptr ;
        tq_din  = attn_y[15:0] ;
    end else if ((state == S_MLP_WAIT) && mlp_yv) begin
        tq_ceb  = 1'b0 ;
        tq_web  = 1'b0 ;
        tq_addr = cap_ptr ;
        tq_din  = mlp_y[15:0] ;
    end
    // tmp-on-q: residual read
    else if (((state == S_RES1) && (res_subphase == 2'd0)) ||
             ((state == S_RES2) && (res_subphase == 2'd0) && (res_rp < TOK_FLAT[13:0]))) begin
        tq_ceb  = 1'b0 ;
        tq_web  = 1'b1 ;
        tq_addr = res_rp ;
    end
end

// norm1 -> tok1 staging latch (keyed by S_NORM1 beat)
always @(posedge clk) begin
    if (reset) begin
        norm1_stg_wr_flat_lat <= 14'd0 ;
        norm1_stg_wr_din_lat  <= 16'd0 ;
        norm1_stg_wr_do_lat   <= 1'b0 ;
    end else begin
        norm1_stg_wr_do_lat <= 1'b0 ;
        if (state == S_NORM1 && ln1_out_beat) begin
            norm1_stg_wr_flat_lat <= tmp_cap_flat ;
            norm1_stg_wr_din_lat  <= ln1_y_sat ;
            norm1_stg_wr_do_lat   <= 1'b1 ;
        end
    end
end

// norm2 -> tmp-on-q latch (separate LHS domain; avoid mixed writes)
always @(posedge clk) begin
    if (reset) begin
        tmp_wr_flat_lat       <= 14'd0 ;
        tmp_wr_din_lat        <= 16'd0 ;
        tmp_wr_do             <= 1'b0 ;
    end else begin
        tmp_wr_do <= 1'b0 ;
        if (state == S_NORM2 && ln1_out_beat) begin
            tmp_wr_flat_lat <= tmp_cap_flat ;
            tmp_wr_din_lat  <= ln1_y_sat ;
            tmp_wr_do       <= 1'b1 ;
        end
    end
end

// ln1_start_r: delayed copy for norm replay; never use ln1_start in same-cycle
// conditions (gate netlist may infer async-CN / Q->clear feedback).
always @(posedge clk) begin
    if (reset)
        ln1_start_r <= 1'b0 ;
    else
        ln1_start_r <= ln1_start ;
end

// Sub-block start pulses (sync reset only, separate from datapath always).
always @(posedge clk) begin
    if (reset) begin
        ln1_start  <= 1'b0 ;
        attn_start <= 1'b0 ;
        mlp_start  <= 1'b0 ;
    end else begin
        ln1_start  <= 1'b0 ;
        attn_start <= 1'b0 ;
        mlp_start  <= 1'b0 ;
        case (state)
            S_NORM1: begin
                if (ln1_done && (tok_cnt < N_TOKENS))
                    ln1_start <= 1'b1 ;
                else if (!ln1_busy && !rp_stream && (tok_cnt == 9'd0) && !ln1_start_r)
                    ln1_start <= 1'b1 ;
            end
            S_NORM2: begin
                if (ln1_done && (tok_cnt < N_TOKENS))
                    ln1_start <= 1'b1 ;
                else if (!ln1_busy && !rp_stream && (tok_cnt == 9'd0) && !ln1_start_r)
                    ln1_start <= 1'b1 ;
            end
            S_ATTN_FEED: attn_start <= 1'b1 ;
            S_MLP_FEED:  mlp_start  <= 1'b1 ;
            default: ;
        endcase
    end
end

// FSM state
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE ;
    else
        state <= next_state ;
end

// FSM next_state
always @(*) begin
    next_state = state ;
    case (state)
        S_IDLE:      next_state = start ? S_LOAD_X : S_IDLE ;

        S_LOAD_X:    next_state = ((buf_addr == TOK_FLAT[13:0]) ||
                                   (buf_addr == TOK_FLAT[13:0] - 14'd1 && x_valid))
                                   ? S_NORM1 : S_LOAD_X ;

        S_NORM1:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_ATTN_FEED : S_NORM1 ;
        S_NORM2:     next_state = (ln1_done && tok_cnt == N_TOKENS[8:0]) ? S_MLP_FEED  : S_NORM2 ;

        S_ATTN_FEED: next_state = S_ATTN_WAIT ;
        S_ATTN_WAIT: next_state = attn_done ? S_RES1 : S_ATTN_WAIT ;

        S_RES1:      next_state = (res_wp == TOK_FLAT[13:0]) ? S_NORM2 : S_RES1 ;

        S_MLP_FEED:  next_state = S_MLP_WAIT ;
        S_MLP_WAIT:  next_state = mlp_done ? S_RES2 : S_MLP_WAIT ;

        S_RES2:      next_state = (res_wp == TOK_FLAT[13:0]) ? S_DONE : S_RES2 ;

        S_DONE:      next_state = S_IDLE ;
        default:     next_state = S_IDLE ;
    endcase
end

// done, y_valid, residual datapath (starts: see dedicated always above)
always @(posedge clk) begin
    done       <= 1'b0 ;
    y_valid    <= 1'b0 ;
    res_v      <= 1'b0 ;

    if (reset) begin
        buf_addr     <= 14'd0 ;
        tok_cnt      <= 9'd0 ;
        rp_feat      <= 5'd0 ;
        rp_stream    <= 1'b0 ;
        cap_ptr      <= 14'd0 ;
        res_rp       <= 14'd0 ;
        res_wp       <= 14'd0 ;
        res_a        <= 16'sd0 ;
        res_b        <= 16'sd0 ;
        y_o          <= 16'sd0 ;
        x_norm_phase <= 1'b0 ;
        res_subphase <= 2'd0 ;
    end else begin
        case (state)
            S_IDLE: begin
                buf_addr     <= 14'd0 ;
                tok_cnt      <= 9'd0 ;
                rp_feat      <= 5'd0 ;
                rp_stream    <= 1'b0 ;
                cap_ptr      <= 14'd0 ;
                res_rp       <= 14'd0 ;
                res_wp       <= 14'd0 ;
                x_norm_phase <= 1'b0 ;
                res_subphase <= 2'd0 ;
            end

            S_LOAD_X: begin
                if (x_valid && (buf_addr < TOK_FLAT[13:0]))
                    buf_addr <= buf_addr + 14'd1 ;
            end

            S_NORM1: begin
                // Priority kept from old code: ln1_start_r assignments override rp_stream path.
                if (ln1_start_r) begin
                    rp_stream    <= 1'b1 ;
                    rp_feat      <= 5'd0 ;
                    x_norm_phase <= 1'b0 ;
                end else if (rp_stream) begin
                    if (x_norm_phase == 1'b0)
                        x_norm_phase <= 1'b1 ;
                    else begin
                        x_norm_phase <= 1'b0 ;
                        if (rp_feat == EMBED_DIM-1) begin
                            rp_stream <= 1'b0 ;
                            rp_feat   <= 5'd0 ;
                        end else
                            rp_feat <= rp_feat + 5'd1 ;
                    end
                end

                if (ln1_done && tok_cnt == N_TOKENS) begin
                    cap_ptr <= 14'd0 ;
                    tok_cnt <= 9'd0 ;
                    rp_feat <= 5'd0 ;
                end
                else if (ln1_out_beat && ln1_addr == (EMBED_DIM - 1))
                    tok_cnt <= tok_cnt + 9'd1 ;
            end

            S_ATTN_FEED: begin
                cap_ptr <= 14'd0 ;
            end

            S_ATTN_WAIT: begin
                if (attn_yv && (cap_ptr < TOK_FLAT[13:0] - 14'd1))
                    cap_ptr <= cap_ptr + 14'd1 ;
                if (attn_done) begin
                    res_rp       <= 14'd0 ;
                    res_wp       <= 14'd0 ;
                    cap_ptr      <= 14'd0 ;
                    res_subphase <= 2'd0 ;
                end
            end

            S_RES1: begin
                case (res_subphase)
                    2'd0: res_subphase <= 2'd1 ;
                    2'd1: begin
                        res_a  <= st2_q ;
                        res_b  <= tq_q ;
                        res_v  <= 1'b1 ;
                        res_subphase <= 2'd2 ;
                    end
                    2'd2: begin
                        if (res_v_o && (res_wp < TOK_FLAT[13:0])) begin
                            res_wp <= res_wp + 14'd1 ;
                            if (res_rp < TOK_FLAT[13:0] - 14'd1)
                                res_rp <= res_rp + 14'd1 ;
                            res_subphase <= 2'd0 ;
                        end
                    end
                    default: res_subphase <= 2'd0 ;
                endcase
                if (next_state == S_NORM2) begin
                    tok_cnt      <= 9'd0 ;
                    rp_feat      <= 5'd0 ;
                    rp_stream    <= 1'b0 ;
                    x_norm_phase <= 1'b0 ;
                end
            end

            S_NORM2: begin
                // Priority kept from old code: ln1_start_r assignments override rp_stream path.
                if (ln1_start_r) begin
                    rp_stream    <= 1'b1 ;
                    rp_feat      <= 5'd0 ;
                    x_norm_phase <= 1'b0 ;
                end else if (rp_stream) begin
                    if (x_norm_phase == 1'b0)
                        x_norm_phase <= 1'b1 ;
                    else begin
                        x_norm_phase <= 1'b0 ;
                        if (rp_feat == EMBED_DIM-1) begin
                            rp_stream <= 1'b0 ;
                            rp_feat   <= 5'd0 ;
                        end else
                            rp_feat <= rp_feat + 5'd1 ;
                    end
                end

                if (ln1_done && tok_cnt == N_TOKENS)
                    cap_ptr <= 14'd0 ;
                else if (ln1_out_beat && ln1_addr == (EMBED_DIM - 1))
                    tok_cnt <= tok_cnt + 9'd1 ;
            end

            S_MLP_FEED: begin
                cap_ptr <= 14'd0 ;
            end

            S_MLP_WAIT: begin
                if (mlp_yv && (cap_ptr < TOK_FLAT[13:0] - 14'd1))
                    cap_ptr <= cap_ptr + 14'd1 ;
                if (mlp_done) begin
                    res_rp       <= 14'd0 ;
                    res_wp       <= 14'd0 ;
                    cap_ptr      <= 14'd0 ;
                    res_subphase <= 2'd0 ;
                end
            end

            S_RES2: begin
                case (res_subphase)
                    2'd0: res_subphase <= 2'd1 ;
                    2'd1: begin
                        res_a  <= st2_q ;
                        res_b  <= tq_q ;
                        res_v  <= 1'b1 ;
                        res_subphase <= 2'd2 ;
                    end
                    2'd2: begin
                        if (res_v_o && (res_wp < TOK_FLAT[13:0])) begin
                            y_o     <= res_y ;
                            y_valid <= 1'b1 ;
                            res_wp  <= res_wp + 14'd1 ;
                            if (res_rp < TOK_FLAT[13:0] - 14'd1)
                                res_rp <= res_rp + 14'd1 ;
                            res_subphase <= 2'd0 ;
                        end
                    end
                    default: res_subphase <= 2'd0 ;
                endcase
            end

            S_DONE: begin
                done    <= 1'b1 ;
                tok_cnt <= 9'd0 ;
            end

            default: ;
        endcase
    end
end

endmodule
