// =============================================================================
// transformer_block.v  (verilog_backbone3)
// -----------------------------------------------------------------------------
// Pipeline:  S_LOAD_X -> S_NORM1 -> S_ATTN -> S_RES1
//         -> S_NORM2  -> S_MLP   -> S_RES2 -> S_DONE
//
// SRAM time-sharing (6 macros, no 7th Sram_ao):
//   tok1  : inter-block input/output, norm1 staging, ao (care+mlp), capture
//   tok2  : x_buf (residual identity path)
//   q     : care's Q/PROJ, norm2 output, mlp output capture
//   k/v/qkm : shared between care_attention and mlp_ws
//
// Memory contract: every SRAM macro uses CLK(clk) posedge registered-data
//   (addr@T -> Q@T+1). transformer_block's own read paths are aligned for it:
//     S_LOAD_X : tok1 read addr is combinational (load_rp this cycle) so the +1
//                Q latency lands load_data_r in step with the unchanged tok2 write.
//     S_NORM   : layer_norm's combinational x_rd_flat drives tok2 directly (no
//                registered hold) -> x_i returns in the S_LOAD capture cycle.
//     S_RES1/2 : residual reads are issued combinationally at res_sub==0 and
//                captured at res_sub==1; the result write is direct at res_sub==3
//                (posedge writes are clean under CLK(clk), freeing sub0 for reads).
//   Sub-module reads (care/mlp/layer_norm internals) are just muxed through; each
//   submodule already expects the CLK(clk) macro latency.
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

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    input  wire signed [15:0] norm_wgt_i,
    input  wire signed [15:0] norm_bias_i,
    output wire [15:0] wgt_addr_o,
    output wire [7:0]  bias_addr_o,

    output wire        busy,
    output reg         done,

    output reg         sram_tok1_ceb_o,
    output reg         sram_tok1_web_o,
    output reg  [13:0] sram_tok1_addr_o,
    output reg  [15:0] sram_tok1_din_o,
    input  wire [15:0] sram_tok1_q_i,

    output reg         sram_tok2_ceb_o,
    output reg         sram_tok2_web_o,
    output reg  [13:0] sram_tok2_addr_o,
    output reg  [15:0] sram_tok2_din_o,
    input  wire [15:0] sram_tok2_q_i,

    output reg         sram_q_ceb_o,
    output reg         sram_q_web_o,
    output reg  [13:0] sram_q_addr_o,
    output reg  [15:0] sram_q_din_o,
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

// =========================================================================
// Parameters
// =========================================================================
parameter TOK_FLAT = N_TOKENS * EMBED_DIM;
parameter LN_RCP  = 65536 / EMBED_DIM;

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

// =========================================================================
// FSM
// =========================================================================
reg [3:0] state, next_state;

// =========================================================================
// Shared counters / pointers
// =========================================================================
reg [13:0] load_rp;
reg [13:0] load_wp;
reg        load_started;
reg [15:0] load_data_r;
reg        loadx_rd_en;
reg [13:0] loadx_rd_addr;
reg        loadx_wr_en;
reg [13:0] loadx_wr_addr;

reg [8:0]  tok_cnt;
reg        ln_start;
reg        ln_start_r;
reg [13:0] ln_cap_flat;

reg        attn_start;
reg [13:0] cap_ptr;

reg        mlp_start;
reg [13:0] mlp_cap_ptr;

reg [1:0]  res_sub;
reg [13:0] res_rp;
reg [13:0] res_wp;
reg signed [15:0] res_a, res_b;
reg        res_v;

// =========================================================================
// Sub-module wires: layer_norm_pip
// =========================================================================
wire        ln_x_rd_en;
wire [13:0] ln_x_rd_flat;
wire        ln_x_rd_pend;
wire        ln_x_rd_wait;
wire        ln_sram_rd;
wire [9:0]  ln_feat_addr;
wire        ln_busy, ln_done;
wire signed [15:0] ln_y_o;
wire        ln_y_valid;

// =========================================================================
// Sub-module wires: care_attention
// =========================================================================
wire        ca_norm_rd_en;
wire [13:0] ca_norm_rd_flat;
wire [12:0] ca_wgt_addr;
wire [7:0]  ca_bias_addr;
wire        ca_busy, ca_done;
wire signed [15:0] ca_y_o;
wire        ca_y_valid;
wire [6:0]  ca_y_neu;
wire [8:0]  ca_px_tok;

wire        ca_q_ceb, ca_q_web;
wire [13:0] ca_q_addr;
wire [15:0] ca_q_din;

wire        ca_k_ceb, ca_k_web;
wire [13:0] ca_k_addr;
wire [15:0] ca_k_din;

wire        ca_v_ceb, ca_v_web;
wire [13:0] ca_v_addr;
wire [15:0] ca_v_din;

wire        ca_qkm_ceb, ca_qkm_web;
wire [13:0] ca_qkm_addr;
wire [15:0] ca_qkm_din;

wire        ca_ao_ceb, ca_ao_web;
wire [13:0] ca_ao_addr;
wire [15:0] ca_ao_din;

// =========================================================================
// Sub-module wires: mlp_ws
// =========================================================================
wire        mlp_norm_rd_en;
wire [13:0] mlp_norm_rd_flat;
wire [15:0] mlp_wgt_addr;
wire [7:0]  mlp_bias_addr;
wire        mlp_busy, mlp_done;
wire signed [15:0] mlp_y_o;
wire        mlp_y_valid;

wire        mlp_k_ceb, mlp_k_web;
wire [13:0] mlp_k_addr;
wire [15:0] mlp_k_din;

wire        mlp_v_ceb, mlp_v_web;
wire [13:0] mlp_v_addr;
wire [15:0] mlp_v_din;

wire        mlp_qkm_ceb, mlp_qkm_web;
wire [13:0] mlp_qkm_addr;
wire [15:0] mlp_qkm_din;

wire        mlp_ao_ceb, mlp_ao_web;
wire [13:0] mlp_ao_addr;
wire [15:0] mlp_ao_din;

// =========================================================================
// Sub-module wires: residual
// =========================================================================
wire signed [15:0] res_y;
wire        res_v_o;

// CLK(clk) residual SRAM access (replaces the old negedge-safe registered holds):
//   read  : combinational addr=res_rp asserted in res_sub==0 -> macro Q@sub1 capture
//           (res_a/res_b are latched in res_sub==1, one cycle later).
//   write : direct addr=res_wp/din=res_y in res_sub==3 (posedge write is clean under
//           CLK(clk); no defer needed, which also frees sub0 for the read port).
wire res_rd_active = (state == S_RES1 || state == S_RES2) &&
                     (res_sub == 2'd0) && (res_rp < TOK_FLAT[13:0]);
wire res_wr_active = (state == S_RES1 || state == S_RES2) && (res_sub == 2'd3);

// =========================================================================
// Convenience
// =========================================================================
wire in_attn = (state == S_ATTN_FEED) || (state == S_ATTN_WAIT);
wire in_mlp  = (state == S_MLP_FEED)  || (state == S_MLP_WAIT);

assign busy = (state != S_IDLE);

// =========================================================================
// ROM address mux
// =========================================================================
assign wgt_addr_o =
    (state == S_NORM1)  ? {3'b000, 3'b0, ln_feat_addr} :
    (state == S_NORM2)  ? {3'b001, 3'b0, ln_feat_addr} :
    in_attn             ? {3'b010, ca_wgt_addr}         :
    in_mlp              ? mlp_wgt_addr                  :
                           16'd0;

assign bias_addr_o =
    in_attn ? ca_bias_addr :
    in_mlp  ? mlp_bias_addr :
              8'd0;

// =========================================================================
// k / v / qkm mux (attn vs mlp, never simultaneous)
// =========================================================================
assign sram_k_ceb_o  = in_attn ? ca_k_ceb  : in_mlp ? mlp_k_ceb  : 1'b1;
assign sram_k_web_o  = in_attn ? ca_k_web  : in_mlp ? mlp_k_web  : 1'b1;
assign sram_k_addr_o = in_attn ? ca_k_addr : in_mlp ? mlp_k_addr : 14'd0;
assign sram_k_din_o  = in_attn ? ca_k_din  : in_mlp ? mlp_k_din  : 16'd0;

assign sram_v_ceb_o  = in_attn ? ca_v_ceb  : in_mlp ? mlp_v_ceb  : 1'b1;
assign sram_v_web_o  = in_attn ? ca_v_web  : in_mlp ? mlp_v_web  : 1'b1;
assign sram_v_addr_o = in_attn ? ca_v_addr : in_mlp ? mlp_v_addr : 14'd0;
assign sram_v_din_o  = in_attn ? ca_v_din  : in_mlp ? mlp_v_din  : 16'd0;

assign sram_qkm_ceb_o  = in_attn ? ca_qkm_ceb  : in_mlp ? mlp_qkm_ceb  : 1'b1;
assign sram_qkm_web_o  = in_attn ? ca_qkm_web  : in_mlp ? mlp_qkm_web  : 1'b1;
assign sram_qkm_addr_o = in_attn ? ca_qkm_addr : in_mlp ? mlp_qkm_addr : 14'd0;
assign sram_qkm_din_o  = in_attn ? ca_qkm_din  : in_mlp ? mlp_qkm_din  : 16'd0;

// =========================================================================
// Sub-module instances
// =========================================================================
layer_norm_pip #(
    .FEAT_DIM  (EMBED_DIM),
    .FEAT_AW   (5),
    .RCP_NUM   (LN_RCP)
) u_ln (
    .clk             (clk),
    .reset           (reset),
    .start           (ln_start),
    .token_base_flat ({tok_cnt, 5'b0}),
    .x_rd_en         (ln_x_rd_en),
    .x_rd_flat       (ln_x_rd_flat),
    .x_i             ($signed(sram_tok2_q_i)),
    .w_i             (norm_wgt_i),
    .b_i             (norm_bias_i),
    .feat_addr_o     (ln_feat_addr),
    .busy            (ln_busy),
    .done            (ln_done),
    .y_o             (ln_y_o),
    .y_valid         (ln_y_valid),
    .x_rd_pend_o     (ln_x_rd_pend),
    .x_rd_wait_o     (ln_x_rd_wait)
);

assign ln_sram_rd = ln_x_rd_en | ln_x_rd_pend | ln_x_rd_wait;

care_attention #(
    .EMBED_DIM (EMBED_DIM),
    .NUM_HEADS (4),
    .HEAD_DIM  (EMBED_DIM / 4),
    .N_TOKENS  (N_TOKENS)
) u_attn (
    .clk             (clk),
    .reset           (reset),
    .start           (attn_start),
    .norm_rd_en      (ca_norm_rd_en),
    .norm_rd_flat    (ca_norm_rd_flat),
    .norm_x          ($signed(sram_tok1_q_i)),
    .wgt_i           (wgt_i),
    .bias_i          (bias_i),
    .wgt_addr_o      (ca_wgt_addr),
    .bias_addr_o     (ca_bias_addr),
    .busy            (ca_busy),
    .done            (ca_done),
    .y_o             (ca_y_o),
    .y_valid         (ca_y_valid),
    .y_neu_o         (ca_y_neu),
    .px_tok_o        (ca_px_tok),
    .sram_q_ceb_o    (ca_q_ceb),
    .sram_q_web_o    (ca_q_web),
    .sram_q_addr_o   (ca_q_addr),
    .sram_q_din_o    (ca_q_din),
    .sram_q_q_i      (sram_q_q_i),
    .sram_k_ceb_o    (ca_k_ceb),
    .sram_k_web_o    (ca_k_web),
    .sram_k_addr_o   (ca_k_addr),
    .sram_k_din_o    (ca_k_din),
    .sram_k_q_i      (sram_k_q_i),
    .sram_v_ceb_o    (ca_v_ceb),
    .sram_v_web_o    (ca_v_web),
    .sram_v_addr_o   (ca_v_addr),
    .sram_v_din_o    (ca_v_din),
    .sram_v_q_i      (sram_v_q_i),
    .sram_qkm_ceb_o  (ca_qkm_ceb),
    .sram_qkm_web_o  (ca_qkm_web),
    .sram_qkm_addr_o (ca_qkm_addr),
    .sram_qkm_din_o  (ca_qkm_din),
    .sram_qkm_q_i    (sram_qkm_q_i),
    .sram_ao_ceb_o   (ca_ao_ceb),
    .sram_ao_web_o   (ca_ao_web),
    .sram_ao_addr_o  (ca_ao_addr),
    .sram_ao_din_o   (ca_ao_din),
    .sram_ao_q_i     (sram_tok1_q_i)
);

mlp_ws #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (MLP_DIM),
    .N_TOKENS  (N_TOKENS)
) u_mlp (
    .clk             (clk),
    .reset           (reset),
    .start           (mlp_start),
    .norm_rd_en      (mlp_norm_rd_en),
    .norm_rd_flat    (mlp_norm_rd_flat),
    .norm_x          ($signed(sram_q_q_i)),
    .wgt_i           (wgt_i),
    .bias_i          (bias_i),
    .wgt_addr_o      (mlp_wgt_addr),
    .bias_addr_o     (mlp_bias_addr),
    .sram_k_ceb_o    (mlp_k_ceb),
    .sram_k_web_o    (mlp_k_web),
    .sram_k_addr_o   (mlp_k_addr),
    .sram_k_din_o    (mlp_k_din),
    .sram_k_q_i      (sram_k_q_i),
    .sram_v_ceb_o    (mlp_v_ceb),
    .sram_v_web_o    (mlp_v_web),
    .sram_v_addr_o   (mlp_v_addr),
    .sram_v_din_o    (mlp_v_din),
    .sram_v_q_i      (sram_v_q_i),
    .sram_qkm_ceb_o  (mlp_qkm_ceb),
    .sram_qkm_web_o  (mlp_qkm_web),
    .sram_qkm_addr_o (mlp_qkm_addr),
    .sram_qkm_din_o  (mlp_qkm_din),
    .sram_qkm_q_i    (sram_qkm_q_i),
    .sram_ao_ceb_o   (mlp_ao_ceb),
    .sram_ao_web_o   (mlp_ao_web),
    .sram_ao_addr_o  (mlp_ao_addr),
    .sram_ao_din_o   (mlp_ao_din),
    .sram_ao_q_i     (sram_tok1_q_i),
    .busy            (mlp_busy),
    .done            (mlp_done),
    .y_o             (mlp_y_o),
    .y_valid         (mlp_y_valid)
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

// =========================================================================
// tok1 mux (combinational)
// =========================================================================
always @(*) begin
    sram_tok1_ceb_o  = 1'b1;
    sram_tok1_web_o  = 1'b1;
    sram_tok1_addr_o = 14'd0;
    sram_tok1_din_o  = 16'd0;

    case (state)
        S_LOAD_X: begin
            if (loadx_rd_en) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = 1'b1;
                sram_tok1_addr_o = loadx_rd_addr;
            end
        end

        S_NORM1: begin
            if (ln_y_valid) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = 1'b0;
                sram_tok1_addr_o = ln_cap_flat;
                sram_tok1_din_o  = ln_y_o;
            end
        end

        S_ATTN_FEED,
        S_ATTN_WAIT: begin
            if (!ca_ao_ceb) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = ca_ao_web;
                sram_tok1_addr_o = ca_ao_addr;
                sram_tok1_din_o  = ca_ao_din;
            end else if (ca_norm_rd_en) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = 1'b1;
                sram_tok1_addr_o = ca_norm_rd_flat;
            end else if (ca_y_valid) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = 1'b0;
                sram_tok1_addr_o = cap_ptr;
                sram_tok1_din_o  = ca_y_o;
            end
        end

        S_RES1: begin
            if (res_rd_active) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = 1'b1;
                sram_tok1_addr_o = res_rp;
            end
        end

        S_MLP_FEED,
        S_MLP_WAIT: begin
            if (!mlp_ao_ceb) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = mlp_ao_web;
                sram_tok1_addr_o = mlp_ao_addr;
                sram_tok1_din_o  = mlp_ao_din;
            end
        end

        S_RES2: begin
            if (res_wr_active) begin
                sram_tok1_ceb_o  = 1'b0;
                sram_tok1_web_o  = 1'b0;
                sram_tok1_addr_o = res_wp;
                sram_tok1_din_o  = res_y;
            end
        end

        default: ;
    endcase
end

// =========================================================================
// tok2 mux (combinational)
// =========================================================================
always @(*) begin
    sram_tok2_ceb_o  = 1'b1;
    sram_tok2_web_o  = 1'b1;
    sram_tok2_addr_o = 14'd0;
    sram_tok2_din_o  = 16'd0;

    case (state)
        S_LOAD_X: begin
            if (loadx_wr_en) begin
                sram_tok2_ceb_o  = 1'b0;
                sram_tok2_web_o  = 1'b0;
                sram_tok2_addr_o = loadx_wr_addr;
                sram_tok2_din_o  = load_data_r;
            end
        end

        S_NORM1,
        S_NORM2: begin
            if (ln_sram_rd) begin
                // CLK(clk): drive layer_norm's combinational x_rd_flat straight to
                // tok2 (no registered hold). x_rd_flat is stable across issue/pend/
                // wait, so the macro returns x_i one cycle later, matching layer_norm's
                // S_LOAD capture window. A registered hold would slip x one cycle late.
                sram_tok2_ceb_o  = 1'b0;
                sram_tok2_web_o  = 1'b1;
                sram_tok2_addr_o = ln_x_rd_flat;
            end
        end

        S_RES1: begin
            if (res_wr_active) begin
                sram_tok2_ceb_o  = 1'b0;
                sram_tok2_web_o  = 1'b0;
                sram_tok2_addr_o = res_wp;
                sram_tok2_din_o  = res_y;
            end else if (res_rd_active) begin
                sram_tok2_ceb_o  = 1'b0;
                sram_tok2_web_o  = 1'b1;
                sram_tok2_addr_o = res_rp;
            end
        end

        S_RES2: begin
            if (res_rd_active) begin
                sram_tok2_ceb_o  = 1'b0;
                sram_tok2_web_o  = 1'b1;
                sram_tok2_addr_o = res_rp;
            end
        end

        default: ;
    endcase
end

// =========================================================================
// sram_q mux (combinational)
// =========================================================================
always @(*) begin
    sram_q_ceb_o  = 1'b1;
    sram_q_web_o  = 1'b1;
    sram_q_addr_o = 14'd0;
    sram_q_din_o  = 16'd0;

    case (state)
        S_ATTN_FEED,
        S_ATTN_WAIT: begin
            sram_q_ceb_o  = ca_q_ceb;
            sram_q_web_o  = ca_q_web;
            sram_q_addr_o = ca_q_addr;
            sram_q_din_o  = ca_q_din;
        end

        S_NORM2: begin
            if (ln_y_valid) begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b0;
                sram_q_addr_o = ln_cap_flat;
                sram_q_din_o  = ln_y_o;
            end
        end

        S_MLP_FEED,
        S_MLP_WAIT: begin
            if (mlp_norm_rd_en) begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b1;
                sram_q_addr_o = mlp_norm_rd_flat;
            end else if (mlp_y_valid) begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b0;
                sram_q_addr_o = mlp_cap_ptr;
                sram_q_din_o  = mlp_y_o;
            end
        end

        S_RES2: begin
            if (res_rd_active) begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b1;
                sram_q_addr_o = res_rp;
            end
        end

        default: ;
    endcase
end

// =========================================================================
// FSM state register
// =========================================================================
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// =========================================================================
// FSM next-state logic
// =========================================================================
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:      if (start) next_state = S_LOAD_X;

        S_LOAD_X:    if (load_wp == TOK_FLAT[13:0])
                         next_state = S_NORM1;

        S_NORM1:     if (ln_done && tok_cnt == N_TOKENS[8:0] - 9'd1)
                         next_state = S_ATTN_FEED;

        S_ATTN_FEED: next_state = S_ATTN_WAIT;

        S_ATTN_WAIT: if (ca_done) next_state = S_RES1;

        S_RES1:      if (res_wp == TOK_FLAT[13:0])
                         next_state = S_NORM2;

        S_NORM2:     if (ln_done && tok_cnt == N_TOKENS[8:0] - 9'd1)
                         next_state = S_MLP_FEED;

        S_MLP_FEED:  next_state = S_MLP_WAIT;

        S_MLP_WAIT:  if (mlp_done) next_state = S_RES2;

        S_RES2:      if (res_wp == TOK_FLAT[13:0])
                         next_state = S_DONE;

        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

// =========================================================================
// ln_start delayed (prevent double-pulse)
// =========================================================================
always @(posedge clk) begin
    if (reset)     ln_start_r <= 1'b0;
    else           ln_start_r <= ln_start;
end

// =========================================================================
// Sub-block start pulses
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        ln_start   <= 1'b0;
        attn_start <= 1'b0;
        mlp_start  <= 1'b0;
    end else begin
        ln_start   <= 1'b0;
        attn_start <= 1'b0;
        mlp_start  <= 1'b0;

        case (state)
            S_NORM1,
            S_NORM2: begin
                if (ln_done && tok_cnt < N_TOKENS[8:0] - 9'd1)
                    ln_start <= 1'b1;
                else if (!ln_busy && !ln_start_r && tok_cnt < N_TOKENS[8:0])
                    ln_start <= 1'b1;
            end
            S_ATTN_FEED: attn_start <= 1'b1;
            S_MLP_FEED:  mlp_start  <= 1'b1;
            default: ;
        endcase
    end
end

// =========================================================================
// S_LOAD_X tok1 read (CLK(clk): combinational address, addr@T -> Q@T+1)
//   Present load_rp this cycle so load_data_r captures tok1[load_rp] one cycle
//   later, keeping the read/write pointer offset identical to the old ~clk model.
// =========================================================================
always @(*) begin
    if (state == S_LOAD_X && load_rp < TOK_FLAT[13:0]) begin
        loadx_rd_en   = 1'b1;
        loadx_rd_addr = load_rp;
    end else begin
        loadx_rd_en   = 1'b0;
        loadx_rd_addr = 14'd0;
    end
end

// tok2 write hold
always @(posedge clk) begin
    if (reset || state == S_IDLE)
        loadx_wr_en <= 1'b0;
    else if (state == S_LOAD_X && load_started &&
             load_wp < TOK_FLAT[13:0]) begin
        loadx_wr_en   <= 1'b1;
        loadx_wr_addr <= load_wp;
    end else
        loadx_wr_en <= 1'b0;
end

// =========================================================================
// Main datapath (registered)
// =========================================================================
always @(posedge clk) begin
    done  <= 1'b0;
    res_v <= 1'b0;

    if (reset) begin
        load_rp     <= 14'd0;
        load_wp     <= 14'd0;
        load_started <= 1'b0;
        load_data_r <= 16'd0;
        tok_cnt     <= 9'd0;
        ln_cap_flat <= 14'd0;
        cap_ptr     <= 14'd0;
        mlp_cap_ptr <= 14'd0;
        res_sub     <= 2'd0;
        res_rp      <= 14'd0;
        res_wp      <= 14'd0;
        res_a       <= 16'sd0;
        res_b       <= 16'sd0;
    end else begin
        case (state)
            // ---------------------------------------------------------
            S_IDLE: begin
                load_rp     <= 14'd0;
                load_wp     <= 14'd0;
                load_started <= 1'b0;
                tok_cnt     <= 9'd0;
                ln_cap_flat <= 14'd0;
                cap_ptr     <= 14'd0;
                mlp_cap_ptr <= 14'd0;
                res_sub     <= 2'd0;
                res_rp      <= 14'd0;
                res_wp      <= 14'd0;
            end

            // ---------------------------------------------------------
            S_LOAD_X: begin
                load_data_r <= sram_tok1_q_i;
                if (load_rp < TOK_FLAT[13:0])
                    load_rp <= load_rp + 14'd1;
                if (load_started && load_wp < TOK_FLAT[13:0])
                    load_wp <= load_wp + 14'd1;
                load_started <= 1'b1;
            end

            // ---------------------------------------------------------
            S_NORM1: begin
                if (ln_y_valid)
                    ln_cap_flat <= ln_cap_flat + 14'd1;

                if (ln_start)
                    ln_cap_flat <= {tok_cnt, 5'b0};

                if (ln_done) begin
                    if (tok_cnt < N_TOKENS[8:0] - 9'd1)
                        tok_cnt <= tok_cnt + 9'd1;
                    else
                        tok_cnt <= tok_cnt + 9'd1;
                end

                if (next_state == S_ATTN_FEED) begin
                    cap_ptr <= 14'd0;
                    tok_cnt <= 9'd0;
                end
            end

            // ---------------------------------------------------------
            S_ATTN_FEED: begin
                cap_ptr <= 14'd0;
            end

            // ---------------------------------------------------------
            S_ATTN_WAIT: begin
                if (ca_y_valid)
                    cap_ptr <= cap_ptr + 14'd1;

                if (ca_done) begin
                    res_rp  <= 14'd0;
                    res_wp  <= 14'd0;
                    res_sub <= 2'd0;
                end
            end

            // ---------------------------------------------------------
            S_RES1: begin
                case (res_sub)
                    2'd0: res_sub <= 2'd1;
                    // sub=1: capture reads issued at sub0 (CLK(clk): Q valid now)
                    2'd1: begin
                        res_a <= $signed(sram_tok2_q_i);
                        res_b <= $signed(sram_tok1_q_i);
                        res_v <= 1'b1;
                        res_sub <= 2'd2;
                    end
                    // sub=2: wait residual 1-cycle latency (res_v_o)
                    2'd2: begin
                        if (res_v_o && res_wp < TOK_FLAT[13:0])
                            res_sub <= 2'd3;
                    end
                    // sub=3: res_wr_active drives the direct tok2 write; advance ptrs
                    2'd3: begin
                        res_wp <= res_wp + 14'd1;
                        if (res_rp < TOK_FLAT[13:0] - 14'd1)
                            res_rp <= res_rp + 14'd1;
                        res_sub <= 2'd0;
                    end
                endcase

                if (next_state == S_NORM2) begin
                    tok_cnt     <= 9'd0;
                    ln_cap_flat <= 14'd0;
                end
            end

            // ---------------------------------------------------------
            S_NORM2: begin
                if (ln_y_valid)
                    ln_cap_flat <= ln_cap_flat + 14'd1;

                if (ln_start)
                    ln_cap_flat <= {tok_cnt, 5'b0};

                if (ln_done) begin
                    tok_cnt <= tok_cnt + 9'd1;
                end

                if (next_state == S_MLP_FEED) begin
                    mlp_cap_ptr <= 14'd0;
                end
            end

            // ---------------------------------------------------------
            S_MLP_FEED: begin
                mlp_cap_ptr <= 14'd0;
            end

            // ---------------------------------------------------------
            S_MLP_WAIT: begin
                if (mlp_y_valid)
                    mlp_cap_ptr <= mlp_cap_ptr + 14'd1;

                if (mlp_done) begin
                    res_rp  <= 14'd0;
                    res_wp  <= 14'd0;
                    res_sub <= 2'd0;
                end
            end

            // ---------------------------------------------------------
            S_RES2: begin
                case (res_sub)
                    2'd0: res_sub <= 2'd1;
                    // sub=1: capture reads issued at sub0 (CLK(clk): Q valid now)
                    2'd1: begin
                        res_a <= $signed(sram_tok2_q_i);
                        res_b <= $signed(sram_q_q_i);
                        res_v <= 1'b1;
                        res_sub <= 2'd2;
                    end
                    // sub=2: wait residual 1-cycle latency (res_v_o)
                    2'd2: begin
                        if (res_v_o && res_wp < TOK_FLAT[13:0])
                            res_sub <= 2'd3;
                    end
                    // sub=3: res_wr_active drives the direct tok1 write; advance ptrs
                    2'd3: begin
                        res_wp <= res_wp + 14'd1;
                        if (res_rp < TOK_FLAT[13:0] - 14'd1)
                            res_rp <= res_rp + 14'd1;
                        res_sub <= 2'd0;
                    end
                endcase
            end

            // ---------------------------------------------------------
            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

endmodule
