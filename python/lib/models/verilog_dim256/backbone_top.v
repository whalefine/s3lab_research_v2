// =============================================================================
// backbone_top.v  (verilog_dim256_backbone)
//
// SGLATrack Backbone Top for dim256 (EMBED_DIM=256, N_TOKENS=320).
// Bit-accurate mirror of run_backbone_numpy_dim256_q88.main() backbone stage:
//   blocks 0..START_LAYER=5  (fixed)
//   sel_block_i (golden = 6) (adaptive block selection)
//   backbone final layer_norm.
//
// Weights DO NOT come from rom_*.v instances in this top. Instead the parent
// TB muxes ROM data from $readmemb arrays based on block_idx_o + address.
// This module is memory-model-free and portable across TB and gate-level flows.
//
// FSM:
//   S_IDLE           -> S_LOAD_IN (on start)
//   S_LOAD_IN        -> S_RUN_FIXED (after N_TOKENS*EMBED_DIM tokens loaded)
//   S_RUN_FIXED      -> S_RUN_SELECTED (after block_idx reaches START_LAYER)
//   S_RUN_SELECTED   -> S_BACKBONE_NORM (after transformer_block done)
//   S_BACKBONE_NORM  -> S_DONE (after all N_TOKENS normalized)
//   S_DONE           -> S_IDLE
//
// Plan B I/O: Backbone result stays in Sram_tok1 for the head to read.
// y_o/y_valid are tied off (present for legacy port compat only).
//
// =============================================================================
// Address encoding (this module + TEST_backbone must agree):
// -----------------------------------------------------------------------------
//   block_idx_o[3:0]  : current transformer block index (0..N_BLOCKS-1).
//                       Selects which block's weight table the TB muxes.
//                       Undefined / stale outside S_RUN_FIXED / S_RUN_SELECTED.
//
//   wgt_addr_o[19:0]  : attention or MLP weight *local* address (per block).
//     bit[19]     = module_type  (0 = attention,   1 = MLP)
//     For attention (bit[19]=0):
//       bit[18]         = 0 (reserved)
//       bits[17:0]      = care_attention.wgt_addr_o passthrough
//                         range 0..196607     -> QKV weight,
//                                                 index = neuron[9:0]*256 + feat[7:0]
//                         range 196608..262143 -> PROJ weight,
//                                                 index = (addr - 196608)
//                                                 = neuron[7:0]*256 + feat[7:0]
//     For MLP (bit[19]=1):
//       bit[18]         = layer  (0 = FC1, 1 = FC2)
//       bits[17:0]      = mlp_ws local index (neuron * IN_DIM + feat)
//                         FC1: neuron[9:0]*256 + feat[7:0], max 262143
//                         FC2: neuron[7:0]*1024 + feat[9:0], max 262143
//
//   bias_addr_o[9:0]  : attention or MLP bias local address (per block).
//     Attention QKV : bits[9:0] = neuron (0..767)
//     Attention PROJ: bits[7:0] = neuron (0..255), bits[9:8] = 0
//     MLP FC1       : bits[9:0] = neuron (0..1023)
//     MLP FC2       : bits[7:0] = neuron (0..255), bits[9:8] = 0
//     TB disambiguates by inspecting wgt_addr_o[19:18] plus (for attn) the
//     QKV/PROJ range of wgt_addr_o[17:0].
//
//   norm_feat_addr_o[9:0] : block-level LayerNorm gamma/beta address.
//     bit[9]     = norm_layer  (0 = norm1, 1 = norm2)
//     bit[8]     = 0 (reserved)
//     bits[7:0]  = feat_index  (0..255)
//     Used with block_idx_o to pick the (block, norm1/norm2, feat) triple.
//
//   bn_feat_addr_o[9:0] : *backbone* final LayerNorm gamma/beta address.
//     bits[9:8]  = 0 (reserved)
//     bits[7:0]  = feat_index (0..255)
//     Independent of block_idx_o (backbone_norm has one shared weight set).
//
// ROM read latency (contract with TB):
//   posedge T   : DUT drives wgt_addr_o / bias_addr_o / norm_feat_addr_o /
//                 bn_feat_addr_o
//   posedge T+1 : TB delivers wgt_i / bias_i / norm_wgt_i / norm_bias_i /
//                 bn_wgt_i / bn_bias_i (posedge-registered ROM Q).
//   This matches layer_norm_pip 3-phase NORM and care_attention/mlp_ws 2-phase
//   ROM load, so no additional retiming is needed.
//
// SRAM ports (17-bit addr) match care_attention.v / mlp_ws.v (dim256):
//   tok1, tok2, q, k, v, qkm : 16-bit wide, single-port, CLK = clk.
//
// Golden: memory/Activation/backbone_after_norm_backbone_out_bi.txt (81920).
// =============================================================================

module backbone_top #(
    parameter EMBED_DIM   = 256,
    parameter N_TOKENS    = 320,
    parameter START_LAYER = 5,
    parameter N_BLOCKS    = 7
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire [3:0]  sel_block_i,

    // External streaming input (S_LOAD_IN)
    input  wire signed [15:0] x_i,
    input  wire        x_valid,
    output wire        x_ready,

    // Status
    output wire        busy,
    output reg         done,

    // Optional streaming output (Plan B: tied off; head reads Sram_tok1)
    output wire signed [15:0] y_o,
    output wire        y_valid,

    // SRAM ports (single-port, CLK = clk, posedge registered-data)
    output wire        sram_tok1_ceb_o,
    output wire        sram_tok1_web_o,
    output wire [16:0] sram_tok1_addr_o,
    output wire [15:0] sram_tok1_din_o,
    input  wire [15:0] sram_tok1_q_i,

    output wire        sram_tok2_ceb_o,
    output wire        sram_tok2_web_o,
    output wire [16:0] sram_tok2_addr_o,
    output wire [15:0] sram_tok2_din_o,
    input  wire [15:0] sram_tok2_q_i,

    output wire        sram_q_ceb_o,
    output wire        sram_q_web_o,
    output wire [16:0] sram_q_addr_o,
    output wire [15:0] sram_q_din_o,
    input  wire [15:0] sram_q_q_i,

    output wire        sram_k_ceb_o,
    output wire        sram_k_web_o,
    output wire [16:0] sram_k_addr_o,
    output wire [15:0] sram_k_din_o,
    input  wire [15:0] sram_k_q_i,

    output wire        sram_v_ceb_o,
    output wire        sram_v_web_o,
    output wire [16:0] sram_v_addr_o,
    output wire [15:0] sram_v_din_o,
    input  wire [15:0] sram_v_q_i,

    output wire        sram_qkm_ceb_o,
    output wire        sram_qkm_web_o,
    output wire [16:0] sram_qkm_addr_o,
    output wire [15:0] sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i,

    // Weight interface to TB (block_idx + addresses; data returned 1 cycle later)
    output wire [3:0]  block_idx_o,
    output wire [19:0] wgt_addr_o,
    output wire [9:0]  bias_addr_o,
    output wire [9:0]  norm_feat_addr_o,
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    input  wire signed [15:0] norm_wgt_i,
    input  wire signed [15:0] norm_bias_i,

    // Backbone final norm weight interface (single block-independent weight set)
    output wire [9:0]  bn_feat_addr_o,
    input  wire signed [15:0] bn_wgt_i,
    input  wire signed [15:0] bn_bias_i
);

// =========================================================================
// Local params
// =========================================================================
localparam TOK_FLAT     = N_TOKENS * EMBED_DIM;      // 81920 for dim256
localparam TOK_FLAT_M1  = TOK_FLAT - 1;              // 81919
localparam LN_RCP       = 65536 / EMBED_DIM;         // 256 for FEAT_DIM=256
localparam N_TOKENS_M1  = N_TOKENS - 1;              // 319

// FSM state encoding (parameter, not localparam, per verilog_rule.mdc)
parameter S_IDLE          = 4'd0;
parameter S_LOAD_IN       = 4'd1;
parameter S_RUN_FIXED     = 4'd2;
parameter S_RUN_SELECTED  = 4'd3;
parameter S_BACKBONE_NORM = 4'd4;
parameter S_DONE          = 4'd5;

// =========================================================================
// Registers (all declared at module level; no reg inside always block)
// =========================================================================
reg [3:0]  state, next_state;
reg [3:0]  block_idx;
reg [3:0]  sel_block_r;

// S_LOAD_IN: streaming write pointer and registered write latch
reg [16:0] load_wr_ptr;
reg        loadin_wr_en;
reg [16:0] loadin_wr_addr;
reg [15:0] loadin_wr_din;

// transformer_block start pulse
reg        tb_start;

// Backbone final norm control
reg        bn_start;
reg        bn_start_r;
reg [8:0]  bn_tok_cnt;
reg [16:0] bn_cap_flat;

reg        bn_wr_en;
reg [16:0] bn_wr_addr;
reg [15:0] bn_wr_din;

// Sram_tok1 combinational mux drivers
reg        bt_s1_ceb;
reg        bt_s1_web;
reg [16:0] bt_s1_addr;
reg [15:0] bt_s1_din;

// =========================================================================
// Transformer block wires
// =========================================================================
wire        tb_busy, tb_done;
wire [19:0] tb_wgt_addr;
wire [9:0]  tb_bias_addr;
wire [9:0]  tb_norm_feat_addr;

wire        tb_tok1_ceb, tb_tok1_web;
wire [16:0] tb_tok1_addr;
wire [15:0] tb_tok1_din;

// =========================================================================
// Backbone final layer_norm_pip wires
// =========================================================================
wire [9:0]         bn_feat_addr;
wire               bn_x_rd_en;
wire [16:0]        bn_x_rd_flat;
wire               bn_x_rd_pend;
wire               bn_x_rd_wait;
wire               bn_sram_rd;
wire               bn_busy, bn_done;
wire signed [15:0] bn_y_o;
wire               bn_y_valid;

assign bn_sram_rd = bn_x_rd_en | bn_x_rd_pend | bn_x_rd_wait;

// =========================================================================
// Transformer block instantiation (dim256)
//   Assumes the (to-be-updated) transformer_block.v uses:
//     wgt_addr_o [19:0]        - attn / mlp weight local addr (see header)
//     bias_addr_o [9:0]        - attn / mlp bias local addr
//     norm_feat_addr_o [9:0]   - norm1 / norm2 feat addr (bit[9] = norm_layer)
//     SRAM ports [16:0]
//   These wider ports are required for EMBED_DIM=256 addressing.
// =========================================================================
transformer_block #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (4 * EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_tb (
    .clk              (clk),
    .reset            (reset),
    .start            (tb_start),
    .wgt_i            (wgt_i),
    .bias_i           (bias_i),
    .norm_wgt_i       (norm_wgt_i),
    .norm_bias_i      (norm_bias_i),
    .wgt_addr_o       (tb_wgt_addr),
    .bias_addr_o      (tb_bias_addr),
    .norm_feat_addr_o (tb_norm_feat_addr),
    .busy             (tb_busy),
    .done             (tb_done),
    .sram_tok1_ceb_o  (tb_tok1_ceb),
    .sram_tok1_web_o  (tb_tok1_web),
    .sram_tok1_addr_o (tb_tok1_addr),
    .sram_tok1_din_o  (tb_tok1_din),
    .sram_tok1_q_i    (sram_tok1_q_i),
    .sram_tok2_ceb_o  (sram_tok2_ceb_o),
    .sram_tok2_web_o  (sram_tok2_web_o),
    .sram_tok2_addr_o (sram_tok2_addr_o),
    .sram_tok2_din_o  (sram_tok2_din_o),
    .sram_tok2_q_i    (sram_tok2_q_i),
    .sram_q_ceb_o     (sram_q_ceb_o),
    .sram_q_web_o     (sram_q_web_o),
    .sram_q_addr_o    (sram_q_addr_o),
    .sram_q_din_o     (sram_q_din_o),
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

// =========================================================================
// Backbone final layer_norm_pip (in-place read/write on Sram_tok1)
// Golden: Activation/backbone_after_norm_backbone_out_bi.txt
// =========================================================================
layer_norm_pip #(
    .FEAT_DIM  (EMBED_DIM),
    .FEAT_AW   (8),
    .RCP_SHIFT (16),
    .RCP_NUM   (LN_RCP)
) u_bn (
    .clk             (clk),
    .reset           (reset),
    .start           (bn_start),
    .token_base_flat ({bn_tok_cnt, 8'b0}),
    .x_rd_en         (bn_x_rd_en),
    .x_rd_flat       (bn_x_rd_flat),
    .x_i             ($signed(sram_tok1_q_i)),
    .w_i             (bn_wgt_i),
    .b_i             (bn_bias_i),
    .feat_addr_o     (bn_feat_addr),
    .busy            (bn_busy),
    .done            (bn_done),
    .y_o             (bn_y_o),
    .y_valid         (bn_y_valid),
    .x_rd_pend_o     (bn_x_rd_pend),
    .x_rd_wait_o     (bn_x_rd_wait)
);

// =========================================================================
// External weight/address routing (this module -> TB combinational muxes)
// =========================================================================
assign block_idx_o      = block_idx;
assign wgt_addr_o       = tb_wgt_addr;
assign bias_addr_o      = tb_bias_addr;
assign norm_feat_addr_o = tb_norm_feat_addr;
assign bn_feat_addr_o   = bn_feat_addr;

// =========================================================================
// Sram_tok1 driver assignments (muxed below)
// =========================================================================
assign sram_tok1_ceb_o  = bt_s1_ceb;
assign sram_tok1_web_o  = bt_s1_web;
assign sram_tok1_addr_o = bt_s1_addr;
assign sram_tok1_din_o  = bt_s1_din;

// =========================================================================
// FSM segment 1: state register
// =========================================================================
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// =========================================================================
// FSM segment 2: next-state logic (default = hold state -> no latch)
// =========================================================================
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:
            if (start)
                next_state = S_LOAD_IN;
        S_LOAD_IN:
            if (load_wr_ptr == TOK_FLAT[16:0])
                next_state = S_RUN_FIXED;
        S_RUN_FIXED:
            if (tb_done && (block_idx == START_LAYER[3:0]))
                next_state = S_RUN_SELECTED;
        S_RUN_SELECTED:
            if (tb_done)
                next_state = S_BACKBONE_NORM;
        S_BACKBONE_NORM:
            if (bn_done && (bn_tok_cnt == N_TOKENS_M1[8:0]))
                next_state = S_DONE;
        S_DONE:
            next_state = S_IDLE;
        default:
            next_state = S_IDLE;
    endcase
end

// =========================================================================
// done registered output (pulse in S_DONE)
// =========================================================================
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE)
        done <= 1'b1;
    else
        done <= 1'b0;
end

// =========================================================================
// transformer_block start pulse
//   Fires one cycle after entering RUN_FIXED / RUN_SELECTED with !tb_busy.
//   Registered default 0 guarantees single-cycle pulse (see verilog_rule.mdc
//   pulse template).
// =========================================================================
always @(posedge clk) begin
    if (reset)
        tb_start <= 1'b0;
    else begin
        tb_start <= 1'b0;
        if ((state == S_RUN_FIXED || state == S_RUN_SELECTED) && !tb_busy)
            tb_start <= 1'b1;
    end
end

// =========================================================================
// S_LOAD_IN: streaming input pointer
// =========================================================================
always @(posedge clk) begin
    if (reset)
        load_wr_ptr <= 17'd0;
    else if (state == S_IDLE)
        load_wr_ptr <= 17'd0;
    else if (state == S_LOAD_IN && x_valid &&
             (load_wr_ptr < TOK_FLAT[16:0]))
        load_wr_ptr <= load_wr_ptr + 17'd1;
end

// =========================================================================
// S_LOAD_IN: registered write latch (posedge-clean tok1 write)
// =========================================================================
always @(posedge clk) begin
    if (reset || state == S_IDLE) begin
        loadin_wr_en <= 1'b0;
    end else if (state == S_LOAD_IN && x_valid &&
                 (load_wr_ptr < TOK_FLAT[16:0])) begin
        loadin_wr_en   <= 1'b1;
        loadin_wr_addr <= load_wr_ptr;
        loadin_wr_din  <= x_i;
    end else begin
        loadin_wr_en <= 1'b0;
    end
end

// =========================================================================
// bn_start delayed one cycle (prevents double-pulse; mirrors u_ln pattern)
// =========================================================================
always @(posedge clk) begin
    if (reset)
        bn_start_r <= 1'b0;
    else
        bn_start_r <= bn_start;
end

// =========================================================================
// Backbone norm registered write latch (writes bn_y_o back into Sram_tok1)
// =========================================================================
always @(posedge clk) begin
    if (reset || state != S_BACKBONE_NORM) begin
        bn_wr_en <= 1'b0;
    end else if (bn_y_valid) begin
        bn_wr_en   <= 1'b1;
        bn_wr_addr <= bn_cap_flat;
        bn_wr_din  <= bn_y_o;
    end else begin
        bn_wr_en <= 1'b0;
    end
end

// =========================================================================
// Backbone norm control (tok counter, cap_flat, bn_start pulse)
//   Mirrors transformer_block.S_NORM1: kick-start the first token, then chain
//   subsequent tokens on bn_done. cap_flat is set on bn_start, and increments
//   on each valid output beat.
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        bn_tok_cnt  <= 9'd0;
        bn_cap_flat <= 17'd0;
        bn_start    <= 1'b0;
    end else if (state == S_IDLE) begin
        bn_tok_cnt  <= 9'd0;
        bn_cap_flat <= 17'd0;
        bn_start    <= 1'b0;
    end else begin
        bn_start <= 1'b0;

        if (state == S_BACKBONE_NORM) begin
            if (bn_done && (bn_tok_cnt < N_TOKENS_M1[8:0]))
                bn_start <= 1'b1;
            else if (!bn_busy && !bn_start_r && (bn_tok_cnt < N_TOKENS[8:0]))
                bn_start <= 1'b1;

            if (bn_y_valid)
                bn_cap_flat <= bn_cap_flat + 17'd1;
            if (bn_start)
                bn_cap_flat <= {bn_tok_cnt, 8'b0};

            if (bn_done && (bn_tok_cnt < N_TOKENS_M1[8:0]))
                bn_tok_cnt <= bn_tok_cnt + 9'd1;
        end
    end
end

// =========================================================================
// block_idx tracker + sel_block_r latch
// =========================================================================
always @(posedge clk) begin
    if (reset) begin
        block_idx   <= 4'd0;
        sel_block_r <= 4'd0;
    end else begin
        case (state)
            S_IDLE: begin
                block_idx <= 4'd0;
                if (start)
                    sel_block_r <= sel_block_i;
            end

            S_RUN_FIXED: begin
                if (tb_done && (block_idx < START_LAYER[3:0]))
                    block_idx <= block_idx + 4'd1;
            end

            S_RUN_SELECTED: begin
                block_idx <= sel_block_r;
            end

            default: ;
        endcase
    end
end

// =========================================================================
// Sram_tok1 combinational mux
//   Priority (per verilog_rule.mdc 8.4 - documented order):
//     1) S_BACKBONE_NORM writeback (bn_wr_en)
//     2) S_BACKBONE_NORM read (bn_sram_rd) - drive layer_norm_pip x_rd_flat
//        directly (no registered hold; x_rd_flat is stable across issue/pend/
//        wait so the CLK(clk) macro returns x_i in the S_LOAD capture cycle)
//     3) transformer_block owns tok1 while busy
//     4) S_LOAD_IN write latch
//   Idle default: CEB=1, WEB=1 (disabled read).
// =========================================================================
always @(*) begin
    bt_s1_ceb  = 1'b1;
    bt_s1_web  = 1'b1;
    bt_s1_addr = 17'd0;
    bt_s1_din  = 16'd0;

    if (state == S_BACKBONE_NORM) begin
        if (bn_wr_en) begin
            bt_s1_ceb  = 1'b0;
            bt_s1_web  = 1'b0;
            bt_s1_addr = bn_wr_addr;
            bt_s1_din  = bn_wr_din;
        end else if (bn_sram_rd) begin
            bt_s1_ceb  = 1'b0;
            bt_s1_web  = 1'b1;
            bt_s1_addr = bn_x_rd_flat;
        end
    end else if (tb_busy) begin
        bt_s1_ceb  = tb_tok1_ceb;
        bt_s1_web  = tb_tok1_web;
        bt_s1_addr = tb_tok1_addr;
        bt_s1_din  = tb_tok1_din;
    end else if (loadin_wr_en) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b0;
        bt_s1_addr = loadin_wr_addr;
        bt_s1_din  = loadin_wr_din;
    end
end

// =========================================================================
// Top-level status / streaming output
//   Plan B: y_o / y_valid tied off; downstream head reads Sram_tok1 directly.
// =========================================================================
assign y_o     = 16'sd0;
assign y_valid = 1'b0;
assign busy    = (state != S_IDLE);
assign x_ready = (state == S_LOAD_IN) &&
                 (load_wr_ptr < TOK_FLAT[16:0]);

endmodule
