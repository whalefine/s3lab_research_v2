// =============================================================================
// backbone_top.v  (verilog_backbone3)
//
// SGLATrack Backbone Top，對齊 run_backbone_numpy_shared_trunk.main()。
// 執行 blocks 0~START_LAYER（固定層）+ sel_block_i 指定 block（6~11）+ backbone norm。
//
// 與 verilog_backbone2 差異：
//   - transformer_block 無 x_i/y_o 串流；S_LOAD_X 自 Sram_tok1 讀入
//   - block 0 前由本模組 S_LOAD_IN 將外部 merged token 寫入 Sram_tok1
//   - backbone norm 使用 layer_norm_pip + tok1 就地讀寫（同 block norm 協定）
//   - bias ROM 位址取自 transformer_block.bias_addr_o
//
// Golden: Activation/backbone_after_norm_backbone_out_bi.txt
// Plan B: norm 結果留在 Sram_tok1；y_o/y_valid 不輸出（head 直接讀 tok1）
// =============================================================================

module backbone_top #(
    parameter EMBED_DIM   = 32,
    parameter N_TOKENS    = 320,
    parameter START_LAYER = 5,
    parameter N_BLOCKS    = 12
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire [3:0]  sel_block_i,
    input  wire signed [15:0] x_i,
    input  wire        x_valid,
    output wire        busy,
    output wire        x_ready,
    output reg         done,
    output wire signed [15:0] y_o,
    output wire        y_valid,
    output wire        sram_tok1_ceb_o,
    output wire        sram_tok1_web_o,
    output wire [13:0] sram_tok1_addr_o,
    output wire [15:0] sram_tok1_din_o,
    input  wire [15:0] sram_tok1_q_i,
    output wire        sram_tok2_ceb_o,
    output wire        sram_tok2_web_o,
    output wire [13:0] sram_tok2_addr_o,
    output wire [15:0] sram_tok2_din_o,
    input  wire [15:0] sram_tok2_q_i,
    output wire        sram_q_ceb_o,
    output wire        sram_q_web_o,
    output wire [13:0] sram_q_addr_o,
    output wire [15:0] sram_q_din_o,
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

localparam TOK_FLAT = N_TOKENS * EMBED_DIM;
localparam LN_RCP   = 65536 / EMBED_DIM;

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
parameter S_IDLE          = 4'd0;
parameter S_LOAD_IN       = 4'd1;
parameter S_RUN_FIXED     = 4'd2;
parameter S_RUN_SELECTED  = 4'd3;
parameter S_BACKBONE_NORM = 4'd4;
parameter S_DONE          = 4'd5;

reg [3:0] state, next_state;
reg [3:0] block_idx;
reg [3:0] sel_block_r;

reg [13:0] load_wr_ptr;
reg        loadin_wr_en;
reg [13:0] loadin_wr_addr;
reg [15:0] loadin_wr_din;

reg        bt_s1_ceb;
reg        bt_s1_web;
reg [13:0] bt_s1_addr;
reg [15:0] bt_s1_din;

reg        tb_start;

reg                bn_start;
reg                bn_start_r;
reg [8:0]          bn_tok_cnt;
reg [13:0]         bn_cap_flat;

reg                bn_wr_en;
reg [13:0]         bn_wr_addr;
reg [15:0]         bn_wr_din;

wire [15:0] s1_q;

wire tb_busy, tb_done;
wire [15:0] tb_wgt_addr;
wire [7:0]  tb_bias_addr;

// Forward declare before u_tb (assigns are below ROM section)
wire signed [15:0] tb_norm_wgt_mux;
wire signed [15:0] tb_norm_bias_mux;
wire signed [15:0] tb_am_wgt_mux;
wire signed [15:0] tb_am_bias_mux;

wire        tb_tok1_ceb, tb_tok1_web;
wire [13:0] tb_tok1_addr;
wire [15:0] tb_tok1_din;

// ---------------------------------------------------------------------------
// transformer_block
// ---------------------------------------------------------------------------
transformer_block #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (4 * EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_tb (
    .clk              (clk),
    .reset            (reset),
    .start            (tb_start),
    .wgt_i            (tb_am_wgt_mux),
    .bias_i           (tb_am_bias_mux),
    .norm_wgt_i       (tb_norm_wgt_mux),
    .norm_bias_i      (tb_norm_bias_mux),
    .wgt_addr_o       (tb_wgt_addr),
    .bias_addr_o      (tb_bias_addr),
    .busy             (tb_busy),
    .done             (tb_done),
    .sram_tok1_ceb_o  (tb_tok1_ceb),
    .sram_tok1_web_o  (tb_tok1_web),
    .sram_tok1_addr_o (tb_tok1_addr),
    .sram_tok1_din_o  (tb_tok1_din),
    .sram_tok1_q_i    (s1_q),
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

// ---------------------------------------------------------------------------
// Backbone final layer_norm_pip (in-place read/write on Sram_tok1)
// Golden: Activation/backbone_after_norm_backbone_out_bi.txt
// ---------------------------------------------------------------------------
wire signed [15:0] bn_wgt_mux, bn_bias_mux;
wire [9:0]         bn_feat_addr;

wire               bn_x_rd_en;
wire [13:0]        bn_x_rd_flat;
wire               bn_x_rd_pend;
wire               bn_x_rd_wait;
wire               bn_sram_rd;

wire               bn_busy, bn_done;
wire signed [15:0] bn_y_o;
wire               bn_y_valid;

assign bn_sram_rd = bn_x_rd_en | bn_x_rd_pend | bn_x_rd_wait;

layer_norm_pip #(
    .FEAT_DIM (EMBED_DIM),
    .FEAT_AW  (5),
    .RCP_NUM  (LN_RCP)
) u_bn (
    .clk             (clk),
    .reset           (reset),
    .start           (bn_start),
    .token_base_flat ({bn_tok_cnt, 5'b0}),
    .x_rd_en         (bn_x_rd_en),
    .x_rd_flat       (bn_x_rd_flat),
    .x_i             ($signed(s1_q)),
    .w_i             (bn_wgt_mux),
    .b_i             (bn_bias_mux),
    .feat_addr_o     (bn_feat_addr),
    .busy            (bn_busy),
    .done            (bn_done),
    .y_o             (bn_y_o),
    .y_valid         (bn_y_valid),
    .x_rd_pend_o     (bn_x_rd_pend),
    .x_rd_wait_o     (bn_x_rd_wait)
);

assign sram_tok1_ceb_o  = bt_s1_ceb;
assign sram_tok1_web_o  = bt_s1_web;
assign sram_tok1_addr_o = bt_s1_addr;
assign sram_tok1_din_o  = bt_s1_din;
assign s1_q             = sram_tok1_q_i;

// ---------------------------------------------------------------------------
// ROM Q wires
// ---------------------------------------------------------------------------
wire signed [15:0] q_norm1_w, q_norm1_b;
wire signed [15:0] q_norm2_w, q_norm2_b;
wire signed [15:0] q_qkv_b;
wire signed [15:0] q_qkv_w_03, q_qkv_w_46;
wire signed [15:0] q_proj_w,   q_proj_b;
wire signed [15:0] q_fc1_b;
wire signed [15:0] q_fc1_w_03, q_fc1_w_46;
wire signed [15:0] q_fc2_b;
wire signed [15:0] q_fc2_w_03, q_fc2_w_46;
wire signed [15:0] q_bnorm_w,  q_bnorm_b;

// ---------------------------------------------------------------------------
// ROM address computation
// ---------------------------------------------------------------------------
wire [2:0]  wtype       = tb_wgt_addr[15:13];
wire [12:0] local_addr  = tb_wgt_addr[12:0];

wire [7:0] blk_x32   = ({4'b0, block_idx} << 5);
wire [7:0] addr_n1w  = blk_x32 + {3'b0, local_addr[4:0]};
wire [7:0] addr_n1b  = addr_n1w;
wire [7:0] addr_n2w  = addr_n1w;
wire [7:0] addr_n2b  = addr_n1w;

wire [7:0] addr_projb = blk_x32 + {3'b0, tb_bias_addr[4:0]};

wire [9:0] blk_x96  = ({6'b0, block_idx} << 6) + ({7'b0, block_idx} << 5);
wire [9:0] addr_qkvb = blk_x96 + {3'b0, tb_bias_addr[6:0]};

wire [9:0] blk_x128  = ({3'b0, block_idx} << 7);
wire [9:0] addr_fc1b = blk_x128 + {3'b0, tb_bias_addr[6:0]};

wire [7:0] addr_fc2b = blk_x32 + {3'b0, tb_bias_addr[4:0]};

wire [12:0] blk_x1024 = ({3'b0, block_idx} << 10);
wire [12:0] addr_projw = blk_x1024 + {3'b0, local_addr[9:0]};

wire [13:0] blk_x3072_03 = ({4'b0, block_idx[1:0]} << 11) +
                            ({5'b0, block_idx[1:0]} << 10);
wire [13:0] addr_qkvw03  = blk_x3072_03 + {1'b0, local_addr[12:0]};

wire [13:0] blk_x4096_03 = {block_idx[1:0], 12'b0};
wire [13:0] addr_fc1w03  = blk_x4096_03 + {1'b0, local_addr[12:0]};
wire [13:0] addr_fc2w03  = blk_x4096_03 + {1'b0, local_addr[12:0]};

wire [1:0]  blk_off46    = block_idx[1:0];
wire [13:0] blk_x3072_46 = ({4'b0, blk_off46} << 11) + ({5'b0, blk_off46} << 10);
wire [13:0] addr_qkvw46  = blk_x3072_46 + {1'b0, local_addr[12:0]};
wire [13:0] blk_x4096_46 = {blk_off46, 12'b0};
wire [13:0] addr_fc1w46  = blk_x4096_46 + {1'b0, local_addr[12:0]};
wire [13:0] addr_fc2w46  = blk_x4096_46 + {1'b0, local_addr[12:0]};

wire [6:0] addr_bnorm = {2'b00, bn_feat_addr[4:0]};

// ---------------------------------------------------------------------------
// ROM chip-enables (active low)
// ---------------------------------------------------------------------------
wire bb_active = (state == S_RUN_FIXED || state == S_RUN_SELECTED);
wire bn_active = (state == S_BACKBONE_NORM);
wire blk_lo    = (block_idx <= 4'd3);
wire blk_hi    = (block_idx >= 4'd4);

wire ceb_n1w    = !(bb_active && wtype == 3'b000);
wire ceb_n1b    = !(bb_active && wtype == 3'b000);
wire ceb_n2w    = !(bb_active && wtype == 3'b001);
wire ceb_n2b    = !(bb_active && wtype == 3'b001);
wire ceb_qkvb   = !(bb_active && wtype == 3'b010);
wire ceb_projw  = !(bb_active && wtype == 3'b010);
wire ceb_projb  = !(bb_active && wtype == 3'b010);
wire ceb_fc1b   = !(bb_active && wtype == 3'b100);
wire ceb_fc2b   = !(bb_active && wtype == 3'b101);
wire ceb_qkvw03 = !(bb_active && wtype == 3'b010 && blk_lo);
wire ceb_qkvw46 = !(bb_active && wtype == 3'b010 && blk_hi);
wire ceb_fc1w03 = !(bb_active && wtype == 3'b100 && blk_lo);
wire ceb_fc1w46 = !(bb_active && wtype == 3'b100 && blk_hi);
wire ceb_fc2w03 = !(bb_active && wtype == 3'b101 && blk_lo);
wire ceb_fc2w46 = !(bb_active && wtype == 3'b101 && blk_hi);
wire ceb_bnw    = !bn_active;
wire ceb_bnb    = !bn_active;

// ---------------------------------------------------------------------------
// Weight / bias mux -> transformer_block
//   norm: dedicated bus (no FC1/QKV mux) for STA + layer_norm_pip 3-phase S_NORM
//   attn/mlp: separate bus
// ---------------------------------------------------------------------------
wire is_qkv  = (wtype == 3'b010) && (local_addr < 13'd3072);
wire is_proj = (wtype == 3'b010) && (local_addr >= 13'd3072);

assign tb_norm_wgt_mux =
    (wtype == 3'b000) ? q_norm1_w :
    (wtype == 3'b001) ? q_norm2_w :
    16'sd0;

assign tb_norm_bias_mux =
    (wtype == 3'b000) ? q_norm1_b :
    (wtype == 3'b001) ? q_norm2_b :
    16'sd0;

assign tb_am_wgt_mux =
    (is_qkv)          ? (blk_lo ? q_qkv_w_03 : q_qkv_w_46) :
    (is_proj)         ? q_proj_w :
    (wtype == 3'b100) ? (blk_lo ? q_fc1_w_03 : q_fc1_w_46) :
    (wtype == 3'b101) ? (blk_lo ? q_fc2_w_03 : q_fc2_w_46) :
    16'sd0;

assign tb_am_bias_mux =
    (is_qkv)          ? q_qkv_b :
    (is_proj)         ? q_proj_b :
    (wtype == 3'b100) ? q_fc1_b :
    (wtype == 3'b101) ? q_fc2_b :
    16'sd0;

assign bn_wgt_mux  = q_bnorm_w;
assign bn_bias_mux = q_bnorm_b;

// ---------------------------------------------------------------------------
// Block weight ROM instantiations (CLK = clk; posedge registered-data,
//   combinational address addr@T -> Q@T+1, matching the unit-test ROM models
//   that transformer_block / care_attention / mlp_ws were verified against).
// ---------------------------------------------------------------------------
rom_backbone_blocks_0_6_norm1_weight u_rom_n1w (
    .A(addr_n1w), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n1w),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm1_w));

rom_backbone_blocks_0_6_norm1_bias u_rom_n1b (
    .A(addr_n1b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n1b),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm1_b));

rom_backbone_blocks_0_6_norm2_weight u_rom_n2w (
    .A(addr_n2w), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n2w),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm2_w));

rom_backbone_blocks_0_6_norm2_bias u_rom_n2b (
    .A(addr_n2b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n2b),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm2_b));

rom_backbone_blocks_0_6_attn_qkv_bias u_rom_qkvb (
    .A(addr_qkvb), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_qkvb),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_qkv_b));

rom_backbone_blocks_0_3_attn_qkv_weight u_rom_qkvw03 (
    .A(addr_qkvw03), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_qkvw03),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_qkv_w_03));

rom_backbone_blocks_4_6_attn_qkv_weight u_rom_qkvw46 (
    .A(addr_qkvw46), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_qkvw46),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_qkv_w_46));

rom_backbone_blocks_0_6_attn_proj_weight u_rom_projw (
    .A(addr_projw), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_projw),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_proj_w));

rom_backbone_blocks_0_6_attn_proj_bias u_rom_projb (
    .A(addr_projb), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_projb),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_proj_b));

rom_backbone_blocks_0_6_mlp_fc1_bias u_rom_fc1b (
    .A(addr_fc1b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc1b),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc1_b));

rom_backbone_blocks_0_3_mlp_fc1_weight u_rom_fc1w03 (
    .A(addr_fc1w03), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc1w03),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc1_w_03));

rom_backbone_blocks_4_6_mlp_fc1_weight u_rom_fc1w46 (
    .A(addr_fc1w46), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc1w46),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc1_w_46));

rom_backbone_blocks_0_6_mlp_fc2_bias u_rom_fc2b (
    .A(addr_fc2b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc2b),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc2_b));

rom_backbone_blocks_0_3_mlp_fc2_weight u_rom_fc2w03 (
    .A(addr_fc2w03), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc2w03),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc2_w_03));

rom_backbone_blocks_4_6_mlp_fc2_weight u_rom_fc2w46 (
    .A(addr_fc2w46), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc2w46),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc2_w_46));

rom_backbone_norm_weight u_rom_bnw (
    .A(addr_bnorm), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_bnw),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_bnorm_w));

rom_backbone_norm_bias u_rom_bnb (
    .A(addr_bnorm), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_bnb),
    .CLK(clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_bnorm_b));

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:
            next_state = start ? S_LOAD_IN : S_IDLE;

        S_LOAD_IN:
            next_state = (load_wr_ptr == TOK_FLAT[13:0]) ? S_RUN_FIXED : S_LOAD_IN;

        S_RUN_FIXED:
            next_state = (tb_done && block_idx == START_LAYER)
                         ? S_RUN_SELECTED : S_RUN_FIXED;

        S_RUN_SELECTED:
            next_state = tb_done ? S_BACKBONE_NORM : S_RUN_SELECTED;

        S_BACKBONE_NORM:
            next_state = (bn_done && (bn_tok_cnt == N_TOKENS[8:0] - 9'd1))
                         ? S_DONE : S_BACKBONE_NORM;

        S_DONE:
            next_state = S_IDLE;

        default:
            next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: datapath control
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE)
        done <= 1'b1;
    else
        done <= 1'b0;
end

always @(posedge clk) begin
    if (reset)
        tb_start <= 1'b0;
    else begin
        tb_start <= 1'b0;
        if ((state == S_RUN_FIXED || state == S_RUN_SELECTED) && !tb_busy)
            tb_start <= 1'b1;
    end
end

always @(posedge clk) begin
    if (reset)
        load_wr_ptr <= 14'd0;
    else if (state == S_IDLE)
        load_wr_ptr <= 14'd0;
    else if (state == S_LOAD_IN && x_valid &&
             (load_wr_ptr < TOK_FLAT[13:0]))
        load_wr_ptr <= load_wr_ptr + 14'd1;
end

// S_LOAD_IN registered write hold (negedge-safe for real SRAM)
always @(posedge clk) begin
    if (reset || state == S_IDLE) begin
        loadin_wr_en <= 1'b0;
    end else if (state == S_LOAD_IN && x_valid &&
                 (load_wr_ptr < TOK_FLAT[13:0])) begin
        loadin_wr_en   <= 1'b1;
        loadin_wr_addr <= load_wr_ptr;
        loadin_wr_din  <= x_i;
    end else begin
        loadin_wr_en <= 1'b0;
    end
end

// bn_start delayed (prevent double-pulse; same as transformer_block u_ln)
always @(posedge clk) begin
    if (reset)
        bn_start_r <= 1'b0;
    else
        bn_start_r <= bn_start;
end

// tok1 registered write hold (deferred 1 cycle; posedge write is clean under CLK(clk))
always @(posedge clk) begin
    if (reset || state != S_BACKBONE_NORM)
        bn_wr_en <= 1'b0;
    else if (bn_y_valid) begin
        bn_wr_en   <= 1'b1;
        bn_wr_addr <= bn_cap_flat;
        bn_wr_din  <= bn_y_o;
    end else
        bn_wr_en <= 1'b0;
end

// u_bn start + per-token cap_flat (mirror transformer_block S_NORM1)
always @(posedge clk) begin
    if (reset) begin
        bn_tok_cnt  <= 9'd0;
        bn_cap_flat <= 14'd0;
        bn_start    <= 1'b0;
    end else if (state == S_IDLE) begin
        bn_tok_cnt  <= 9'd0;
        bn_cap_flat <= 14'd0;
        bn_start    <= 1'b0;
    end else begin
        bn_start <= 1'b0;

        if (state == S_BACKBONE_NORM) begin
            if (bn_done && (bn_tok_cnt < N_TOKENS[8:0] - 9'd1))
                bn_start <= 1'b1;
            else if (!bn_busy && !bn_start_r && (bn_tok_cnt < N_TOKENS[8:0]))
                bn_start <= 1'b1;

            if (bn_y_valid)
                bn_cap_flat <= bn_cap_flat + 14'd1;
            if (bn_start)
                bn_cap_flat <= {bn_tok_cnt, 5'b0};

            if (bn_done && (bn_tok_cnt < N_TOKENS[8:0] - 9'd1))
                bn_tok_cnt <= bn_tok_cnt + 9'd1;
        end
    end
end

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
                if (tb_done && (block_idx < START_LAYER))
                    block_idx <= block_idx + 4'd1;
            end

            S_RUN_SELECTED: begin
                block_idx <= sel_block_r;
            end

            default: ;
        endcase
    end
end

// Sram_tok1 mux: backbone norm (priority) / tb / external load
always @(*) begin
    bt_s1_ceb  = 1'b1;
    bt_s1_web  = 1'b1;
    bt_s1_addr = 14'd0;
    bt_s1_din  = 16'd0;

    if (state == S_BACKBONE_NORM) begin
        if (bn_wr_en) begin
            bt_s1_ceb  = 1'b0;
            bt_s1_web  = 1'b0;
            bt_s1_addr = bn_wr_addr;
            bt_s1_din  = bn_wr_din;
        end else if (bn_sram_rd) begin
            // CLK(clk): drive layer_norm's combinational x_rd_flat straight to tok1
            // (no registered hold). x_rd_flat is stable across issue/pend/wait, so the
            // posedge macro returns x_i in the S_LOAD capture cycle. A registered hold
            // would slip x one cycle late (same fix as transformer_block S_NORM).
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

assign y_o     = 16'sd0;
assign y_valid = 1'b0;
assign busy    = (state != S_IDLE);
assign x_ready = (state == S_LOAD_IN) &&
                 (load_wr_ptr < TOK_FLAT[13:0]);

endmodule
