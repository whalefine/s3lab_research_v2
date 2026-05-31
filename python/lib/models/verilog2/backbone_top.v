// =============================================================================
// backbone_top.v
//
// SGLATrack Backbone Top，對齊 run_backbone_numpy_shared_trunk.main()。
// 執行 blocks 0~5（固定層）+ sel_block_i 指定的 block（6~11）+ backbone norm。
//
// ROM 對應（參照 rom.pdf；所有 blocks 0~6 的資料依 block 順序拼接）：
//
//   rom_backbone_blocks_0_6_norm1_weight         7×32  = 224  (memory/*.v module names)
//   rom_backbone_blocks_0_6_norm1_bias           7×32  = 224
//   rom_backbone_blocks_0_6_norm2_weight         7×32  = 224
//   rom_backbone_blocks_0_6_norm2_bias           7×32  = 224
//   rom_backbone_blocks_0_6_attn_qkv_bias        7×96  = 672
//   rom_backbone_blocks_0_3_attn_qkv_weight      4×3072=12288
//   rom_backbone_blocks_4_6_attn_qkv_weight      3×3072= 9216
//   rom_backbone_blocks_0_6_attn_proj_weight     7×1024= 7168
//   rom_backbone_blocks_0_6_attn_proj_bias       7×32  = 224
//   rom_backbone_blocks_0_6_mlp_fc1_bias         7×128 = 896
//   rom_backbone_blocks_0_3_mlp_fc1_weight       4×4096=16384
//   rom_backbone_blocks_4_6_mlp_fc1_weight       3×4096=12288
//   rom_backbone_blocks_0_6_mlp_fc2_bias         7×32  = 224
//   rom_backbone_blocks_0_3_mlp_fc2_weight       4×4096=16384
//   rom_backbone_blocks_4_6_mlp_fc2_weight       3×4096=12288
//   rom_backbone_norm_weight                     1×32  =  32
//   rom_backbone_norm_bias                       1×32  =  32
//
// wgt_addr_o type encoding (bits [15:13]):
//   3'b000 = norm1   3'b001 = norm2
//   3'b010 = attn (qkv/proj unified; local<3072→qkv, ≥3072→proj)
//   3'b100 = fc1     3'b101 = fc2
//
// Inter-block token chaining:
//   Sram_tok1 captures each transformer_block output; replayed to next block.
//   Block 0 reads external x_i/x_valid; blocks 1+ replay from Sram_tok1.
//
// Activation SRAM macros in sglatrack_top; port mux in backbone_top / transformer_block.
//   Sram_tok1 macro (sram_tok1_* ports): inter-block + norm1 + backbone norm.
//   Plan B: S_OUT removed; head reads Sram_tok1 directly in S_FILL.
//
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

    // Adaptive block selection (absolute index 6~11)
    input  wire [3:0]  sel_block_i,

    // Token input stream (post-embedding)
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // Status
    output wire        busy,
    output wire        x_ready,   // 1 when external x_valid is consumed (block 0 S_LOAD_X)
    output reg         done,

    // Output token stream (backbone norm output)
    output wire signed [15:0] y_o,
    output wire        y_valid,

    // Sram_tok1 macro: inter-block tok_buf + backbone norm in-place + output stream
    output wire        sram_tok1_ceb_o,
    output wire        sram_tok1_web_o,
    output wire [13:0] sram_tok1_addr_o,
    output wire [15:0] sram_tok1_din_o,
    input  wire [15:0] sram_tok1_q_i,

    // transformer_block x_buf (Sram_tok2); tmp-on-q (Sram_q, shared with care)
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

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
parameter S_IDLE         = 3'd0;
parameter S_RUN_FIXED    = 3'd1;
parameter S_RUN_SELECTED = 3'd2;
parameter S_BACKBONE_NORM= 3'd3;
// S_OUT removed (Plan B: head reads Sram_tok1 directly)
parameter S_DONE         = 3'd4;

reg [2:0] state, next_state;
reg [3:0] block_idx;
reg [3:0] sel_block_r;

// ---------------------------------------------------------------------------
// Inter-block token buffer (Sram_tok1)
// ---------------------------------------------------------------------------

reg [13:0] tok_wr_ptr;   // write: incremented on tb_y_valid
reg [13:0] tok_rd_ptr;   // read:  drive ahead of consume (SRAM path)
reg        tok_replay;   // 0 for block 0; 1 for all subsequent blocks

reg        tok_rp_phase;    // 0=ADDR (s1 read tok_rd_ptr), 1=USE (x_valid, s1_q stable)
reg        bn_s1_phase;     // 0=ADDR (s1 read bn_rp_feat), 1=USE (bn x_valid)
reg        bt_s1_ceb;
reg        bt_s1_web;
reg [13:0] bt_s1_addr;
reg [15:0] bt_s1_din;
wire [15:0] s1_q;

wire tb_busy, tb_done;
wire tb_x_ready;

// ---------------------------------------------------------------------------
// transformer_block control / data
// ---------------------------------------------------------------------------
wire [15:0] tb_wgt_addr;
wire signed [15:0] bb_wgt_mux;
wire signed [15:0] bb_bias_mux;
wire signed [15:0] tb_y;
wire tb_y_valid;
reg  tb_start;

// Input mux: block 0 uses external stream; subsequent blocks replay tok buffer
wire signed [15:0] tb_x_mux =
    tok_replay ? s1_q : x_i;
wire tb_rp_use =
    tok_replay && tb_busy && (tok_rd_ptr < N_TOKENS*EMBED_DIM) && tok_rp_phase;
wire tb_xv_mux =
    tok_replay ? tb_rp_use : x_valid;

wire              tb_norm1_stg_wr_do;
wire [13:0]       tb_norm1_stg_wr_flat;
wire [15:0]       tb_norm1_stg_wr_din;
wire              tb_norm1_stg_rd_en;
wire [13:0]       tb_norm1_stg_rd_flat;
wire signed [15:0] tb_norm1_stg_x;

assign tb_norm1_stg_x = s1_q;

transformer_block #(
    .EMBED_DIM(EMBED_DIM),
    .MLP_DIM  (4*EMBED_DIM),
    .N_TOKENS (N_TOKENS)
) u_tb (
    .clk(clk), .reset(reset), .start(tb_start),
    .x_i(tb_x_mux), .x_valid(tb_xv_mux),
    .wgt_i   (bb_wgt_mux),
    .bias_i  (bb_bias_mux),
    .wgt_addr_o(tb_wgt_addr),
    .busy(tb_busy), .x_ready(tb_x_ready), .done(tb_done),
    .y_o(tb_y), .y_valid(tb_y_valid),
    .sram_tok2_ceb_o   (sram_tok2_ceb_o),
    .sram_tok2_web_o   (sram_tok2_web_o),
    .sram_tok2_addr_o  (sram_tok2_addr_o),
    .sram_tok2_din_o   (sram_tok2_din_o),
    .sram_tok2_q_i     (sram_tok2_q_i),
    .norm1_stg_wr_do   (tb_norm1_stg_wr_do),
    .norm1_stg_wr_flat (tb_norm1_stg_wr_flat),
    .norm1_stg_wr_din  (tb_norm1_stg_wr_din),
    .norm1_stg_rd_en   (tb_norm1_stg_rd_en),
    .norm1_stg_rd_flat (tb_norm1_stg_rd_flat),
    .norm1_stg_x       (tb_norm1_stg_x),
    .sram_q_ceb_o       (sram_q_ceb_o),
    .sram_q_web_o       (sram_q_web_o),
    .sram_q_addr_o      (sram_q_addr_o),
    .sram_q_din_o       (sram_q_din_o),
    .sram_q_q_i         (sram_q_q_i),
    .sram_k_ceb_o       (sram_k_ceb_o),
    .sram_k_web_o       (sram_k_web_o),
    .sram_k_addr_o      (sram_k_addr_o),
    .sram_k_din_o       (sram_k_din_o),
    .sram_k_q_i         (sram_k_q_i),
    .sram_v_ceb_o       (sram_v_ceb_o),
    .sram_v_web_o       (sram_v_web_o),
    .sram_v_addr_o      (sram_v_addr_o),
    .sram_v_din_o       (sram_v_din_o),
    .sram_v_q_i         (sram_v_q_i),
    .sram_qkm_ceb_o     (sram_qkm_ceb_o),
    .sram_qkm_web_o     (sram_qkm_web_o),
    .sram_qkm_addr_o    (sram_qkm_addr_o),
    .sram_qkm_din_o     (sram_qkm_din_o),
    .sram_qkm_q_i       (sram_qkm_q_i)
);

// ---------------------------------------------------------------------------
// Backbone final layer_norm
// ---------------------------------------------------------------------------
wire signed [15:0] bn_wgt_mux, bn_bias_mux;
wire [9:0] bn_feat_addr;

wire bn_busy, bn_done;
wire signed [15:0] bn_y;
wire signed [15:0] bn_y_sat;
wire bn_y_valid;
reg  bn_start;
reg  [8:0] bn_tok_cnt;   // which token u_bn is processing (0..N_TOKENS-1)

// Replay Sram_tok1 -> u_bn: stream EMBED_DIM values per token.
// Must not assert x_valid until u_bn is busy (LN has left S_IDLE after start);
// otherwise the first LOAD cycle can miss a sample and u_bn stays in S_LOAD forever.
reg       bn_rp_stream;   // x_valid to u_bn during feature replay
reg       bn_arm;         // set with bn_start; wait bn_busy before streaming
reg [4:0] bn_rp_feat;     // 0..EMBED_DIM-1 (same as transformer_block rp_feat)

wire [4:0] bn_feat_cnt = bn_rp_feat;

wire [13:0] bn_rp_addr = bn_tok_cnt * EMBED_DIM + {9'b0, bn_feat_cnt};
wire signed [15:0] bn_x_mux =
    s1_q;
wire bn_rp_use = bn_rp_stream && bn_s1_phase;
wire bn_xv_mux =
    bn_rp_use;

// Flat index for out_buf capture (row-major tok*C+ch); must match numpy golden flatten.
wire [13:0] bn_cap_flat = bn_tok_cnt * EMBED_DIM + {9'b0, bn_feat_addr[4:0]};

wire bn_out_beat;

layer_norm #(.FEAT_DIM(EMBED_DIM), .RCP_NUM(65536/EMBED_DIM)) u_bn (
    .clk(clk), .reset(reset), .start(bn_start),
    .x_i(bn_x_mux), .x_valid(bn_xv_mux),
    .w_i(bn_wgt_mux), .b_i(bn_bias_mux),
    .feat_addr_o(bn_feat_addr),
    .busy(bn_busy), .done(bn_done),
    .y_o(bn_y), .y_valid(bn_y_valid),
    .y_sat_o(bn_y_sat),
    .out_beat_o(bn_out_beat)
);

// Posedge latch norm y_sat_o + flat (layer_norm out_beat_o); bn_wr_do -> tok1 1 cycle later.
reg [13:0]        bn_wr_flat_lat;
reg signed [15:0] bn_wr_din_lat;
reg               bn_wr_do;

// Plan B: S_OUT removed; Sram_tok1 retains backbone norm for head direct read

assign sram_tok1_ceb_o  = bt_s1_ceb;
assign sram_tok1_web_o  = bt_s1_web;
assign sram_tok1_addr_o = bt_s1_addr;
assign sram_tok1_din_o  = bt_s1_din;
assign s1_q             = sram_tok1_q_i;

// ---------------------------------------------------------------------------
// ROM Q wires (per weight type)
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
//   tb_wgt_addr[15:13] = weight type (3-bit)
//   tb_wgt_addr[12:0]  = local address within type
// ---------------------------------------------------------------------------
wire [2:0]  wtype      = tb_wgt_addr[15:13];
wire [12:0] local_addr = tb_wgt_addr[12:0];

// norm1/norm2 (32 entries per block): addr = block_idx*32 + local[4:0] (feat idx)
wire [7:0] blk_x32  = ({4'b0, block_idx} << 5);
wire [7:0] addr_n1w  = blk_x32 + {3'b0, local_addr[4:0]};
wire [7:0] addr_n1b  = addr_n1w;
wire [7:0] addr_n2w  = addr_n1w;
wire [7:0] addr_n2b  = addr_n1w;
// proj_b (32 entries per block): one bias per output neuron.
// proj weight ROM layout = neuron-outer / feat-inner (PyTorch nn.Linear C-order),
// so for proj local_addr = (px_neu<<5) + mac_idx → local[9:5] = neuron, local[4:0] = feat.
// Bias must index by neuron, hence local[9:5] (NOT [4:0], which would pick by feat).
wire [7:0] addr_projb = blk_x32 + {3'b0, local_addr[9:5]};

// qkv_b (96 entries per block): addr = block_idx*96 + neuron[6:0]
//   neuron = local_addr[12:5] (= mlp_neu or attn neuron)
wire [9:0] blk_x96   = ({6'b0, block_idx} << 6) + ({7'b0, block_idx} << 5);
wire [9:0] addr_qkvb  = blk_x96 + {3'b0, local_addr[12:5]};

// fc1_b (128 entries per block): addr = block_idx*128 + neuron[6:0]
//   neuron = local_addr[11:5]
wire [9:0] blk_x128  = ({3'b0, block_idx} << 7);
wire [9:0] addr_fc1b  = blk_x128 + {3'b0, local_addr[11:5]};

// fc2_b (32 entries per block): addr = block_idx*32 + neuron[4:0]
//   neuron = local_addr[11:7]
wire [7:0] addr_fc2b  = blk_x32 + {3'b0, local_addr[11:7]};

// proj_w (1024 entries per block): addr = block_idx*1024 + local[9:0]
wire [12:0] blk_x1024 = ({3'b0, block_idx} << 10);
wire [12:0] addr_projw = blk_x1024 + {3'b0, local_addr[9:0]};

// qkv/fc1/fc2 weight blocks 0-3: addr = block_idx*N + local
//   QKV: N=3072 = 2048+1024 → block_idx<<11 + block_idx<<10
wire [13:0] blk_x3072_03 = ({4'b0, block_idx[1:0]} << 11) +
                            ({5'b0, block_idx[1:0]} << 10);
wire [13:0] addr_qkvw03  = blk_x3072_03 + {1'b0, local_addr[12:0]};

//   FC1/FC2: N=4096 → block_idx<<12
wire [13:0] blk_x4096_03 = {block_idx[1:0], 12'b0};
wire [13:0] addr_fc1w03  = blk_x4096_03 + {1'b0, local_addr[12:0]};
wire [13:0] addr_fc2w03  = blk_x4096_03 + {1'b0, local_addr[12:0]};

// qkv/fc1/fc2 weight blocks 4-6: (block_idx-4)*N + local
wire [1:0]  blk_off46    = block_idx[1:0];   // 4→0, 5→1, 6→2
wire [13:0] blk_x3072_46 = ({4'b0, blk_off46} << 11) + ({5'b0, blk_off46} << 10);
wire [13:0] addr_qkvw46  = blk_x3072_46 + {1'b0, local_addr[12:0]};
wire [13:0] blk_x4096_46 = {blk_off46, 12'b0};
wire [13:0] addr_fc1w46  = blk_x4096_46 + {1'b0, local_addr[12:0]};
wire [13:0] addr_fc2w46  = blk_x4096_46 + {1'b0, local_addr[12:0]};

// Backbone norm ROM (indexed by feat_addr from layer_norm); pad to macro addr width
wire [6:0] addr_bnorm = {2'b00, bn_feat_addr[4:0]};

// ---------------------------------------------------------------------------
// ROM chip-enables (active low)
// ---------------------------------------------------------------------------
wire bb_active = (state == S_RUN_FIXED || state == S_RUN_SELECTED);
wire bn_active = (state == S_BACKBONE_NORM);
wire blk_lo    = (block_idx <= 4'd3);
wire blk_hi    = (block_idx >= 4'd4);

// Type-gated enables: only the ROM matching the current wtype is enabled
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
// Weight / bias mux → transformer_block
//   wtype[2:0] from tb_wgt_addr[15:13]; local_addr from [12:0].
//   For attn: local_addr < 3072 → qkv, ≥ 3072 → proj.
// ---------------------------------------------------------------------------
wire is_qkv  = (wtype == 3'b010) && (local_addr < 13'd3072);
wire is_proj = (wtype == 3'b010) && (local_addr >= 13'd3072);

assign bb_wgt_mux =
    (wtype == 3'b000) ? q_norm1_w :
    (wtype == 3'b001) ? q_norm2_w :
    (is_qkv)          ? (blk_lo ? q_qkv_w_03 : q_qkv_w_46) :
    (is_proj)         ? q_proj_w :
    (wtype == 3'b100) ? (blk_lo ? q_fc1_w_03 : q_fc1_w_46) :
    (wtype == 3'b101) ? (blk_lo ? q_fc2_w_03 : q_fc2_w_46) :
    16'sd0;

assign bb_bias_mux =
    (wtype == 3'b000) ? q_norm1_b :
    (wtype == 3'b001) ? q_norm2_b :
    (is_qkv)          ? q_qkv_b :
    (is_proj)         ? q_proj_b :
    (wtype == 3'b100) ? q_fc1_b :
    (wtype == 3'b101) ? q_fc2_b :
    16'sd0;

// Backbone final norm uses dedicated ROM
assign bn_wgt_mux  = q_bnorm_w;
assign bn_bias_mux = q_bnorm_b;

// ---------------------------------------------------------------------------
// Backbone block weight ROM instantiations (CLK = ~clk for falling-edge read)
// ---------------------------------------------------------------------------
rom_backbone_blocks_0_6_norm1_weight u_rom_n1w (
    .A(addr_n1w), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n1w),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm1_w));

rom_backbone_blocks_0_6_norm1_bias u_rom_n1b (
    .A(addr_n1b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n1b),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm1_b));

rom_backbone_blocks_0_6_norm2_weight u_rom_n2w (
    .A(addr_n2w), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n2w),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm2_w));

rom_backbone_blocks_0_6_norm2_bias u_rom_n2b (
    .A(addr_n2b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_n2b),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_norm2_b));

rom_backbone_blocks_0_6_attn_qkv_bias u_rom_qkvb (
    .A(addr_qkvb), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_qkvb),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_qkv_b));

rom_backbone_blocks_0_3_attn_qkv_weight u_rom_qkvw03 (
    .A(addr_qkvw03), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_qkvw03),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_qkv_w_03));

rom_backbone_blocks_4_6_attn_qkv_weight u_rom_qkvw46 (
    .A(addr_qkvw46), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_qkvw46),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_qkv_w_46));

rom_backbone_blocks_0_6_attn_proj_weight u_rom_projw (
    .A(addr_projw), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_projw),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_proj_w));

rom_backbone_blocks_0_6_attn_proj_bias u_rom_projb (
    .A(addr_projb), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_projb),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_proj_b));

rom_backbone_blocks_0_6_mlp_fc1_bias u_rom_fc1b (
    .A(addr_fc1b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc1b),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc1_b));

rom_backbone_blocks_0_3_mlp_fc1_weight u_rom_fc1w03 (
    .A(addr_fc1w03), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc1w03),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc1_w_03));

rom_backbone_blocks_4_6_mlp_fc1_weight u_rom_fc1w46 (
    .A(addr_fc1w46), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc1w46),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc1_w_46));

rom_backbone_blocks_0_6_mlp_fc2_bias u_rom_fc2b (
    .A(addr_fc2b), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc2b),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc2_b));

rom_backbone_blocks_0_3_mlp_fc2_weight u_rom_fc2w03 (
    .A(addr_fc2w03), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc2w03),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc2_w_03));

rom_backbone_blocks_4_6_mlp_fc2_weight u_rom_fc2w46 (
    .A(addr_fc2w46), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_fc2w46),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_fc2_w_46));

rom_backbone_norm_weight u_rom_bnw (
    .A(addr_bnorm), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_bnw),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_bnorm_w));

rom_backbone_norm_bias u_rom_bnb (
    .A(addr_bnorm), .AM(), .CEBM(), .BIST(1'b0), .CEB(ceb_bnb),
    .CLK(~clk), .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0), .Q(q_bnorm_b));

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
            next_state = start ? S_RUN_FIXED : S_IDLE;
        S_RUN_FIXED:
            // After block START_LAYER done, switch to adaptive block
            next_state = (tb_done && block_idx == START_LAYER)
                         ? S_RUN_SELECTED : S_RUN_FIXED;
        S_RUN_SELECTED:
            next_state = tb_done ? S_BACKBONE_NORM : S_RUN_SELECTED;
        S_BACKBONE_NORM:
            next_state = (bn_done && (bn_tok_cnt == N_TOKENS-1))
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
    done     <= 1'b0;
    tb_start <= 1'b0;
    bn_start <= 1'b0;

    if (reset) begin
        block_idx   <= 4'd0;
        sel_block_r <= 4'd0;
        tok_wr_ptr  <= 14'd0;
        tok_rd_ptr  <= 14'd0;
        tok_replay  <= 1'b0;
        bn_tok_cnt      <= 9'd0;
        bn_arm          <= 1'b0;
        bn_rp_feat      <= 5'd0;
        bn_rp_stream    <= 1'b0;
        bn_wr_flat_lat <= 14'd0;
        bn_wr_din_lat  <= 16'd0;
        bn_wr_do       <= 1'b0;
        tok_rp_phase    <= 1'b0;
        bn_s1_phase     <= 1'b0;
    end else begin
        // ---- Capture transformer_block output -> tok buffer ----
        if (tb_y_valid)
            tok_wr_ptr <= tok_wr_ptr + 14'd1;
        // 2-phase s1 replay: ADDR then USE (do not bump rd_ptr on ADDR beat)
        if (tok_replay && tb_busy && (tok_rd_ptr < N_TOKENS*EMBED_DIM)) begin
            if (tok_rp_phase == 1'b0)
                tok_rp_phase <= 1'b1;
            else begin
                tok_rd_ptr   <= tok_rd_ptr + 14'd1;
                tok_rp_phase <= 1'b0;
            end
        end else
            tok_rp_phase <= 1'b0;

        // norm y_sat_o posedge latch -> tok1 in-place (see layer_norm out_beat_o)
        if (state == S_BACKBONE_NORM && bn_out_beat) begin
            bn_wr_flat_lat <= bn_cap_flat;
            bn_wr_din_lat  <= bn_y_sat;
        end
        bn_wr_do <= (state == S_BACKBONE_NORM && bn_out_beat);

        // ---- Backbone norm: gated feature stream into u_bn ----
        if (state == S_BACKBONE_NORM) begin
            if (bn_arm && bn_busy) begin
                bn_rp_stream <= 1'b1;
                bn_rp_feat   <= 5'd0;
                bn_arm       <= 1'b0;
            end
            if (bn_rp_stream) begin
                if (bn_s1_phase == 1'b0)
                    bn_s1_phase <= 1'b1;
                else begin
                    bn_s1_phase <= 1'b0;
                    if (bn_rp_feat == EMBED_DIM - 1) begin
                        bn_rp_stream <= 1'b0;
                        bn_rp_feat   <= 5'd0;
                    end else
                        bn_rp_feat <= bn_rp_feat + 5'd1;
                end
            end
            else
                bn_s1_phase <= 1'b0;
        end

        case (state)
            // ------------------------------------------------------------
            S_IDLE: begin
                block_idx   <= 4'd0;
                tok_replay  <= 1'b0;
                tok_wr_ptr  <= 14'd0;
                tok_rd_ptr  <= 14'd0;
                bn_tok_cnt      <= 9'd0;
                bn_arm          <= 1'b0;
                bn_rp_feat      <= 5'd0;
                bn_rp_stream    <= 1'b0;
                tok_rp_phase    <= 1'b0;
                bn_s1_phase     <= 1'b0;
                if (start)
                    sel_block_r <= sel_block_i;
            end

            // ------------------------------------------------------------
            // Run fixed blocks 0..START_LAYER sequentially
            // ------------------------------------------------------------
            S_RUN_FIXED: begin
                // Start u_tb for current block when idle
                if (!tb_busy) begin
                    tb_start   <= 1'b1;
                    tok_rd_ptr <= 14'd0;  // reset read for this block's replay
                    tok_rp_phase <= 1'b0;
                end

                // When block completes: advance block_idx and enable replay
                if (tb_done) begin
                    tok_wr_ptr <= 14'd0;   // reset write for next block
                    if (block_idx < START_LAYER) begin
                        block_idx  <= block_idx + 4'd1;
                        tok_replay <= 1'b1;  // blocks 1+ replay Sram_tok1
                    end
                    // if block_idx == START_LAYER: FSM transitions to S_RUN_SELECTED
                end
            end

            // ------------------------------------------------------------
            // Run adaptive selected block
            // ------------------------------------------------------------
            S_RUN_SELECTED: begin
                block_idx <= sel_block_r;

                if (!tb_busy) begin
                    tb_start   <= 1'b1;
                    tok_rd_ptr <= 14'd0;
                    tok_rp_phase <= 1'b0;
                end

                if (tb_done) begin
                    tok_wr_ptr <= 14'd0;
                    // tok_replay stays 1; backbone norm reads Sram_tok1
                end
            end

            // ------------------------------------------------------------
            // Backbone norm: process each token through u_bn
            // ------------------------------------------------------------
            S_BACKBONE_NORM: begin
                if (bn_arm && bn_busy)
                    bn_s1_phase <= 1'b0;
                // Pulse start; arm stream — x_valid only after u_bn busy (see block above)
                if (!bn_busy && !bn_rp_stream && !bn_arm) begin
                    bn_start <= 1'b1;
                    bn_arm   <= 1'b1;
                end

                // Advance token counter after each u_bn completion
                if (bn_done && (bn_tok_cnt < N_TOKENS - 1))
                    bn_tok_cnt <= bn_tok_cnt + 9'd1;
            end

            // ------------------------------------------------------------
            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

// tok1 port mux: norm1 staging / capture / replay / backbone norm
always @(*) begin
    bt_s1_ceb  = 1'b1;
    bt_s1_web  = 1'b1;
    bt_s1_addr = 14'd0;
    bt_s1_din  = 16'd0;

    if (state == S_BACKBONE_NORM && bn_wr_do) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b0;
        bt_s1_addr = bn_wr_flat_lat;
        bt_s1_din  = bn_wr_din_lat;
    end else if ((state == S_RUN_FIXED || state == S_RUN_SELECTED) &&
                 tb_norm1_stg_wr_do) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b0;
        bt_s1_addr = tb_norm1_stg_wr_flat;
        bt_s1_din  = tb_norm1_stg_wr_din;
    end else if ((state == S_RUN_FIXED || state == S_RUN_SELECTED) &&
                 tb_norm1_stg_rd_en) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b1;
        bt_s1_addr = tb_norm1_stg_rd_flat;
    end else if ((state == S_RUN_FIXED || state == S_RUN_SELECTED) && tb_y_valid) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b0;
        bt_s1_addr = tok_wr_ptr;
        bt_s1_din  = tb_y;
    end else if ((state == S_RUN_FIXED || state == S_RUN_SELECTED) &&
                 tok_replay && tb_busy && (tok_rd_ptr < N_TOKENS*EMBED_DIM) &&
                 (tok_rp_phase == 1'b0)) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b1;
        bt_s1_addr = tok_rd_ptr;
    end else if (state == S_BACKBONE_NORM && bn_rp_stream && (bn_s1_phase == 1'b0)) begin
        bt_s1_ceb  = 1'b0;
        bt_s1_web  = 1'b1;
        bt_s1_addr = bn_rp_addr;
    end
end

// ---------------------------------------------------------------------------
// Plan B: S_OUT removed; y_o/y_valid left undriven (head reads Sram_tok1 directly)
// ---------------------------------------------------------------------------
assign y_o     = 16'sd0;
assign y_valid = 1'b0;
assign busy    = (state != S_IDLE);
assign x_ready = !tok_replay && tb_x_ready;

endmodule
