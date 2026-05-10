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
//   tok_buf captures each transformer_block's output; replayed to next block.
//   Block 0 reads external x_i/x_valid; blocks 1+ replay tok_buf.
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
    output reg         done,

    // Output token stream (backbone norm output)
    output wire signed [15:0] y_o,
    output wire        y_valid
);

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
parameter S_IDLE         = 3'd0;
parameter S_RUN_FIXED    = 3'd1;
parameter S_RUN_SELECTED = 3'd2;
parameter S_BACKBONE_NORM= 3'd3;
parameter S_OUT          = 3'd4;
parameter S_DONE         = 3'd5;

reg [2:0] state, next_state;
reg [3:0] block_idx;
reg [3:0] sel_block_r;

// ---------------------------------------------------------------------------
// Inter-block token buffer
//   tok_buf captures transformer_block output after each block run.
//   Replayed to next block (block 0 uses external x_i/x_valid).
// ---------------------------------------------------------------------------
reg signed [15:0] tok_buf [0:N_TOKENS*EMBED_DIM-1];
reg [13:0] tok_wr_ptr;   // write: incremented on tb_y_valid
reg [13:0] tok_rd_ptr;   // read:  incremented during replay to u_tb
reg        tok_replay;   // 0 for block 0; 1 for all subsequent blocks

// transformer_block status (must precede tok_rp_active — uses tb_busy)
wire tb_busy, tb_done;

// tok_rp_active: 1 when we should be streaming tok_buf to u_tb
wire tok_rp_active = tok_replay && tb_busy && (tok_rd_ptr < N_TOKENS*EMBED_DIM);

// ---------------------------------------------------------------------------
// transformer_block control / data
// ---------------------------------------------------------------------------
wire [15:0] tb_wgt_addr;
wire signed [15:0] bb_wgt_mux;
wire signed [15:0] bb_bias_mux;
wire signed [15:0] tb_y;
wire tb_y_valid;
reg  tb_start;

// Input mux: block 0 uses external stream; subsequent blocks replay tok_buf
wire signed [15:0] tb_x_mux  = tok_replay ? tok_buf[tok_rd_ptr] : x_i;
wire               tb_xv_mux = tok_replay ? tok_rp_active       : x_valid;

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
    .block_idx (block_idx),
    .busy(tb_busy), .done(tb_done),
    .y_o(tb_y), .y_valid(tb_y_valid)
);

// ---------------------------------------------------------------------------
// Backbone final layer_norm
// ---------------------------------------------------------------------------
wire signed [15:0] bn_wgt_mux, bn_bias_mux;
wire [9:0] bn_feat_addr;

wire bn_busy, bn_done;
wire signed [15:0] bn_y;
wire bn_y_valid;
reg  bn_start;
reg  [8:0] bn_tok_cnt;   // which token u_bn is processing (0..N_TOKENS-1)

// Replay tok_buf → u_bn: stream EMBED_DIM values per token
reg [4:0] bn_feat_cnt;    // 0..EMBED_DIM-1 within current token
reg       bn_rp_stream;   // active during EMBED_DIM replay cycles

wire [13:0] bn_rp_addr = bn_tok_cnt * EMBED_DIM + {9'b0, bn_feat_cnt};
wire signed [15:0] bn_x_mux  = tok_buf[bn_rp_addr];
wire               bn_xv_mux = bn_rp_stream;

layer_norm #(.FEAT_DIM(EMBED_DIM), .RCP_NUM(65536/EMBED_DIM)) u_bn (
    .clk(clk), .reset(reset), .start(bn_start),
    .x_i(bn_x_mux), .x_valid(bn_xv_mux),
    .w_i(bn_wgt_mux), .b_i(bn_bias_mux),
    .feat_addr_o(bn_feat_addr),
    .busy(bn_busy), .done(bn_done),
    .y_o(bn_y), .y_valid(bn_y_valid)
);

// ---------------------------------------------------------------------------
// Backbone norm output buffer (captures bn_y for streaming in S_OUT)
// ---------------------------------------------------------------------------
reg signed [15:0] out_buf [0:N_TOKENS*EMBED_DIM-1];
reg [13:0] out_wr_addr;  // write address (incremented on bn_y_valid)
reg [13:0] out_rd_addr;  // read address (incremented in S_OUT)

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

// norm1/norm2/proj_b (32 entries per block): addr = block_idx*32 + local[4:0]
wire [7:0] blk_x32  = ({4'b0, block_idx} << 5);
wire [7:0] addr_n1w  = blk_x32 + {3'b0, local_addr[4:0]};
wire [7:0] addr_n1b  = addr_n1w;
wire [7:0] addr_n2w  = addr_n1w;
wire [7:0] addr_n2b  = addr_n1w;
wire [7:0] addr_projb = blk_x32 + {3'b0, local_addr[4:0]};

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
            // Process N_TOKENS tokens one-by-one through u_bn
            next_state = (bn_done && bn_tok_cnt == N_TOKENS-1)
                         ? S_OUT : S_BACKBONE_NORM;
        S_OUT:
            next_state = (out_rd_addr == N_TOKENS*EMBED_DIM-1) ? S_DONE : S_OUT;
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
        bn_tok_cnt  <= 9'd0;
        bn_feat_cnt <= 5'd0;
        bn_rp_stream <= 1'b0;
        out_wr_addr <= 14'd0;
        out_rd_addr <= 14'd0;
    end else begin
        // ---- Always: capture transformer_block output → tok_buf ----
        if (tb_y_valid) begin
            tok_buf[tok_wr_ptr] <= tb_y;
            tok_wr_ptr <= tok_wr_ptr + 14'd1;
        end

        // ---- Always: advance tok_buf read pointer during replay ----
        if (tok_rp_active)
            tok_rd_ptr <= tok_rd_ptr + 14'd1;

        // ---- Always: capture backbone norm output → out_buf ----
        if (bn_y_valid) begin
            out_buf[out_wr_addr] <= bn_y;
            out_wr_addr <= out_wr_addr + 14'd1;
        end

        // ---- Always: advance bn_rp_stream feature counter ----
        if (bn_rp_stream) begin
            if (bn_feat_cnt == EMBED_DIM-1) begin
                bn_rp_stream <= 1'b0;
                bn_feat_cnt  <= 5'd0;
            end else begin
                bn_feat_cnt <= bn_feat_cnt + 5'd1;
            end
        end

        case (state)
            // ------------------------------------------------------------
            S_IDLE: begin
                block_idx   <= 4'd0;
                tok_replay  <= 1'b0;
                tok_wr_ptr  <= 14'd0;
                tok_rd_ptr  <= 14'd0;
                bn_tok_cnt  <= 9'd0;
                bn_feat_cnt <= 5'd0;
                bn_rp_stream <= 1'b0;
                out_wr_addr <= 14'd0;
                out_rd_addr <= 14'd0;
                if (start) sel_block_r <= sel_block_i;
            end

            // ------------------------------------------------------------
            // Run fixed blocks 0..START_LAYER sequentially
            // ------------------------------------------------------------
            S_RUN_FIXED: begin
                // Start u_tb for current block when idle
                if (!tb_busy) begin
                    tb_start   <= 1'b1;
                    tok_rd_ptr <= 14'd0;  // reset read for this block's replay
                end

                // When block completes: advance block_idx and enable replay
                if (tb_done) begin
                    tok_wr_ptr <= 14'd0;   // reset write for next block
                    if (block_idx < START_LAYER) begin
                        block_idx  <= block_idx + 4'd1;
                        tok_replay <= 1'b1;  // blocks 1+ use tok_buf
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
                end

                if (tb_done) begin
                    tok_wr_ptr <= 14'd0;
                    // tok_replay stays 1; backbone norm reads tok_buf
                end
            end

            // ------------------------------------------------------------
            // Backbone norm: process each token through u_bn
            // ------------------------------------------------------------
            S_BACKBONE_NORM: begin
                // Start u_bn for current token when idle and not streaming
                if (!bn_busy && !bn_rp_stream) begin
                    bn_start    <= 1'b1;
                    bn_rp_stream <= 1'b1;
                    bn_feat_cnt  <= 5'd0;
                end

                // Advance token counter after each u_bn completion
                if (bn_done && bn_tok_cnt < N_TOKENS-1)
                    bn_tok_cnt <= bn_tok_cnt + 9'd1;
            end

            // ------------------------------------------------------------
            // Stream out_buf to y_o
            // ------------------------------------------------------------
            S_OUT: begin
                out_rd_addr <= out_rd_addr + 14'd1;
            end

            // ------------------------------------------------------------
            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Output stream
// ---------------------------------------------------------------------------
assign y_o     = out_buf[out_rd_addr];
assign y_valid = (state == S_OUT);
assign busy    = (state != S_IDLE);

endmodule
