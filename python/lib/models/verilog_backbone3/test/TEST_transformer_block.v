`timescale 1ns/10ps

`include "inv_sqrt_lut_seed.v"
`include "inv_sqrt_nr.v"
`include "recip_lut_seed.v"
`include "recip_nr.v"
`include "common/vec_mac8.v"
`include "common/q88_ops.v"
`include "linear_vec8.v"
`include "residual.v"
`include "layer_norm_pip.v"
`include "care_attention.v"
`include "mlp_ws.v"
`include "transformer_block.v"

// =============================================================================
// TEST_transformer_block.v -- block0 end-to-end with per-stage golden check
//
// DUT: transformer_block  (norm1 -> attn -> res1 -> norm2 -> mlp -> res2)
//
// Golden (all from block 0):
//   Input:  merged_tokens_bi.txt  (320*32 = 10240)
//   NORM1:  backbone_blocks_0_after_norm1_out_bi.txt
//   ATTN:   backbone_blocks_0_after_attn_attn_out_bi.txt
//   RES1:   backbone_blocks_0_after_residual_add1_out_bi.txt
//   NORM2:  backbone_blocks_0_after_norm2_out_bi.txt
//   MLP:    backbone_blocks_0_mlp_after_mlp_out_bi.txt
//   BLOCK:  backbone_blocks_0_after_block_out_bi.txt
//
// Weights:
//   norm1:  backbone_blocks_0_norm1_{weight,bias}_bi.txt
//   norm2:  backbone_blocks_0_norm2_{weight,bias}_bi.txt
//   attn:   backbone_blocks_0_attn_{qkv,proj}_{weight,bias}_bi.txt
//   mlp:    backbone_blocks_0_mlp_fc{1,2}_{weight,bias}_bi.txt
//
// VCS (run from verilog_backbone3/test):
//   vcs TEST_transformer_block.v +incdir+.. +incdir+../common \
//       +lint=TFIPC-L +define+TSMC_CM_NO_WARNING \
//       -debug_access+all -debug_region+cell | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|transformer_block done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_transformer_block;

parameter CYCLE     = 2.0;
parameter EMBED_DIM = 32;
parameter MLP_DIM   = 128;
parameter N_TOKENS  = 320;
parameter TOK_FLAT  = N_TOKENS * EMBED_DIM;

// Weight ROM sizes
parameter NORM_W_DEPTH   = EMBED_DIM;
parameter QKV_W_DEPTH    = 96 * EMBED_DIM;
parameter QKV_B_DEPTH    = 96;
parameter PROJ_W_DEPTH   = EMBED_DIM * EMBED_DIM;
parameter PROJ_B_DEPTH   = EMBED_DIM;
parameter FC1_W_DEPTH    = MLP_DIM * EMBED_DIM;
parameter FC1_B_DEPTH    = MLP_DIM;
parameter FC2_W_DEPTH    = EMBED_DIM * MLP_DIM;
parameter FC2_B_DEPTH    = EMBED_DIM;

// SRAM sizes
parameter K_MEM_DEPTH   = 16384;
parameter V_MEM_DEPTH   = 16384;
parameter QKM_MEM_DEPTH = 64 * MLP_DIM;  // tokens 256..319 fc1 scratch (mlp_ws)

// =========================================================================
// DUT signals
// =========================================================================
reg         clk, reset, start;

wire [15:0] wgt_addr_o;
wire [7:0]  bias_addr_o;
wire        busy, done;

wire signed [15:0] wgt_i;
wire signed [15:0] bias_i;

wire        sram_tok1_ceb_o, sram_tok1_web_o;
wire [13:0] sram_tok1_addr_o;
wire [15:0] sram_tok1_din_o;
wire [15:0] sram_tok1_q_i;

wire        sram_tok2_ceb_o, sram_tok2_web_o;
wire [13:0] sram_tok2_addr_o;
wire [15:0] sram_tok2_din_o;
wire [15:0] sram_tok2_q_i;

wire        sram_q_ceb_o, sram_q_web_o;
wire [13:0] sram_q_addr_o;
wire [15:0] sram_q_din_o;
wire [15:0] sram_q_q_i;

wire        sram_k_ceb_o, sram_k_web_o;
wire [13:0] sram_k_addr_o;
wire [15:0] sram_k_din_o;
wire [15:0] sram_k_q_i;

wire        sram_v_ceb_o, sram_v_web_o;
wire [13:0] sram_v_addr_o;
wire [15:0] sram_v_din_o;
wire [15:0] sram_v_q_i;

wire        sram_qkm_ceb_o, sram_qkm_web_o;
wire [13:0] sram_qkm_addr_o;
wire [15:0] sram_qkm_din_o;
wire [15:0] sram_qkm_q_i;

// =========================================================================
// SRAM behavioral models (register arrays)
// =========================================================================
reg [15:0] TOK1_MEM [0:TOK_FLAT-1];
reg [15:0] TOK2_MEM [0:TOK_FLAT-1];
reg [15:0] Q_MEM    [0:TOK_FLAT-1];
reg [15:0] K_MEM    [0:K_MEM_DEPTH-1];
reg [15:0] V_MEM    [0:V_MEM_DEPTH-1];
reg [15:0] QKM_MEM  [0:QKM_MEM_DEPTH-1];

reg [13:0] tok1_raddr_q, tok2_raddr_q, q_raddr_q;
reg [13:0] k_raddr_q, v_raddr_q, qkm_raddr_q;

reg [15:0] GOLD_NORM1 [0:TOK_FLAT-1];
reg [15:0] GOLD_ATTN  [0:TOK_FLAT-1];
reg [15:0] GOLD_RES1  [0:TOK_FLAT-1];
reg [15:0] GOLD_NORM2 [0:TOK_FLAT-1];
reg [15:0] GOLD_MLP   [0:TOK_FLAT-1];
reg [15:0] GOLD_BLOCK [0:TOK_FLAT-1];
reg [15:0] INPUT_MEM  [0:TOK_FLAT-1];

reg [15:0] NORM1_W [0:NORM_W_DEPTH-1];
reg [15:0] NORM1_B [0:NORM_W_DEPTH-1];
reg [15:0] NORM2_W [0:NORM_W_DEPTH-1];
reg [15:0] NORM2_B [0:NORM_W_DEPTH-1];

reg [15:0] QKV_W   [0:QKV_W_DEPTH-1];
reg [15:0] QKV_B   [0:QKV_B_DEPTH-1];
reg [15:0] PROJ_W  [0:PROJ_W_DEPTH-1];
reg [15:0] PROJ_B  [0:PROJ_B_DEPTH-1];

reg [15:0] FC1_W   [0:FC1_W_DEPTH-1];
reg [15:0] FC1_B   [0:FC1_B_DEPTH-1];
reg [15:0] FC2_W   [0:FC2_W_DEPTH-1];
reg [15:0] FC2_B   [0:FC2_B_DEPTH-1];

reg [15:0] w_addr_q;
reg [7:0]  b_addr_q;

reg [31:0] cycle_cnt;
integer    chk_i;
reg [31:0] chk_mism;
reg [31:0] chk_first;
reg [3:0]  prev_state;

// SRAM read: negedge latch
always @(negedge clk) begin
    if (!sram_tok1_ceb_o && sram_tok1_web_o)
        tok1_raddr_q <= sram_tok1_addr_o;
    if (!sram_tok2_ceb_o && sram_tok2_web_o)
        tok2_raddr_q <= sram_tok2_addr_o;
    if (!sram_q_ceb_o && sram_q_web_o)
        q_raddr_q <= sram_q_addr_o;
    if (!sram_k_ceb_o && sram_k_web_o)
        k_raddr_q <= sram_k_addr_o;
    if (!sram_v_ceb_o && sram_v_web_o)
        v_raddr_q <= sram_v_addr_o;
    if (!sram_qkm_ceb_o && sram_qkm_web_o)
        qkm_raddr_q <= sram_qkm_addr_o;
end

assign sram_tok1_q_i = TOK1_MEM[tok1_raddr_q];
assign sram_tok2_q_i = TOK2_MEM[tok2_raddr_q];
assign sram_q_q_i    = Q_MEM[q_raddr_q];
assign sram_k_q_i    = K_MEM[k_raddr_q];
assign sram_v_q_i    = V_MEM[v_raddr_q];
assign sram_qkm_q_i  = QKM_MEM[qkm_raddr_q[12:0]];

// SRAM write: posedge capture (matches existing TB convention)
always @(posedge clk) begin
    if (!sram_tok1_ceb_o && !sram_tok1_web_o)
        TOK1_MEM[sram_tok1_addr_o] <= sram_tok1_din_o;
    if (!sram_tok2_ceb_o && !sram_tok2_web_o)
        TOK2_MEM[sram_tok2_addr_o] <= sram_tok2_din_o;
    if (!sram_q_ceb_o && !sram_q_web_o)
        Q_MEM[sram_q_addr_o] <= sram_q_din_o;
    if (!sram_k_ceb_o && !sram_k_web_o)
        K_MEM[sram_k_addr_o] <= sram_k_din_o;
    if (!sram_v_ceb_o && !sram_v_web_o)
        V_MEM[sram_v_addr_o] <= sram_v_din_o;
    if (!sram_qkm_ceb_o && !sram_qkm_web_o)
        QKM_MEM[sram_qkm_addr_o[12:0]] <= sram_qkm_din_o;
end

// =========================================================================
// ROM address decode
//   norm w/b  : combinational (layer_norm_pip needs same-cycle response)
//   attn/mlp  : negedge latch (2-phase protocol, data at T+1)
// =========================================================================
wire [2:0] wtype_live = wgt_addr_o[15:13];

// Norm weight/bias: purely combinational from wgt_addr_o
wire signed [15:0] norm_wgt =
    (wtype_live == 3'b000) ? $signed(NORM1_W[wgt_addr_o[4:0]]) :
    (wtype_live == 3'b001) ? $signed(NORM2_W[wgt_addr_o[4:0]]) :
    16'sd0;
wire signed [15:0] norm_bias =
    (wtype_live == 3'b000) ? $signed(NORM1_B[wgt_addr_o[4:0]]) :
    (wtype_live == 3'b001) ? $signed(NORM2_B[wgt_addr_o[4:0]]) :
    16'sd0;

// Attn/MLP weight/bias: negedge-latched (1-cycle ROM latency)
wire [2:0] wtype_lat = w_addr_q[15:13];

always @(negedge clk) begin
    w_addr_q <= wgt_addr_o;
    b_addr_q <= bias_addr_o;
end

wire signed [15:0] attn_mlp_wgt =
    (wtype_lat == 3'b010) ? (w_addr_q[12:0] >= 13'd3072 ?
                             $signed(PROJ_W[w_addr_q[12:0] - 13'd3072]) :
                             $signed(QKV_W[w_addr_q[12:0]])) :
    (wtype_lat == 3'b100) ? $signed(FC1_W[w_addr_q[12:0]]) :
    (wtype_lat == 3'b101) ? $signed(FC2_W[w_addr_q[12:0]]) :
    16'sd0;

wire signed [15:0] attn_mlp_bias =
    (wtype_lat == 3'b010) ? (w_addr_q[12:0] >= 13'd3072 ?
                             $signed(PROJ_B[b_addr_q[4:0]]) :
                             $signed(QKV_B[b_addr_q[6:0]])) :
    (wtype_lat == 3'b100) ? $signed(FC1_B[b_addr_q]) :
    (wtype_lat == 3'b101) ? $signed(FC2_B[b_addr_q[4:0]]) :
    16'sd0;

// Final mux: norm dedicated port (2-phase layer_norm ROM); attn/mlp latched
assign wgt_i       = attn_mlp_wgt;
assign bias_i      = attn_mlp_bias;
wire signed [15:0] norm_wgt_i  = norm_wgt;
wire signed [15:0] norm_bias_i = norm_bias;

// =========================================================================
// DUT
// =========================================================================
transformer_block #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (MLP_DIM),
    .N_TOKENS  (N_TOKENS)
) u_dut (
    .clk              (clk),
    .reset            (reset),
    .start            (start),
    .wgt_i            (wgt_i),
    .bias_i           (bias_i),
    .norm_wgt_i       (norm_wgt_i),
    .norm_bias_i      (norm_bias_i),
    .wgt_addr_o       (wgt_addr_o),
    .bias_addr_o      (bias_addr_o),
    .busy             (busy),
    .done             (done),
    .sram_tok1_ceb_o  (sram_tok1_ceb_o),
    .sram_tok1_web_o  (sram_tok1_web_o),
    .sram_tok1_addr_o (sram_tok1_addr_o),
    .sram_tok1_din_o  (sram_tok1_din_o),
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
// Clock
// =========================================================================
always #(CYCLE/2.0) clk = ~clk;

// =========================================================================
// Counters
// =========================================================================
always @(posedge clk) begin
    if (reset) cycle_cnt <= 32'd0;
    else       cycle_cnt <= cycle_cnt + 32'd1;
end

// =========================================================================
// Per-stage comparison (runs after each state transition)
// =========================================================================
always @(posedge clk) prev_state <= u_dut.state;

// NORM1 check: tok1 vs GOLD_NORM1 when entering S_ATTN_FEED
always @(posedge clk) begin
    if (prev_state == 4'd2 && u_dut.state == 4'd3) begin
        chk_mism  = 32'd0;
        chk_first = 32'hFFFF_FFFF;
        for (chk_i = 0; chk_i < TOK_FLAT; chk_i = chk_i + 1) begin
            if (TOK1_MEM[chk_i] !== GOLD_NORM1[chk_i]) begin
                chk_mism = chk_mism + 32'd1;
                if (chk_first == 32'hFFFF_FFFF) chk_first = chk_i;
            end
        end
        $display("[%0t] NORM1 check @ cycle %0d:", $time, cycle_cnt);
        if (chk_mism == 0)
            $display("  [PASS] norm1 output in tok1 matches golden (%0d elems)", TOK_FLAT);
        else begin
            $display("  [FAIL] norm1 mismatch = %0d / %0d  first_bad = %0d", chk_mism, TOK_FLAT, chk_first);
            $display("         RTL = %04h  GOLD = %04h", TOK1_MEM[chk_first], GOLD_NORM1[chk_first]);
        end
    end
end

// ATTN check: tok1 vs GOLD_ATTN when entering S_RES1
always @(posedge clk) begin
    if (prev_state == 4'd4 && u_dut.state == 4'd5) begin
        chk_mism  = 32'd0;
        chk_first = 32'hFFFF_FFFF;
        for (chk_i = 0; chk_i < TOK_FLAT; chk_i = chk_i + 1) begin
            if (TOK1_MEM[chk_i] !== GOLD_ATTN[chk_i]) begin
                chk_mism = chk_mism + 32'd1;
                if (chk_first == 32'hFFFF_FFFF) chk_first = chk_i;
            end
        end
        $display("[%0t] ATTN check @ cycle %0d:", $time, cycle_cnt);
        if (chk_mism == 0)
            $display("  [PASS] attn output in tok1 matches golden (%0d elems)", TOK_FLAT);
        else begin
            $display("  [FAIL] attn mismatch = %0d / %0d  first_bad = %0d", chk_mism, TOK_FLAT, chk_first);
            $display("         RTL = %04h  GOLD = %04h", TOK1_MEM[chk_first], GOLD_ATTN[chk_first]);
        end
    end
end

// RES1 check: tok2 vs GOLD_RES1 when entering S_NORM2
always @(posedge clk) begin
    if (prev_state == 4'd5 && u_dut.state == 4'd6) begin
        chk_mism  = 32'd0;
        chk_first = 32'hFFFF_FFFF;
        for (chk_i = 0; chk_i < TOK_FLAT; chk_i = chk_i + 1) begin
            if (TOK2_MEM[chk_i] !== GOLD_RES1[chk_i]) begin
                chk_mism = chk_mism + 32'd1;
                if (chk_first == 32'hFFFF_FFFF) chk_first = chk_i;
            end
        end
        $display("[%0t] RES1 check @ cycle %0d:", $time, cycle_cnt);
        if (chk_mism == 0)
            $display("  [PASS] residual1 in tok2 matches golden (%0d elems)", TOK_FLAT);
        else begin
            $display("  [FAIL] res1 mismatch = %0d / %0d  first_bad = %0d", chk_mism, TOK_FLAT, chk_first);
            $display("         RTL = %04h  GOLD = %04h", TOK2_MEM[chk_first], GOLD_RES1[chk_first]);
        end
    end
end

// NORM2 check: Q_MEM vs GOLD_NORM2 when entering S_MLP_FEED
always @(posedge clk) begin
    if (prev_state == 4'd6 && u_dut.state == 4'd7) begin
        chk_mism  = 32'd0;
        chk_first = 32'hFFFF_FFFF;
        for (chk_i = 0; chk_i < TOK_FLAT; chk_i = chk_i + 1) begin
            if (Q_MEM[chk_i] !== GOLD_NORM2[chk_i]) begin
                chk_mism = chk_mism + 32'd1;
                if (chk_first == 32'hFFFF_FFFF) chk_first = chk_i;
            end
        end
        $display("[%0t] NORM2 check @ cycle %0d:", $time, cycle_cnt);
        if (chk_mism == 0)
            $display("  [PASS] norm2 output in sram_q matches golden (%0d elems)", TOK_FLAT);
        else begin
            $display("  [FAIL] norm2 mismatch = %0d / %0d  first_bad = %0d", chk_mism, TOK_FLAT, chk_first);
            $display("         RTL = %04h  GOLD = %04h", Q_MEM[chk_first], GOLD_NORM2[chk_first]);
        end
    end
end

// MLP check: Q_MEM vs GOLD_MLP when entering S_RES2
always @(posedge clk) begin
    if (prev_state == 4'd8 && u_dut.state == 4'd9) begin
        chk_mism  = 32'd0;
        chk_first = 32'hFFFF_FFFF;
        for (chk_i = 0; chk_i < TOK_FLAT; chk_i = chk_i + 1) begin
            if (Q_MEM[chk_i] !== GOLD_MLP[chk_i]) begin
                chk_mism = chk_mism + 32'd1;
                if (chk_first == 32'hFFFF_FFFF) chk_first = chk_i;
            end
        end
        $display("[%0t] MLP check @ cycle %0d:", $time, cycle_cnt);
        if (chk_mism == 0)
            $display("  [PASS] mlp output in sram_q matches golden (%0d elems)", TOK_FLAT);
        else begin
            $display("  [FAIL] mlp mismatch = %0d / %0d  first_bad = %0d", chk_mism, TOK_FLAT, chk_first);
            $display("         RTL = %04h  GOLD = %04h", Q_MEM[chk_first], GOLD_MLP[chk_first]);
        end
    end
end

// BLOCK check: tok1 vs GOLD_BLOCK on done
always @(posedge clk) begin
    if (done) begin
        chk_mism  = 32'd0;
        chk_first = 32'hFFFF_FFFF;
        for (chk_i = 0; chk_i < TOK_FLAT; chk_i = chk_i + 1) begin
            if (TOK1_MEM[chk_i] !== GOLD_BLOCK[chk_i]) begin
                chk_mism = chk_mism + 32'd1;
                if (chk_first == 32'hFFFF_FFFF) chk_first = chk_i;
            end
        end
        $display("[%0t] BLOCK check @ cycle %0d:", $time, cycle_cnt);
        if (chk_mism == 0)
            $display("  [PASS] block output in tok1 matches golden (%0d elems)", TOK_FLAT);
        else begin
            $display("  [FAIL] block mismatch = %0d / %0d  first_bad = %0d", chk_mism, TOK_FLAT, chk_first);
            $display("         RTL = %04h  GOLD = %04h", TOK1_MEM[chk_first], GOLD_BLOCK[chk_first]);
        end
    end
end

// =========================================================================
// Main test sequence
// =========================================================================
integer load_i;

initial begin
    $fsdbDumpfile("transformer_block.fsdb");
    $fsdbDumpvars;

    $set_toggle_region("u_dut");
    $toggle_start();

    // Load golden activation data
    $readmemb({`GOLDEN_ACT, "/merged_tokens_bi.txt"},                              INPUT_MEM);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"},           GOLD_NORM1);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_attn_attn_out_bi.txt"},       GOLD_ATTN);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_residual_add1_out_bi.txt"},   GOLD_RES1);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm2_out_bi.txt"},           GOLD_NORM2);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_mlp_after_mlp_out_bi.txt"},         GOLD_MLP);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_block_out_bi.txt"},           GOLD_BLOCK);

    // Load weight / bias ROMs
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm1_weight_bi.txt"}, NORM1_W);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm1_bias_bi.txt"},   NORM1_B);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm2_weight_bi.txt"}, NORM2_W);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm2_bias_bi.txt"},   NORM2_B);

    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_weight_bi.txt"},  QKV_W);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_bias_bi.txt"},    QKV_B);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_proj_weight_bi.txt"}, PROJ_W);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_proj_bias_bi.txt"},   PROJ_B);

    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_weight_bi.txt"}, FC1_W);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_bias_bi.txt"},   FC1_B);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_weight_bi.txt"}, FC2_W);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_bias_bi.txt"},   FC2_B);

    // Pre-load tok1 with input tokens
    for (load_i = 0; load_i < TOK_FLAT; load_i = load_i + 1)
        TOK1_MEM[load_i] = INPUT_MEM[load_i];

    // Clear other SRAMs
    for (load_i = 0; load_i < TOK_FLAT; load_i = load_i + 1) begin
        TOK2_MEM[load_i] = 16'd0;
        Q_MEM[load_i]    = 16'd0;
    end
    for (load_i = 0; load_i < K_MEM_DEPTH; load_i = load_i + 1)
        K_MEM[load_i] = 16'd0;
    for (load_i = 0; load_i < V_MEM_DEPTH; load_i = load_i + 1)
        V_MEM[load_i] = 16'd0;
    for (load_i = 0; load_i < QKM_MEM_DEPTH; load_i = load_i + 1)
        QKM_MEM[load_i] = 16'd0;

    $display("[TB] transformer_block block0 test (N_TOKENS=%0d)", N_TOKENS);
    $display("[TB] Golden loaded from %s", `GOLDEN_ACT);

    clk   = 0;
    reset = 1;
    start = 0;
    cycle_cnt = 0;

    #(CYCLE * 2) reset = 0;
    #(CYCLE);

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    $display("[TB] start pulsed @ cycle %0d", cycle_cnt);

    wait (done === 1'b1);
    @(posedge clk);

    $display("\n==== transformer_block done @ cycle %0d ====", cycle_cnt);

    $toggle_stop();
    $toggle_report("transformer_block_rtl.saif", 1.0e-9, "u_dut");

    #(CYCLE * 5);
    $finish;
end

// =========================================================================
// Timeout
// =========================================================================
initial begin
    #(CYCLE * 10_000_000);
    $display("[TB] TIMEOUT @ cycle %0d  state=%0d", cycle_cnt, u_dut.state);
    $toggle_stop();
    $toggle_report("transformer_block_rtl.saif", 1.0e-9, "u_dut");
    $finish;
end

endmodule
