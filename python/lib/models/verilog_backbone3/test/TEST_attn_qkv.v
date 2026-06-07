`timescale 1ns/10ps

`include "common/vec_mac8.v"
`include "linear_vec8.v"
`include "care_attention/attn_qkv.v"

// =============================================================================
// TEST_attn_qkv.v -- block0 QKV unit test (token 0 default)
//
// DUT: attn_qkv EMBED_DIM=32 N_TOKENS=1 (single-token slice).
// Golden (numpy trunk, full tensor shape H,N,d then slice token TOK_IDX):
//   x   : Activation/backbone_blocks_0_after_norm1_out_bi.txt
//   Q   : Activation/backbone_blocks_0_attn_after_qkv_q_bi.txt
//   K   : Activation/backbone_blocks_0_attn_after_qkv_k_bi.txt
//   V   : Activation/backbone_blocks_0_attn_after_qkv_v_bi.txt
//   W/B : Weight/backbone_blocks_0_attn_qkv_{weight,bias}_bi.txt
//
// norm1 SRAM model: addr @ posedge T -> data valid @ posedge T+1 (negedge latch).
// Q/K/V SRAM model: write @ posedge when ceb=0 web=0 (same beat as DUT drive).
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_attn_qkv.v +incdir+.. +incdir+../common +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|attn_qkv done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_attn_qkv;

parameter CYCLE     = 2.0;
parameter EMBED_DIM = 32;
parameter NUM_HEADS = 4;
parameter HEAD_DIM  = 8;
parameter N_TOKENS  = 1;
parameter TOK_IDX   = 0;
parameter TOK_BASE  = TOK_IDX * EMBED_DIM;

parameter N_TOKENS_GOLD = 320;
parameter HD_ELEMS      = NUM_HEADS * N_TOKENS_GOLD * HEAD_DIM;
parameter TOK_HD_ELEMS  = NUM_HEADS * HEAD_DIM;

parameter QKV_W_DEPTH = 96 * EMBED_DIM;
parameter QKV_B_DEPTH = 96;

reg         clk;
reg         reset;
reg         start;

wire        norm_rd_en;
wire [13:0] norm_rd_flat;
wire [12:0] wgt_addr_o;
wire        busy;
wire        done;
wire signed [15:0] norm_x;
wire signed [15:0] wgt_i;
wire signed [15:0] bias_i;

wire        sram_q_ceb_o;
wire        sram_q_web_o;
wire [13:0] sram_q_addr_o;
wire [15:0] sram_q_din_o;
wire        sram_k_ceb_o;
wire        sram_k_web_o;
wire [13:0] sram_k_addr_o;
wire [15:0] sram_k_din_o;
wire        sram_v_ceb_o;
wire        sram_v_web_o;
wire [13:0] sram_v_addr_o;
wire [15:0] sram_v_din_o;

reg [15:0] NORM1_ALL [0:10239];
reg [15:0] Q_GOLD_ALL [0:HD_ELEMS-1];
reg [15:0] K_GOLD_ALL [0:HD_ELEMS-1];
reg [15:0] V_GOLD_ALL [0:HD_ELEMS-1];
reg [15:0] GOLD_Q [0:TOK_HD_ELEMS-1];
reg [15:0] GOLD_K [0:TOK_HD_ELEMS-1];
reg [15:0] GOLD_V [0:TOK_HD_ELEMS-1];
reg [15:0] W_MEM [0:QKV_W_DEPTH-1];
reg [15:0] B_MEM [0:QKV_B_DEPTH-1];

reg [15:0] Q_MEM [0:TOK_HD_ELEMS-1];
reg [15:0] K_MEM [0:TOK_HD_ELEMS-1];
reg [15:0] V_MEM [0:TOK_HD_ELEMS-1];

integer load_i;
integer cmp_i;
integer h_i;
integer d_i;
integer flat_gold;
integer flat_dut;

reg [13:0] norm_addr_rom_q;
reg [12:0] w_addr_rom_q;
reg [7:0]  b_addr_rom_q;

reg [31:0] mism_q;
reg [31:0] mism_k;
reg [31:0] mism_v;
reg [31:0] first_bad_q;
reg [31:0] first_bad_k;
reg [31:0] first_bad_v;
reg [31:0] cycle_cnt;

wire [7:0] vm_b_addr_o;

attn_qkv #(
    .EMBED_DIM (EMBED_DIM),
    .NUM_HEADS (NUM_HEADS),
    .HEAD_DIM  (HEAD_DIM),
    .N_TOKENS  (N_TOKENS),
    .QKV_OUT   (96)
) u_dut (
    .clk            (clk),
    .reset          (reset),
    .start          (start),
    .norm_rd_en     (norm_rd_en),
    .norm_rd_flat   (norm_rd_flat),
    .norm_x         (norm_x),
    .wgt_i          (wgt_i),
    .bias_i         (bias_i),
    .wgt_addr_o     (wgt_addr_o),
    .busy           (busy),
    .done           (done),
    .sram_q_ceb_o   (sram_q_ceb_o),
    .sram_q_web_o   (sram_q_web_o),
    .sram_q_addr_o  (sram_q_addr_o),
    .sram_q_din_o   (sram_q_din_o),
    .sram_k_ceb_o   (sram_k_ceb_o),
    .sram_k_web_o   (sram_k_web_o),
    .sram_k_addr_o  (sram_k_addr_o),
    .sram_k_din_o   (sram_k_din_o),
    .sram_v_ceb_o   (sram_v_ceb_o),
    .sram_v_web_o   (sram_v_web_o),
    .sram_v_addr_o  (sram_v_addr_o),
    .sram_v_din_o   (sram_v_din_o)
);

assign vm_b_addr_o = u_dut.u_lin_qkv.u_vec_mac8.b_addr_o;

assign norm_x = $signed(NORM1_ALL[norm_addr_rom_q]);
assign wgt_i  = $signed(W_MEM[w_addr_rom_q]);
assign bias_i = $signed(B_MEM[b_addr_rom_q]);

always #(CYCLE/2.0) clk = ~clk;

always @(negedge clk) begin
    if (norm_rd_en)
        norm_addr_rom_q <= norm_rd_flat;
    w_addr_rom_q <= wgt_addr_o;
    b_addr_rom_q <= vm_b_addr_o;
end

// Behavioral SRAM: latch write on posedge (DUT drives addr/din same beat)
always @(posedge clk) begin
    if (!sram_q_ceb_o && !sram_q_web_o)
        Q_MEM[sram_q_addr_o] <= sram_q_din_o;
    if (!sram_k_ceb_o && !sram_k_web_o)
        K_MEM[sram_k_addr_o] <= sram_k_din_o;
    if (!sram_v_ceb_o && !sram_v_web_o)
        V_MEM[sram_v_addr_o] <= sram_v_din_o;
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"}, NORM1_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_q_bi.txt"}, Q_GOLD_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_k_bi.txt"}, K_GOLD_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_v_bi.txt"}, V_GOLD_ALL);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_weight_bi.txt"}, W_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_bias_bi.txt"},   B_MEM);

    for (load_i = 0; load_i < TOK_HD_ELEMS; load_i = load_i + 1) begin
        h_i = load_i / HEAD_DIM;
        d_i = load_i % HEAD_DIM;
        flat_gold = h_i * N_TOKENS_GOLD * HEAD_DIM + TOK_IDX * HEAD_DIM + d_i;
        GOLD_Q[load_i] = Q_GOLD_ALL[flat_gold];
        GOLD_K[load_i] = K_GOLD_ALL[flat_gold];
        GOLD_V[load_i] = V_GOLD_ALL[flat_gold];
    end

    for (load_i = 0; load_i < TOK_HD_ELEMS; load_i = load_i + 1) begin
        Q_MEM[load_i] = 16'hxxxx;
        K_MEM[load_i] = 16'hxxxx;
        V_MEM[load_i] = 16'hxxxx;
    end

    $display("[TB] attn_qkv block0 token %0d (N_TOKENS=%0d)", TOK_IDX, N_TOKENS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk             = 1'b0;
    reset           = 1'b1;
    start           = 1'b0;
    norm_addr_rom_q = 14'd0;
    w_addr_rom_q    = 13'd0;
    b_addr_rom_q    = 8'd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    mism_q      = 32'd0;
    mism_k      = 32'd0;
    mism_v      = 32'd0;
    first_bad_q = 32'hFFFF_FFFF;
    first_bad_k = 32'hFFFF_FFFF;
    first_bad_v = 32'hFFFF_FFFF;

    for (cmp_i = 0; cmp_i < TOK_HD_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_Q[cmp_i] !== Q_MEM[cmp_i]) begin
            mism_q = mism_q + 32'd1;
            if (first_bad_q == 32'hFFFF_FFFF)
                first_bad_q = cmp_i;
        end
        if (GOLD_K[cmp_i] !== K_MEM[cmp_i]) begin
            mism_k = mism_k + 32'd1;
            if (first_bad_k == 32'hFFFF_FFFF)
                first_bad_k = cmp_i;
        end
        if (GOLD_V[cmp_i] !== V_MEM[cmp_i]) begin
            mism_v = mism_v + 32'd1;
            if (first_bad_v == 32'hFFFF_FFFF)
                first_bad_v = cmp_i;
        end
    end

    $display("---- attn_qkv done @ cycle %0d ----", cycle_cnt);
    $display("[TB] q_mism=%0d k_mism=%0d v_mism=%0d expect_each=%0d",
             mism_q, mism_k, mism_v, TOK_HD_ELEMS);

    if (mism_q !== 0)
        $display("  [FAIL] q_sram mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_q, first_bad_q, Q_MEM[first_bad_q], GOLD_Q[first_bad_q]);
    else if (mism_k !== 0)
        $display("  [FAIL] k_sram mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_k, first_bad_k, K_MEM[first_bad_k], GOLD_K[first_bad_k]);
    else if (mism_v !== 0)
        $display("  [FAIL] v_sram mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_v, first_bad_v, V_MEM[first_bad_v], GOLD_V[first_bad_v]);
    else
        $display("  [PASS] attn_qkv matches backbone_blocks_0_attn_after_qkv_q/k/v");

    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d", cycle_cnt, busy);
    $finish;
end

endmodule
