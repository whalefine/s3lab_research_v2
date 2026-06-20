`timescale 1ns/10ps

`include "inv_sqrt_lut_seed.v"
`include "inv_sqrt_nr.v"
`include "layer_norm_pip.v"

// =============================================================================
// TEST_layer_norm_pip.v -- block0 norm1 unit test (token 0 default)
//
// DUT: layer_norm_pip FEAT_DIM=32.
// Golden (numpy trunk):
//   x   : Activation/merged_tokens_bi.txt (block0 norm1 input, token-major)
//   y   : Activation/backbone_blocks_0_after_norm1_out_bi.txt
//   W/B : Weight/backbone_blocks_0_norm1_{weight,bias}_bi.txt
//
// Memory models: CLK=clk posedge registered-data (no negedge latch).
//   x SRAM : x_rd_flat @ posedge T -> x_i @ T+1 (matches S_LOAD pend/wait window).
//   W/B ROM: feat_addr_o @ posedge T -> w_i/b_i @ T+1 (P0 addr -> w/b valid P1 -> y P2).
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_layer_norm_pip.v +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|layer_norm done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_layer_norm_pip;

parameter CYCLE    = 2.0;
parameter FEAT_DIM = 32;
parameter TOK_IDX  = 0;
parameter TOK_BASE = TOK_IDX * FEAT_DIM;

reg         clk;
reg         reset;
reg         start;
reg  [13:0] token_base_flat;

wire        x_rd_en;
wire [13:0] x_rd_flat;
wire [9:0]  feat_addr_o;
wire        busy;
wire        done;
wire signed [15:0] x_i;
wire signed [15:0] w_i;
wire signed [15:0] b_i;
wire signed [15:0] y_o;
wire        y_valid;

reg [15:0] GOLD_Y [0:FEAT_DIM-1];
reg [15:0] MERGED_ALL [0:10239];
reg [15:0] NORM1_ALL  [0:10239];
reg [15:0] W_MEM [0:FEAT_DIM-1];
reg [15:0] B_MEM [0:FEAT_DIM-1];

integer load_i;

// CLK=clk posedge registered-data read outputs (no negedge latch)
reg signed [15:0] x_i_r;
reg signed [15:0] w_i_r;
reg signed [15:0] b_i_r;

reg [5:0]  y_recv_cnt;
reg [31:0] mism_cnt;
reg [31:0] first_bad_feat;
reg [31:0] cycle_cnt;

layer_norm_pip #(
    .FEAT_DIM  (FEAT_DIM),
    .FEAT_AW   (5),
    .RCP_NUM   (65536 / FEAT_DIM)
) u_dut (
    .clk             (clk),
    .reset           (reset),
    .start           (start),
    .token_base_flat (token_base_flat),
    .x_rd_en         (x_rd_en),
    .x_rd_flat       (x_rd_flat),
    .x_i             (x_i),
    .w_i             (w_i),
    .b_i             (b_i),
    .feat_addr_o     (feat_addr_o),
    .busy            (busy),
    .done            (done),
    .y_o             (y_o),
    .y_valid         (y_valid),
    .x_rd_pend_o     (),
    .x_rd_wait_o     ()
);

// CLK=clk posedge registered-data read models (1-cycle latency: addr@T -> q@T+1)
assign x_i = x_i_r;
assign w_i = w_i_r;
assign b_i = b_i_r;

always #(CYCLE/2.0) clk = ~clk;

// posedge synchronous read: register memory contents addressed in the current cycle
always @(posedge clk) begin
    x_i_r <= $signed(MERGED_ALL[x_rd_flat]);
    w_i_r <= $signed(W_MEM[feat_addr_o[4:0]]);
    b_i_r <= $signed(B_MEM[feat_addr_o[4:0]]);
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

// y_valid beats are in-order feat 0..FEAT_DIM-1
always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt     <= 6'd0;
        mism_cnt       <= 32'd0;
        first_bad_feat <= 32'hFFFF_FFFF;
    end else if (y_valid) begin
        if (GOLD_Y[y_recv_cnt[4:0]] !== y_o[15:0]) begin
            mism_cnt <= mism_cnt + 32'd1;
            if (first_bad_feat == 32'hFFFF_FFFF)
                first_bad_feat <= {26'b0, y_recv_cnt[4:0]};
        end
        y_recv_cnt <= y_recv_cnt + 6'd1;
    end
end

initial begin
    $readmemb({`GOLDEN_ACT, "/merged_tokens_bi.txt"}, MERGED_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"}, NORM1_ALL);
    for (load_i = 0; load_i < FEAT_DIM; load_i = load_i + 1)
        GOLD_Y[load_i] = NORM1_ALL[TOK_BASE + load_i];
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm1_weight_bi.txt"}, W_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm1_bias_bi.txt"},   B_MEM);

    $display("[TB] layer_norm_pip block0 norm1 token %0d", TOK_IDX);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk            = 1'b0;
    reset          = 1'b1;
    start          = 1'b0;
    token_base_flat = TOK_BASE[13:0];
    x_i_r = 16'sd0;
    w_i_r = 16'sd0;
    b_i_r = 16'sd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- layer_norm done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d",
             y_recv_cnt, FEAT_DIM, mism_cnt);

    if (y_recv_cnt !== FEAT_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, FEAT_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] norm1_out mismatch count=%0d first_bad_feat=%0d",
                 mism_cnt, first_bad_feat);
    else
        $display("  [PASS] layer_norm_pip matches backbone_blocks_0_after_norm1_out");

    $finish;
end

initial begin
    #(CYCLE * 200_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d",
             cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
