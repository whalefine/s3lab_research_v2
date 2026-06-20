`timescale 1ns/10ps

`include "common/vec_mac8.v"
`include "linear_vec8.v"

// =============================================================================
// TEST_linear_vec8.v -- block0 QKV linear_vec8 unit test (token 0 default)
//
// DUT: linear_vec8 IN_DIM=32 OUT_DIM=96 (QKV).
// Golden (numpy trunk):
//   x   : Activation/backbone_blocks_0_after_norm1_out_bi.txt  (token-major)
//   y   : Activation/backbone_blocks_0_attn_after_qkv_linear_out_bi.txt
//   W/B : Weight/backbone_blocks_0_attn_qkv_{weight,bias}_bi.txt
//
// ROM model: CLK(clk) posedge registered-data. addr (combinational) present in
//   cycle T -> Q registered at posedge -> w_i/b_i valid in cycle T+1.
//
// Default golden paths (sim cwd = verilog_backbone3/test):
//   ../../../../output/golden/vit_care_relu6_numpy_trunk_dim32_out/...
// Override: +define+GOLDEN_ACT=... +define+GOLDEN_WGT=...
// Or symlink: ./TXT_File/Activation ./TXT_File/Weight
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_linear_vec8.v +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|linear_vec8 done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_linear_vec8;

parameter CYCLE    = 2.0;
parameter IN_DIM   = 32;
parameter OUT_DIM  = 96;
parameter TOK_IDX  = 0;
parameter TOK_BASE = TOK_IDX * IN_DIM;
parameter Y_BASE   = TOK_IDX * OUT_DIM;

parameter W_DEPTH  = OUT_DIM * IN_DIM;
parameter B_DEPTH  = OUT_DIM;

reg         clk;
reg         reset;
reg         start;
reg  signed [15:0] x_i;
reg                x_valid;

wire signed [15:0] w_i;
wire signed [15:0] b_i;
wire [12:0]        w_addr_o;
wire [7:0]         vm_b_addr_o;
wire               busy;
wire               done;
wire signed [15:0] y_o;
wire               y_valid;
wire [6:0]         y_neu_o;

reg [15:0] GOLD_X [0:IN_DIM-1];
reg [15:0] GOLD_Y [0:OUT_DIM-1];
reg [15:0] NORM1_ALL [0:10239];
reg [15:0] QKV_ALL  [0:30719];
reg [15:0] W_MEM    [0:W_DEPTH-1];
reg [15:0] B_MEM    [0:B_DEPTH-1];

integer load_i;

// ROM model: CLK(clk) posedge registered-data (no negedge addr latch).
//   w_i/b_i <= MEM[addr@posedge] -> Q valid the next cycle, matching the real macro.
reg signed [15:0] w_i_r;
reg signed [15:0] b_i_r;

reg [5:0]  x_feed_cnt;
reg        x_feeding;

reg [7:0]  y_recv_cnt;
reg [31:0] mism_cnt;
reg [31:0] first_bad_neu;
reg [31:0] cycle_cnt;

linear_vec8 #(
    .IN_DIM  (IN_DIM),
    .OUT_DIM (OUT_DIM)
) u_dut (
    .clk      (clk),
    .reset    (reset),
    .start    (start),
    .x_i      (x_i),
    .x_valid  (x_valid),
    .w_i      (w_i),
    .b_i      (b_i),
    .w_addr_o (w_addr_o),
    .busy     (busy),
    .done     (done),
    .y_o      (y_o),
    .y_valid  (y_valid),
    .y_neu_o  (y_neu_o)
);

// b_addr_o is internal to vec_mac8 (not on linear_vec8 top port)
assign vm_b_addr_o = u_dut.u_vec_mac8.b_addr_o;

assign w_i = w_i_r;
assign b_i = b_i_r;

always #(CYCLE/2.0) clk = ~clk;

// posedge synchronous read: register MEM contents addressed in the current cycle
always @(posedge clk) begin
    w_i_r <= $signed(W_MEM[w_addr_o]);
    b_i_r <= $signed(B_MEM[vm_b_addr_o]);
end

// Stream IN_DIM x beats after start (ignored outside S_LOAD).
// Note: on the start posedge, busy is still 0 (state reg not yet S_LOAD); do not
// clear x_feeding in the same cycle as start or x never loads and DUT hangs in S_LOAD.
always @(posedge clk) begin
    x_valid <= 1'b0;
    x_i     <= 16'sd0;
    if (reset) begin
        x_feed_cnt <= 6'd0;
        x_feeding  <= 1'b0;
    end else if (!busy && !start) begin
        x_feed_cnt <= 6'd0;
        x_feeding  <= 1'b0;
    end else begin
        if (start)
            x_feeding <= 1'b1;
        if (x_feeding && busy && (x_feed_cnt < IN_DIM)) begin
            x_valid <= 1'b1;
            x_i     <= GOLD_X[x_feed_cnt];
            x_feed_cnt <= x_feed_cnt + 6'd1;
            if (x_feed_cnt == IN_DIM - 1)
                x_feeding <= 1'b0;
        end
    end
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

// Compare y on y_valid
always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt    <= 8'd0;
        mism_cnt      <= 32'd0;
        first_bad_neu <= 32'hFFFF_FFFF;
    end else if (y_valid) begin
        y_recv_cnt <= y_recv_cnt + 8'd1;
        if (GOLD_Y[y_neu_o] !== y_o[15:0]) begin
            mism_cnt <= mism_cnt + 32'd1;
            if (first_bad_neu == 32'hFFFF_FFFF)
                first_bad_neu <= {25'b0, y_neu_o};
        end
    end
end

initial begin
    // Load full golden tensors, then slice one token (VCS $readmemb start/end
    // does not skip file lines; using range caused STASKW_RMIEAFK overflow warnings).
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"}, NORM1_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_linear_out_bi.txt"}, QKV_ALL);
    for (load_i = 0; load_i < IN_DIM; load_i = load_i + 1)
        GOLD_X[load_i] = NORM1_ALL[TOK_BASE + load_i];
    for (load_i = 0; load_i < OUT_DIM; load_i = load_i + 1)
        GOLD_Y[load_i] = QKV_ALL[Y_BASE + load_i];
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_weight_bi.txt"}, W_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_bias_bi.txt"},   B_MEM);

    $display("[TB] linear_vec8 QKV token %0d", TOK_IDX);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk          = 1'b0;
    reset        = 1'b1;
    start        = 1'b0;
    w_i_r        = 16'sd0;
    b_i_r        = 16'sd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- linear_vec8 done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d",
             y_recv_cnt, OUT_DIM, mism_cnt);

    if (y_recv_cnt !== OUT_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, OUT_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] qkv_linear_out mismatch count=%0d first_bad_neu=%0d",
                 mism_cnt, first_bad_neu);
    else
        $display("  [PASS] linear_vec8 matches backbone_blocks_0_attn_after_qkv_linear_out");

    $finish;
end

initial begin
    // QKV token0 ~ few x10k cycles; 500k is ample margin
    #(CYCLE * 500_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d",
             cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
