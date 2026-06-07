`timescale 1ns/10ps

`include "common/vec_mac8.v"
`include "linear_vec8.v"
`include "mlp_vec8.v"

// =============================================================================
// TEST_mlp_vec8.v -- block0 MLP unit test (token 0 default)
//
// DUT: mlp_vec8 EMBED_DIM=32 MLP_DIM=128 N_TOKENS=1.
// Golden (numpy trunk):
//   x   : Activation/backbone_blocks_0_after_norm2_out_bi.txt (token-major)
//   y   : Activation/backbone_blocks_0_mlp_after_mlp_out_bi.txt
//   W/B : Weight/backbone_blocks_0_mlp_fc{1,2}_{weight,bias}_bi.txt
//
// norm2 SRAM model: addr @ posedge T -> data valid @ posedge T+1 (negedge latch).
// Weight ROM model: addr @ posedge T -> data valid @ posedge T+1 (negedge latch).
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs -f filelist_mlp.f +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   vcs TEST_mlp_vec8.v +incdir+.. +incdir+../common +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|mlp_vec8 done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_mlp_vec8;

parameter CYCLE     = 2.0;
parameter EMBED_DIM = 32;
parameter MLP_DIM   = 128;
parameter N_TOKENS  = 1;
parameter TOK_IDX   = 0;
parameter TOK_BASE  = TOK_IDX * EMBED_DIM;

parameter FC1_W_DEPTH = MLP_DIM * EMBED_DIM;
parameter FC1_B_DEPTH = MLP_DIM;
parameter FC2_W_DEPTH = EMBED_DIM * MLP_DIM;
parameter FC2_B_DEPTH = EMBED_DIM;

reg         clk;
reg         reset;
reg         start;

wire        norm_rd_en;
wire [13:0] norm_rd_flat;
wire [15:0] wgt_addr_o;
wire        busy;
wire        done;
wire signed [15:0] norm_x;
wire signed [15:0] wgt_i;
wire signed [15:0] bias_i;
wire signed [15:0] y_o;
wire        y_valid;

reg [15:0] GOLD_Y [0:EMBED_DIM-1];
reg [15:0] NORM2_ALL [0:10239];
reg [15:0] MLP_ALL  [0:10239];
reg [15:0] W_FC1 [0:FC1_W_DEPTH-1];
reg [15:0] B_FC1 [0:FC1_B_DEPTH-1];
reg [15:0] W_FC2 [0:FC2_W_DEPTH-1];
reg [15:0] B_FC2 [0:FC2_B_DEPTH-1];

integer load_i;

reg [13:0] norm_addr_rom_q;
reg [12:0] w_addr_rom_q;
reg [7:0]  b_addr_rom_q;

wire [2:0] wtype;

reg [5:0]  y_recv_cnt;
reg [31:0] mism_cnt;
reg [31:0] first_bad_feat;
reg [31:0] cycle_cnt;

wire [7:0] fc1_b_addr_o;
wire [7:0] fc2_b_addr_o;
wire [7:0] vm_b_addr_o;

mlp_vec8 #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (MLP_DIM),
    .N_TOKENS  (N_TOKENS)
) u_dut (
    .clk          (clk),
    .reset        (reset),
    .start        (start),
    .norm_rd_en   (norm_rd_en),
    .norm_rd_flat (norm_rd_flat),
    .norm_x       (norm_x),
    .wgt_i        (wgt_i),
    .bias_i       (bias_i),
    .wgt_addr_o   (wgt_addr_o),
    .busy         (busy),
    .done         (done),
    .y_o          (y_o),
    .y_valid      (y_valid)
);

assign fc1_b_addr_o = u_dut.u_fc1.u_vec_mac8.b_addr_o;
assign fc2_b_addr_o = u_dut.u_fc2.u_vec_mac8.b_addr_o;
assign vm_b_addr_o  = (u_dut.state == 4'd1) ? fc1_b_addr_o : fc2_b_addr_o;

assign wtype = wgt_addr_o[15:13];

assign norm_x = $signed(NORM2_ALL[norm_addr_rom_q]);
assign wgt_i  = (wtype == 3'b100) ? $signed(W_FC1[w_addr_rom_q]) :
                (wtype == 3'b101) ? $signed(W_FC2[w_addr_rom_q]) :
                16'sd0;
assign bias_i = (wtype == 3'b100) ? $signed(B_FC1[b_addr_rom_q]) :
                (wtype == 3'b101) ? $signed(B_FC2[b_addr_rom_q[4:0]]) :
                16'sd0;

always #(CYCLE/2.0) clk = ~clk;

always @(negedge clk) begin
    if (norm_rd_en)
        norm_addr_rom_q <= norm_rd_flat;
    w_addr_rom_q <= wgt_addr_o[12:0];
    b_addr_rom_q <= vm_b_addr_o;
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

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
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm2_out_bi.txt"}, NORM2_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_mlp_after_mlp_out_bi.txt"}, MLP_ALL);
    for (load_i = 0; load_i < EMBED_DIM; load_i = load_i + 1)
        GOLD_Y[load_i] = MLP_ALL[TOK_BASE + load_i];
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_weight_bi.txt"}, W_FC1);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_bias_bi.txt"},   B_FC1);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_weight_bi.txt"}, W_FC2);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_bias_bi.txt"},   B_FC2);

    $display("[TB] mlp_vec8 block0 token %0d (N_TOKENS=%0d)", TOK_IDX, N_TOKENS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk              = 1'b0;
    reset            = 1'b1;
    start            = 1'b0;
    norm_addr_rom_q  = 14'd0;
    w_addr_rom_q     = 13'd0;
    b_addr_rom_q     = 8'd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- mlp_vec8 done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d",
             y_recv_cnt, EMBED_DIM, mism_cnt);

    if (y_recv_cnt !== EMBED_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, EMBED_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] mlp_out mismatch count=%0d first_bad_feat=%0d",
                 mism_cnt, first_bad_feat);
    else
        $display("  [PASS] mlp_vec8 matches backbone_blocks_0_mlp_after_mlp_out");

    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d",
             cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
