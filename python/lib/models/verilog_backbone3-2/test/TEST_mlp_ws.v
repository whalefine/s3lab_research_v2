`timescale 1ns/10ps

`include "mlp_ws.v"

// =============================================================================
// TEST_mlp_ws.v -- block0 Weight-Stationary MLP unit test
//
// DUT: mlp_ws EMBED_DIM=32 MLP_DIM=128 (default N_TOKENS=320).
// Golden (numpy trunk):
//   x   : Activation/backbone_blocks_0_after_norm2_out_bi.txt (token-major)
//   y   : Activation/backbone_blocks_0_mlp_after_mlp_out_bi.txt
//   W/B : Weight/backbone_blocks_0_mlp_fc{1,2}_{weight,bias}_bi.txt
//
// norm2 SRAM model: addr @ posedge T -> data valid @ posedge T+1 (CLK(clk) regd-data).
// Weight ROM model: addr @ posedge T -> data valid @ posedge T+1 (CLK(clk) regd-data).
// k/v/qkm/ao SRAM:  same posedge 1-cycle read latency as care_attention TB.
//
// Quick smoke (1 token):
//   vcs TEST_mlp_ws.v +incdir+.. +define+MLP_WS_N_TOKENS=1 ...
//
// VCS (full 320 tokens):
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_mlp_ws.v +incdir+.. +lint=TFIPC-L +define+TSMC_CM_NO_WARNING \
//     -debug_access+all -debug_region+cell | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|mlp_ws done' simv.log
//
// Waveform / power (written on done or timeout):
//   mlp_ws.fsdb  (Verdi)
//   mlp_ws_rtl.saif  (SAIF toggle, scope u_dut)
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_mlp_ws;

parameter CYCLE     = 2.0;
parameter EMBED_DIM = 32;
parameter MLP_DIM   = 128;
`ifdef MLP_WS_N_TOKENS
parameter N_TOKENS  = `MLP_WS_N_TOKENS;
`else
parameter N_TOKENS  = 320;
`endif
parameter OUT_TOTAL = N_TOKENS * EMBED_DIM;

parameter FC1_W_DEPTH = MLP_DIM * EMBED_DIM;
parameter FC1_B_DEPTH = MLP_DIM;
parameter FC2_W_DEPTH = EMBED_DIM * MLP_DIM;
parameter FC2_B_DEPTH = EMBED_DIM;

parameter K_MEM_DEPTH  = 128 * MLP_DIM;
parameter V_MEM_DEPTH  = 128 * MLP_DIM;
parameter QKM_MEM_DEPTH = 64 * MLP_DIM;
parameter AO_MEM_DEPTH  = N_TOKENS * EMBED_DIM;

reg         clk;
reg         reset;
reg         start;

wire        norm_rd_en;
wire [13:0] norm_rd_flat;
wire [15:0] wgt_addr_o;
wire [7:0]  bias_addr_o;
wire        busy;
wire        done;
wire signed [15:0] norm_x;
wire signed [15:0] wgt_i;
wire signed [15:0] bias_i;
wire signed [15:0] y_o;
wire        y_valid;

wire        sram_k_ceb_o;
wire        sram_k_web_o;
wire [13:0] sram_k_addr_o;
wire [15:0] sram_k_din_o;
wire [15:0] sram_k_q_i;

wire        sram_v_ceb_o;
wire        sram_v_web_o;
wire [13:0] sram_v_addr_o;
wire [15:0] sram_v_din_o;
wire [15:0] sram_v_q_i;

wire        sram_qkm_ceb_o;
wire        sram_qkm_web_o;
wire [13:0] sram_qkm_addr_o;
wire [15:0] sram_qkm_din_o;
wire [15:0] sram_qkm_q_i;

wire        sram_ao_ceb_o;
wire        sram_ao_web_o;
wire [13:0] sram_ao_addr_o;
wire [15:0] sram_ao_din_o;
wire [15:0] sram_ao_q_i;

reg [15:0] NORM2_ALL [0:10239];
reg [15:0] MLP_ALL  [0:10239];
reg [15:0] W_FC1 [0:FC1_W_DEPTH-1];
reg [15:0] B_FC1 [0:FC1_B_DEPTH-1];
reg [15:0] W_FC2 [0:FC2_W_DEPTH-1];
reg [15:0] B_FC2 [0:FC2_B_DEPTH-1];

reg [15:0] K_MEM  [0:K_MEM_DEPTH-1];
reg [15:0] V_MEM  [0:V_MEM_DEPTH-1];
reg [15:0] QKM_MEM [0:QKM_MEM_DEPTH-1];
reg [15:0] AO_MEM [0:AO_MEM_DEPTH-1];

integer load_i;

// CLK(clk) posedge registered-data macro model: Q registers the addressed word on
// posedge clk, so Q is valid the cycle after the address (addr@T -> Q@T+1).
reg signed [15:0] norm_x_r;
reg signed [15:0] wgt_i_r;
reg signed [15:0] bias_i_r;

reg [15:0] sram_k_q_r;
reg [15:0] sram_v_q_r;
reg [15:0] sram_qkm_q_r;
reg [15:0] sram_ao_q_r;

reg [31:0] y_recv_cnt;
reg [31:0] mism_cnt;
reg [31:0] first_bad_idx;
reg [31:0] cycle_cnt;
reg [15:0] first_bad_rtl;
reg [15:0] first_bad_gold;

mlp_ws #(
    .EMBED_DIM (EMBED_DIM),
    .MLP_DIM   (MLP_DIM),
    .N_TOKENS  (N_TOKENS)
) u_dut (
    .clk             (clk),
    .reset           (reset),
    .start           (start),
    .norm_rd_en      (norm_rd_en),
    .norm_rd_flat    (norm_rd_flat),
    .norm_x          (norm_x),
    .wgt_i           (wgt_i),
    .bias_i          (bias_i),
    .wgt_addr_o      (wgt_addr_o),
    .bias_addr_o     (bias_addr_o),
    .sram_k_ceb_o    (sram_k_ceb_o),
    .sram_k_web_o    (sram_k_web_o),
    .sram_k_addr_o   (sram_k_addr_o),
    .sram_k_din_o    (sram_k_din_o),
    .sram_k_q_i      (sram_k_q_i),
    .sram_v_ceb_o    (sram_v_ceb_o),
    .sram_v_web_o    (sram_v_web_o),
    .sram_v_addr_o   (sram_v_addr_o),
    .sram_v_din_o    (sram_v_din_o),
    .sram_v_q_i      (sram_v_q_i),
    .sram_qkm_ceb_o  (sram_qkm_ceb_o),
    .sram_qkm_web_o  (sram_qkm_web_o),
    .sram_qkm_addr_o (sram_qkm_addr_o),
    .sram_qkm_din_o  (sram_qkm_din_o),
    .sram_qkm_q_i    (sram_qkm_q_i),
    .sram_ao_ceb_o   (sram_ao_ceb_o),
    .sram_ao_web_o   (sram_ao_web_o),
    .sram_ao_addr_o  (sram_ao_addr_o),
    .sram_ao_din_o   (sram_ao_din_o),
    .sram_ao_q_i     (sram_ao_q_i),
    .busy            (busy),
    .done            (done),
    .y_o             (y_o),
    .y_valid         (y_valid)
);

assign norm_x = norm_x_r;
assign wgt_i  = wgt_i_r;
assign bias_i = bias_i_r;

assign sram_k_q_i   = sram_k_q_r;
assign sram_v_q_i   = sram_v_q_r;
assign sram_qkm_q_i = sram_qkm_q_r;
assign sram_ao_q_i  = sram_ao_q_r;

always #(CYCLE/2.0) clk = ~clk;

// CLK(clk) posedge registered-data ROM / norm2 model: sample current combinational
// address -> Q valid next cycle. WLOAD/BLOAD present a stable addr across the 2-phase
// window, so the phase-1 consume sees ROM[addr@phase0].
always @(posedge clk) begin
    if (norm_rd_en)
        norm_x_r <= $signed(NORM2_ALL[norm_rd_flat]);
    if (u_dut.state == 4'd1)
        wgt_i_r <= (wgt_addr_o[15:13] == 3'b100) ? $signed(W_FC1[wgt_addr_o[12:0]]) :
                   (wgt_addr_o[15:13] == 3'b101) ? $signed(W_FC2[wgt_addr_o[12:0]]) :
                   16'sd0;
    // BLOAD phase0: sample bias ROM (incl. addr 0) so phase1 capture sees the new word.
    if (u_dut.state == 4'd2 && u_dut.bl_phase == 1'b0)
        bias_i_r <= (wgt_addr_o[15:13] == 3'b100) ? $signed(B_FC1[bias_addr_o]) :
                    (wgt_addr_o[15:13] == 3'b101) ? $signed(B_FC2[bias_addr_o[4:0]]) :
                    16'sd0;
end

// CLK(clk) posedge registered-data SRAM read: on read (CEB=0,WEB=1) latch MEM[addr];
// hold otherwise. Q valid the cycle after the address.
always @(posedge clk) begin
    if (!sram_k_ceb_o && sram_k_web_o)
        sram_k_q_r <= K_MEM[sram_k_addr_o];
    if (!sram_v_ceb_o && sram_v_web_o)
        sram_v_q_r <= V_MEM[sram_v_addr_o];
    if (!sram_qkm_ceb_o && sram_qkm_web_o)
        sram_qkm_q_r <= QKM_MEM[sram_qkm_addr_o[12:0]];
    if (!sram_ao_ceb_o && sram_ao_web_o)
        sram_ao_q_r <= AO_MEM[sram_ao_addr_o[13:0]];
end

// SRAM write @ posedge (same beat as addr/din)
always @(posedge clk) begin
    if (!sram_k_ceb_o && !sram_k_web_o)
        K_MEM[sram_k_addr_o] <= sram_k_din_o;
    if (!sram_v_ceb_o && !sram_v_web_o)
        V_MEM[sram_v_addr_o] <= sram_v_din_o;
    if (!sram_qkm_ceb_o && !sram_qkm_web_o)
        QKM_MEM[sram_qkm_addr_o[12:0]] <= sram_qkm_din_o;
    if (!sram_ao_ceb_o && !sram_ao_web_o)
        AO_MEM[sram_ao_addr_o[13:0]] <= sram_ao_din_o;
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt         <= 32'd0;
        mism_cnt           <= 32'd0;
        first_bad_idx      <= 32'hFFFF_FFFF;
        first_bad_rtl      <= 16'd0;
        first_bad_gold     <= 16'd0;
    end else if (y_valid) begin
        if (MLP_ALL[y_recv_cnt[13:0]] !== y_o[15:0]) begin
            mism_cnt <= mism_cnt + 32'd1;
            if (first_bad_idx == 32'hFFFF_FFFF) begin
                first_bad_idx  <= y_recv_cnt;
                first_bad_rtl  <= y_o[15:0];
                first_bad_gold <= MLP_ALL[y_recv_cnt[13:0]];
            end
        end
        y_recv_cnt <= y_recv_cnt + 32'd1;
    end
end

initial begin
    $fsdbDumpfile("mlp_ws.fsdb");
    $fsdbDumpvars;
    // $fsdbDumpMDA;

    $set_toggle_region("u_dut");
    $toggle_start();

    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm2_out_bi.txt"}, NORM2_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_mlp_after_mlp_out_bi.txt"}, MLP_ALL);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_weight_bi.txt"}, W_FC1);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_bias_bi.txt"},   B_FC1);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_weight_bi.txt"}, W_FC2);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_bias_bi.txt"},   B_FC2);

    for (load_i = 0; load_i < K_MEM_DEPTH; load_i = load_i + 1)
        K_MEM[load_i] = 16'd0;
    for (load_i = 0; load_i < V_MEM_DEPTH; load_i = load_i + 1)
        V_MEM[load_i] = 16'd0;
    for (load_i = 0; load_i < QKM_MEM_DEPTH; load_i = load_i + 1)
        QKM_MEM[load_i] = 16'd0;
    for (load_i = 0; load_i < AO_MEM_DEPTH; load_i = load_i + 1)
        AO_MEM[load_i] = 16'd0;

    $display("[TB] mlp_ws block0 (N_TOKENS=%0d OUT_TOTAL=%0d)", N_TOKENS, OUT_TOTAL);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk             = 1'b0;
    reset           = 1'b1;
    start           = 1'b0;
    norm_x_r        = 16'sd0;
    wgt_i_r         = 16'sd0;
    bias_i_r        = 16'sd0;
    sram_k_q_r      = 16'd0;
    sram_v_q_r      = 16'd0;
    sram_qkm_q_r    = 16'd0;
    sram_ao_q_r     = 16'd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- mlp_ws done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d",
             y_recv_cnt, OUT_TOTAL, mism_cnt);

    if (y_recv_cnt !== OUT_TOTAL)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, OUT_TOTAL);
    else if (mism_cnt !== 0)
        $display("  [FAIL] mlp_out mismatch count=%0d first_bad_idx=%0d rtl=%0h gold=%0h",
                 mism_cnt, first_bad_idx, first_bad_rtl, first_bad_gold);
    else
        $display("  [PASS] mlp_ws matches backbone_blocks_0_mlp_after_mlp_out");

    $toggle_stop();
    $toggle_report("mlp_ws_rtl.saif", 1.0e-9, "u_dut");
    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d state=%0d",
             cycle_cnt, busy, y_recv_cnt, u_dut.state);
    $toggle_stop();
    $toggle_report("mlp_ws_rtl.saif", 1.0e-9, "u_dut");
    $finish;
end

endmodule
