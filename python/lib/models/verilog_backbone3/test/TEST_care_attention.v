`timescale 1ns/10ps

`include "recip_lut_seed.v"
`include "recip_nr.v"
`include "care_attention.v"

// =============================================================================
// TEST_care_attention.v -- block0 E2E (N_TOKENS=320) -- WS rewrite
//
// DUT: care_attention monolithic (WS-QKV+SPLIT -> KV -> QKM -> ZR -> CORE
//                                 -> WS-PROJ -> OUT)
// Input : backbone_blocks_0_after_norm1_out_bi.txt
// Weight: backbone_blocks_0_attn_qkv_{weight,bias}_bi.txt
//         backbone_blocks_0_attn_proj_{weight,bias}_bi.txt
// Expect: backbone_blocks_0_after_attn_attn_out_bi.txt (320*32 = 10240)
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_care_attention.v +incdir+.. +incdir+../common \
//       +lint=TFIPC-L +define+TSMC_CM_NO_WARNING \
//       -debug_access+all -debug_region+cell | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|care_attention done' simv.log
//
// Waveform / power (written on done or timeout):
//   care_attention.fsdb       (Verdi)
//   care_attention_rtl.saif   (SAIF toggle, scope u_dut)
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_care_attention;

parameter CYCLE        = 2.0;
parameter EMBED_DIM    = 32;
parameter NUM_HEADS    = 4;
parameter HEAD_DIM     = 8;
parameter N_TOKENS     = 320;
parameter S_Q88        = 152;
parameter RELU6_MAX    = 1536;
parameter RCP_N_NUM    = 205;
parameter RCP_N_SHIFT  = 16;
parameter KV_Q88_ROUND = 8388608;

parameter HD_ELEMS     = NUM_HEADS * N_TOKENS * HEAD_DIM;
parameter QKM_ELEMS    = NUM_HEADS * N_TOKENS;
parameter OUT_ELEMS    = N_TOKENS * EMBED_DIM;
parameter QKV_W_DEPTH  = 96 * EMBED_DIM;
parameter QKV_B_DEPTH  = 96;
parameter PROJ_W_DEPTH = EMBED_DIM * EMBED_DIM;
parameter PROJ_B_DEPTH = EMBED_DIM;

reg         clk;
reg         reset;
reg         start;

wire        norm_rd_en;
wire [13:0] norm_rd_flat;
wire [12:0] wgt_addr_o;
wire [7:0]  bias_addr_o;
wire        busy;
wire        done;
wire signed [15:0] norm_x;
wire signed [15:0] wgt_i;
wire signed [15:0] bias_i;
wire signed [15:0] y_o;
wire               y_valid;
wire [6:0]         y_neu_o;
wire [8:0]         px_tok_o;

wire        sram_q_ceb_o;
wire        sram_q_web_o;
wire [13:0] sram_q_addr_o;
wire [15:0] sram_q_din_o;
wire [15:0] sram_q_q_i;

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

reg [15:0] NORM1_ALL [0:OUT_ELEMS-1];
reg [15:0] GOLD_OUT  [0:OUT_ELEMS-1];
reg [15:0] OUT_MEM   [0:OUT_ELEMS-1];

reg [15:0] QKV_W_MEM  [0:QKV_W_DEPTH-1];
reg [15:0] QKV_B_MEM  [0:QKV_B_DEPTH-1];
reg [15:0] PROJ_W_MEM [0:PROJ_W_DEPTH-1];
reg [15:0] PROJ_B_MEM [0:PROJ_B_DEPTH-1];

reg [15:0] Q_MEM    [0:HD_ELEMS-1];
reg [15:0] K_MEM    [0:HD_ELEMS-1];
reg [15:0] V_MEM    [0:HD_ELEMS-1];
reg [15:0] QKM_MEM  [0:QKM_ELEMS-1];
reg [15:0] AO_MEM   [0:OUT_ELEMS-1];

integer load_i;
integer cmp_i;

reg [13:0] norm_addr_rom_q;
reg [12:0] w_addr_rom_q;
reg [7:0]  b_addr_rom_q;
reg [13:0] q_raddr_q;
reg [13:0] k_raddr_q;
reg [13:0] v_raddr_q;
reg [13:0] qkm_raddr_q;
reg [13:0] ao_raddr_q;

reg [31:0] mism_out;
reg [31:0] first_bad_out;
reg [31:0] cycle_cnt;
reg [31:0] y_recv_cnt;

// -------------------------------------------------------------------------
// ROM model (combinational, latched at negedge for 1-cycle protocol)
// -------------------------------------------------------------------------
assign norm_x = $signed(NORM1_ALL[norm_addr_rom_q]);

assign wgt_i  = (w_addr_rom_q >= 13'd3072) ?
                $signed(PROJ_W_MEM[w_addr_rom_q - 13'd3072]) :
                $signed(QKV_W_MEM[w_addr_rom_q]);

assign bias_i = (w_addr_rom_q >= 13'd3072) ?
                $signed(PROJ_B_MEM[b_addr_rom_q[4:0]]) :
                $signed(QKV_B_MEM[b_addr_rom_q[6:0]]);

// -------------------------------------------------------------------------
// SRAM read model (negedge latch -> posedge data)
// -------------------------------------------------------------------------
assign sram_q_q_i   = Q_MEM[q_raddr_q];
assign sram_k_q_i   = K_MEM[k_raddr_q];
assign sram_v_q_i   = V_MEM[v_raddr_q];
assign sram_qkm_q_i = QKM_MEM[qkm_raddr_q[10:0]];
assign sram_ao_q_i  = AO_MEM[ao_raddr_q[13:0]];

// -------------------------------------------------------------------------
// DUT
// -------------------------------------------------------------------------
care_attention #(
    .EMBED_DIM    (EMBED_DIM),
    .NUM_HEADS    (NUM_HEADS),
    .HEAD_DIM     (HEAD_DIM),
    .N_TOKENS     (N_TOKENS),
    .S_Q88        (S_Q88),
    .RELU6_MAX    (RELU6_MAX),
    .RCP_N_NUM    (RCP_N_NUM),
    .RCP_N_SHIFT  (RCP_N_SHIFT),
    .KV_Q88_ROUND (KV_Q88_ROUND)
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
    .busy            (busy),
    .done            (done),
    .y_o             (y_o),
    .y_valid         (y_valid),
    .y_neu_o         (y_neu_o),
    .px_tok_o        (px_tok_o),
    .sram_q_ceb_o    (sram_q_ceb_o),
    .sram_q_web_o    (sram_q_web_o),
    .sram_q_addr_o   (sram_q_addr_o),
    .sram_q_din_o    (sram_q_din_o),
    .sram_q_q_i      (sram_q_q_i),
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
    .sram_ao_q_i     (sram_ao_q_i)
);

// -------------------------------------------------------------------------
// Clock
// -------------------------------------------------------------------------
always #(CYCLE/2.0) clk = ~clk;

// -------------------------------------------------------------------------
// Negedge latch (ROM + SRAM read address capture)
// -------------------------------------------------------------------------
always @(negedge clk) begin
    if (norm_rd_en)
        norm_addr_rom_q <= norm_rd_flat;
    w_addr_rom_q <= wgt_addr_o;
    b_addr_rom_q <= bias_addr_o;

    if (!sram_q_ceb_o && sram_q_web_o)
        q_raddr_q <= sram_q_addr_o;
    if (!sram_k_ceb_o && sram_k_web_o)
        k_raddr_q <= sram_k_addr_o;
    if (!sram_v_ceb_o && sram_v_web_o)
        v_raddr_q <= sram_v_addr_o;
    if (!sram_qkm_ceb_o && sram_qkm_web_o)
        qkm_raddr_q <= sram_qkm_addr_o;
    if (!sram_ao_ceb_o && sram_ao_web_o)
        ao_raddr_q <= sram_ao_addr_o;
end

// -------------------------------------------------------------------------
// SRAM write model (posedge capture)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (!sram_q_ceb_o && !sram_q_web_o)
        Q_MEM[sram_q_addr_o] <= sram_q_din_o;
    if (!sram_k_ceb_o && !sram_k_web_o)
        K_MEM[sram_k_addr_o] <= sram_k_din_o;
    if (!sram_v_ceb_o && !sram_v_web_o)
        V_MEM[sram_v_addr_o] <= sram_v_din_o;
    if (!sram_qkm_ceb_o && !sram_qkm_web_o)
        QKM_MEM[sram_qkm_addr_o[10:0]] <= sram_qkm_din_o;
    if (!sram_ao_ceb_o && !sram_ao_web_o)
        AO_MEM[sram_ao_addr_o] <= sram_ao_din_o;
end

// -------------------------------------------------------------------------
// Output capture
// -------------------------------------------------------------------------
always @(negedge clk) begin
    if (y_valid)
        OUT_MEM[px_tok_o * EMBED_DIM + y_neu_o] <= y_o;
end

// -------------------------------------------------------------------------
// Cycle / output counter
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        cycle_cnt  <= 32'd0;
        y_recv_cnt <= 32'd0;
    end else begin
        cycle_cnt <= cycle_cnt + 32'd1;
        if (y_valid)
            y_recv_cnt <= y_recv_cnt + 32'd1;
    end
end

// -------------------------------------------------------------------------
// Main test sequence
// -------------------------------------------------------------------------
initial begin
    $fsdbDumpfile("care_attention.fsdb");
    $fsdbDumpvars;
    // $fsdbDumpMDA;

    $set_toggle_region("u_dut");
    $toggle_start();

    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"}, NORM1_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_attn_attn_out_bi.txt"}, GOLD_OUT);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_weight_bi.txt"}, QKV_W_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_bias_bi.txt"},   QKV_B_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_proj_weight_bi.txt"}, PROJ_W_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_proj_bias_bi.txt"},   PROJ_B_MEM);

    for (load_i = 0; load_i < OUT_ELEMS; load_i = load_i + 1)
        OUT_MEM[load_i] = 16'hxxxx;

    $display("[TB] care_attention WS-rewrite E2E block0 N_TOKENS=%0d OUT_ELEMS=%0d",
             N_TOKENS, OUT_ELEMS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk             = 1'b0;
    reset           = 1'b1;
    start           = 1'b0;
    norm_addr_rom_q = 14'd0;
    w_addr_rom_q    = 13'd0;
    b_addr_rom_q    = 8'd0;
    q_raddr_q       = 14'd0;
    k_raddr_q       = 14'd0;
    v_raddr_q       = 14'd0;
    qkm_raddr_q     = 14'd0;
    ao_raddr_q      = 14'd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    mism_out      = 32'd0;
    first_bad_out = 32'hFFFF_FFFF;

    for (cmp_i = 0; cmp_i < OUT_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_OUT[cmp_i] !== OUT_MEM[cmp_i]) begin
            mism_out = mism_out + 32'd1;
            if (first_bad_out == 32'hFFFF_FFFF)
                first_bad_out = cmp_i;
        end
    end

    $display("---- care_attention done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv_cnt=%0d expect=%0d out_mism=%0d",
             y_recv_cnt, OUT_ELEMS, mism_out);

    if (y_recv_cnt !== OUT_ELEMS)
        $display("  [FAIL] y_recv_cnt mismatch (incomplete output stream)");

    if (mism_out !== 0)
        $display("  [FAIL] attn_out mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_out, first_bad_out,
                 OUT_MEM[first_bad_out], GOLD_OUT[first_bad_out]);
    else if (y_recv_cnt === OUT_ELEMS)
        $display("  [PASS] care_attention matches backbone_blocks_0_after_attn_attn_out");

    $toggle_stop();
    $toggle_report("care_attention_rtl.saif", 1.0e-9, "u_dut");
    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d state=%0d",
             cycle_cnt, busy, u_dut.state);
    $toggle_stop();
    $toggle_report("care_attention_rtl.saif", 1.0e-9, "u_dut");
    $finish;
end

endmodule
