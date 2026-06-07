`timescale 1ns/10ps

`include "care_attention/attn_preprocess.v"

// =============================================================================
// TEST_attn_preprocess.v -- block0 S_SPLIT + S_K_MEAN + S_QK_MEAN (token 0 slice)
//
// DUT: attn_preprocess N_TOKENS=1.
// Input : backbone_blocks_0_attn_after_qkv_{q,k}_bi.txt (pre-split)
// Expect: split q/k, km_buf[32], Sram_qkm[4] via numpy _care_* formulas
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_attn_preprocess.v +incdir+.. +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif

module TEST_attn_preprocess;

parameter CYCLE       = 2.0;
parameter NUM_HEADS   = 4;
parameter HEAD_DIM    = 8;
parameter N_TOKENS    = 1;
parameter TOK_IDX     = 0;
parameter S_Q88       = 152;
parameter RELU6_MAX   = 1536;
parameter RCP_N_NUM   = 205;
parameter RCP_N_SHIFT = 16;

parameter N_TOKENS_GOLD = 320;
parameter HD_ELEMS_GOLD = NUM_HEADS * N_TOKENS_GOLD * HEAD_DIM;
parameter HD_ELEMS      = NUM_HEADS * N_TOKENS * HEAD_DIM;
parameter KM_ELEMS      = NUM_HEADS * HEAD_DIM;
parameter QKM_ELEMS     = NUM_HEADS * N_TOKENS;

reg         clk;
reg         reset;
reg         start;

wire        busy;
wire        done;

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

wire        sram_qkm_ceb_o;
wire        sram_qkm_web_o;
wire [13:0] sram_qkm_addr_o;
wire [15:0] sram_qkm_din_o;

reg [15:0] Q_IN_ALL  [0:HD_ELEMS_GOLD-1];
reg [15:0] K_IN_ALL  [0:HD_ELEMS_GOLD-1];
reg [15:0] Q_IN      [0:HD_ELEMS-1];
reg [15:0] K_IN      [0:HD_ELEMS-1];
reg [15:0] Q_SPLIT   [0:HD_ELEMS-1];
reg [15:0] K_SPLIT   [0:HD_ELEMS-1];
reg [15:0] GOLD_Q    [0:HD_ELEMS-1];
reg [15:0] GOLD_K    [0:HD_ELEMS-1];
reg [15:0] GOLD_KM   [0:KM_ELEMS-1];
reg [15:0] GOLD_QKM  [0:QKM_ELEMS-1];
reg [15:0] Q_MEM     [0:HD_ELEMS-1];
reg [15:0] K_MEM     [0:HD_ELEMS-1];
reg [15:0] QKM_MEM   [0:QKM_ELEMS-1];

integer load_i;
integer cmp_i;
integer h_i;
integer d_i;
integer n_i;
integer flat_gold;
integer flat_hd;
integer prod_q;
integer prod_k;
integer k_sum;
integer km_scaled;
integer qkm_acc;
integer qkm_idx;

reg [13:0] q_raddr_q;
reg [13:0] k_raddr_q;

reg [31:0] mism_q;
reg [31:0] mism_k;
reg [31:0] mism_km;
reg [31:0] mism_qkm;
reg [31:0] first_bad_q;
reg [31:0] first_bad_k;
reg [31:0] first_bad_km;
reg [31:0] first_bad_qkm;
reg [31:0] cycle_cnt;

function signed [15:0] sat16_32;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767)
            out = 16'sh7FFF;
        else if (v < -32768)
            out = -16'sh8000;
        else
            out = v;
        sat16_32 = out;
    end
endfunction

function signed [15:0] sat16_33;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767)
            out = 16'sh7FFF;
        else if (v < -32768)
            out = -16'sh8000;
        else
            out = v;
        sat16_33 = out;
    end
endfunction

function signed [15:0] sat16_48;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767)
            out = 16'sh7FFF;
        else if (v < -32768)
            out = -16'sh8000;
        else
            out = v;
        sat16_48 = out;
    end
endfunction

function signed [15:0] relu6_sat16;
    input signed [15:0] v;
    begin
        if (v < 16'sd0)
            relu6_sat16 = 16'sd0;
        else if (v > RELU6_MAX[15:0])
            relu6_sat16 = RELU6_MAX[15:0];
        else
            relu6_sat16 = v;
    end
endfunction

function integer flat_hd_idx;
    input integer h;
    input integer n;
    input integer d;
    begin
        flat_hd_idx = h * N_TOKENS * HEAD_DIM + n * HEAD_DIM + d;
    end
endfunction

function integer qkm_max1;
    input integer v;
    begin
        if (v < 1)
            qkm_max1 = 1;
        else
            qkm_max1 = v;
    end
endfunction

attn_preprocess #(
    .NUM_HEADS     (NUM_HEADS),
    .HEAD_DIM      (HEAD_DIM),
    .N_TOKENS      (N_TOKENS),
    .S_Q88         (S_Q88),
    .RELU6_MAX     (RELU6_MAX),
    .RCP_N_NUM     (RCP_N_NUM),
    .RCP_N_SHIFT   (RCP_N_SHIFT)
) u_dut (
    .clk             (clk),
    .reset           (reset),
    .start           (start),
    .busy            (busy),
    .done            (done),
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
    .sram_qkm_ceb_o  (sram_qkm_ceb_o),
    .sram_qkm_web_o  (sram_qkm_web_o),
    .sram_qkm_addr_o (sram_qkm_addr_o),
    .sram_qkm_din_o  (sram_qkm_din_o)
);

assign sram_q_q_i = Q_MEM[q_raddr_q];
assign sram_k_q_i = K_MEM[k_raddr_q];

always #(CYCLE/2.0) clk = ~clk;

always @(negedge clk) begin
    if (!sram_q_ceb_o && sram_q_web_o)
        q_raddr_q <= sram_q_addr_o;
    if (!sram_k_ceb_o && sram_k_web_o)
        k_raddr_q <= sram_k_addr_o;
end

always @(posedge clk) begin
    if (!sram_q_ceb_o && !sram_q_web_o)
        Q_MEM[sram_q_addr_o] <= sram_q_din_o;
    if (!sram_k_ceb_o && !sram_k_web_o)
        K_MEM[sram_k_addr_o] <= sram_k_din_o;
    if (!sram_qkm_ceb_o && !sram_qkm_web_o)
        QKM_MEM[sram_qkm_addr_o[4:0]] <= sram_qkm_din_o;
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_q_bi.txt"}, Q_IN_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_k_bi.txt"}, K_IN_ALL);

    for (load_i = 0; load_i < HD_ELEMS; load_i = load_i + 1) begin
        h_i = load_i / HEAD_DIM;
        d_i = load_i % HEAD_DIM;
        flat_gold = h_i * N_TOKENS_GOLD * HEAD_DIM + TOK_IDX * HEAD_DIM + d_i;
        Q_IN[load_i] = Q_IN_ALL[flat_gold];
        K_IN[load_i] = K_IN_ALL[flat_gold];
        Q_MEM[load_i] = Q_IN[load_i];
        K_MEM[load_i] = K_IN[load_i];

        prod_q = $signed(Q_IN[load_i]) * S_Q88;
        prod_k = $signed(K_IN[load_i]) * S_Q88;
        Q_SPLIT[load_i] = relu6_sat16(sat16_32((prod_q + 128) >>> 8));
        K_SPLIT[load_i] = relu6_sat16(sat16_32((prod_k + 128) >>> 8));
        GOLD_Q[load_i]  = Q_SPLIT[load_i];
        GOLD_K[load_i]  = K_SPLIT[load_i];
    end

    for (load_i = 0; load_i < KM_ELEMS; load_i = load_i + 1) begin
        h_i = load_i / HEAD_DIM;
        d_i = load_i % HEAD_DIM;
        k_sum = 0;
        for (n_i = 0; n_i < N_TOKENS; n_i = n_i + 1) begin
            flat_hd = flat_hd_idx(h_i, n_i, d_i);
            k_sum = k_sum + $signed(K_SPLIT[flat_hd]);
        end
        km_scaled = k_sum * RCP_N_NUM;
        GOLD_KM[load_i] = sat16_48((km_scaled + 32768) >>> RCP_N_SHIFT);
    end

    for (load_i = 0; load_i < QKM_ELEMS; load_i = load_i + 1) begin
        h_i = load_i / N_TOKENS;
        n_i = load_i % N_TOKENS;
        qkm_acc = 0;
        for (d_i = 0; d_i < HEAD_DIM; d_i = d_i + 1) begin
            flat_hd = flat_hd_idx(h_i, n_i, d_i);
            qkm_acc = qkm_acc +
                $signed(Q_SPLIT[flat_hd]) * $signed(GOLD_KM[h_i * HEAD_DIM + d_i]);
        end
        GOLD_QKM[load_i] = qkm_max1(sat16_33((qkm_acc + 128) >>> 8));
        QKM_MEM[load_i] = 16'hxxxx;
    end

    $display("[TB] attn_preprocess SPLIT+K_MEAN+QK_MEAN token %0d N_TOKENS=%0d",
             TOK_IDX, N_TOKENS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);

    clk       = 1'b0;
    reset     = 1'b1;
    start     = 1'b0;
    q_raddr_q = 14'd0;
    k_raddr_q = 14'd0;

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
    mism_km     = 32'd0;
    mism_qkm    = 32'd0;
    first_bad_q   = 32'hFFFF_FFFF;
    first_bad_k   = 32'hFFFF_FFFF;
    first_bad_km  = 32'hFFFF_FFFF;
    first_bad_qkm = 32'hFFFF_FFFF;

    for (cmp_i = 0; cmp_i < HD_ELEMS; cmp_i = cmp_i + 1) begin
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
    end

    for (cmp_i = 0; cmp_i < KM_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_KM[cmp_i] !== u_dut.km_buf[cmp_i]) begin
            mism_km = mism_km + 32'd1;
            if (first_bad_km == 32'hFFFF_FFFF)
                first_bad_km = cmp_i;
        end
    end

    for (cmp_i = 0; cmp_i < QKM_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_QKM[cmp_i] !== QKM_MEM[cmp_i]) begin
            mism_qkm = mism_qkm + 32'd1;
            if (first_bad_qkm == 32'hFFFF_FFFF)
                first_bad_qkm = cmp_i;
        end
    end

    $display("---- attn_preprocess done @ cycle %0d ----", cycle_cnt);
    $display("[TB] q_mism=%0d k_mism=%0d km_mism=%0d qkm_mism=%0d",
             mism_q, mism_k, mism_km, mism_qkm);

    if (mism_q !== 0)
        $display("  [FAIL] q_split mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_q, first_bad_q, Q_MEM[first_bad_q], GOLD_Q[first_bad_q]);
    else if (mism_k !== 0)
        $display("  [FAIL] k_split mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_k, first_bad_k, K_MEM[first_bad_k], GOLD_K[first_bad_k]);
    else if (mism_km !== 0)
        $display("  [FAIL] km_buf mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_km, first_bad_km, u_dut.km_buf[first_bad_km], GOLD_KM[first_bad_km]);
    else if (mism_qkm !== 0)
        $display("  [FAIL] qkm mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_qkm, first_bad_qkm, QKM_MEM[first_bad_qkm], GOLD_QKM[first_bad_qkm]);
    else
        $display("  [PASS] attn_preprocess SPLIT+K_MEAN+QK_MEAN matches numpy trunk");

    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d", cycle_cnt, busy);
    $finish;
end

endmodule
