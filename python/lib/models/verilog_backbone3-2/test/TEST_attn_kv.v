`timescale 1ns/10ps

`include "care_attention/attn_kv.v"

// =============================================================================
// TEST_attn_kv.v -- block0 S_KV (token 0 slice, N_TOKENS=1)
//
// DUT: attn_kv reads split k from Sram_k + raw v from Sram_v -> kv_buf[256].
// Input:
//   k (split): computed from backbone_blocks_0_attn_after_qkv_k_bi.txt
//   v (raw)  : backbone_blocks_0_attn_after_qkv_v_bi.txt
// Expect: kv_buf vs numpy _care_kv_q88
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_attn_kv.v +incdir+.. +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|attn_kv done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif

module TEST_attn_kv;

parameter CYCLE        = 2.0;
parameter NUM_HEADS    = 4;
parameter HEAD_DIM     = 8;
parameter N_TOKENS     = 1;
parameter TOK_IDX      = 0;
parameter S_Q88        = 152;
parameter RELU6_MAX    = 1536;
parameter RCP_N_NUM    = 205;
parameter RCP_N_SHIFT  = 16;
parameter KV_Q88_ROUND = 8388608;

parameter N_TOKENS_GOLD = 320;
parameter HD_ELEMS_GOLD = NUM_HEADS * N_TOKENS_GOLD * HEAD_DIM;
parameter HD_ELEMS      = NUM_HEADS * N_TOKENS * HEAD_DIM;
parameter KV_ELEMS      = NUM_HEADS * HEAD_DIM * HEAD_DIM;

reg         clk;
reg         reset;
reg         start;

wire        busy;
wire        done;

wire        sram_k_ceb_o;
wire        sram_k_web_o;
wire [13:0] sram_k_addr_o;
wire [15:0] sram_k_q_i;

wire        sram_v_ceb_o;
wire        sram_v_web_o;
wire [13:0] sram_v_addr_o;
wire [15:0] sram_v_q_i;

reg [15:0] K_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] V_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] K_IN     [0:HD_ELEMS-1];
reg [15:0] V_IN     [0:HD_ELEMS-1];
reg [15:0] K_SPLIT  [0:HD_ELEMS-1];
reg [15:0] GOLD_KV  [0:KV_ELEMS-1];
reg [15:0] K_MEM    [0:HD_ELEMS-1];
reg [15:0] V_MEM    [0:HD_ELEMS-1];

integer load_i;
integer cmp_i;
integer h_i;
integer d_out_i;
integer d_k_i;
integer n_i;
integer flat_gold;
integer flat_hd;
integer prod_k;
integer kv_acc;
integer kv_scaled;
integer kv_oidx;

reg [13:0] k_raddr_q;
reg [13:0] v_raddr_q;

reg [31:0] mism_kv;
reg [31:0] first_bad_kv;
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

function integer kv_flat_idx;
    input integer h;
    input integer d_out;
    input integer d_k;
    begin
        kv_flat_idx = h * HEAD_DIM * HEAD_DIM + d_out * HEAD_DIM + d_k;
    end
endfunction

attn_kv #(
    .NUM_HEADS    (NUM_HEADS),
    .HEAD_DIM     (HEAD_DIM),
    .N_TOKENS     (N_TOKENS),
    .RCP_N_NUM    (RCP_N_NUM),
    .RCP_N_SHIFT  (RCP_N_SHIFT),
    .KV_Q88_ROUND (KV_Q88_ROUND)
) u_dut (
    .clk           (clk),
    .reset         (reset),
    .start         (start),
    .busy          (busy),
    .done          (done),
    .sram_k_ceb_o  (sram_k_ceb_o),
    .sram_k_web_o  (sram_k_web_o),
    .sram_k_addr_o (sram_k_addr_o),
    .sram_k_q_i    (sram_k_q_i),
    .sram_v_ceb_o  (sram_v_ceb_o),
    .sram_v_web_o  (sram_v_web_o),
    .sram_v_addr_o (sram_v_addr_o),
    .sram_v_q_i    (sram_v_q_i)
);

assign sram_k_q_i = K_MEM[k_raddr_q];
assign sram_v_q_i = V_MEM[v_raddr_q];

always #(CYCLE/2.0) clk = ~clk;

always @(negedge clk) begin
    if (!sram_k_ceb_o && sram_k_web_o)
        k_raddr_q <= sram_k_addr_o;
    if (!sram_v_ceb_o && sram_v_web_o)
        v_raddr_q <= sram_v_addr_o;
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_k_bi.txt"}, K_IN_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_v_bi.txt"}, V_IN_ALL);

    for (load_i = 0; load_i < HD_ELEMS; load_i = load_i + 1) begin
        h_i = load_i / HEAD_DIM;
        d_out_i = load_i % HEAD_DIM;
        flat_gold = h_i * N_TOKENS_GOLD * HEAD_DIM + TOK_IDX * HEAD_DIM + d_out_i;
        K_IN[load_i] = K_IN_ALL[flat_gold];
        V_IN[load_i] = V_IN_ALL[flat_gold];

        prod_k = $signed(K_IN[load_i]) * S_Q88;
        K_SPLIT[load_i] = relu6_sat16(sat16_32((prod_k + 128) >>> 8));
        K_MEM[load_i] = K_SPLIT[load_i];
        V_MEM[load_i] = V_IN[load_i];
    end

    for (load_i = 0; load_i < KV_ELEMS; load_i = load_i + 1) begin
        h_i     = load_i / (HEAD_DIM * HEAD_DIM);
        d_out_i = (load_i / HEAD_DIM) % HEAD_DIM;
        d_k_i   = load_i % HEAD_DIM;
        kv_acc  = 0;
        for (n_i = 0; n_i < N_TOKENS; n_i = n_i + 1) begin
            kv_acc = kv_acc +
                $signed(K_SPLIT[flat_hd_idx(h_i, n_i, d_out_i)]) *
                $signed(V_IN[flat_hd_idx(h_i, n_i, d_k_i)]);
        end
        kv_scaled = kv_acc * RCP_N_NUM;
        GOLD_KV[load_i] = sat16_48((kv_scaled + KV_Q88_ROUND) >>> (RCP_N_SHIFT + 8));
    end

    $display("[TB] attn_kv token %0d N_TOKENS=%0d KV_ELEMS=%0d",
             TOK_IDX, N_TOKENS, KV_ELEMS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);

    clk       = 1'b0;
    reset     = 1'b1;
    start     = 1'b0;
    k_raddr_q = 14'd0;
    v_raddr_q = 14'd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    mism_kv      = 32'd0;
    first_bad_kv = 32'hFFFF_FFFF;

    for (cmp_i = 0; cmp_i < KV_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_KV[cmp_i] !== u_dut.kv_buf[cmp_i]) begin
            mism_kv = mism_kv + 32'd1;
            if (first_bad_kv == 32'hFFFF_FFFF)
                first_bad_kv = cmp_i;
        end
    end

    $display("---- attn_kv done @ cycle %0d ----", cycle_cnt);
    $display("[TB] kv_mism=%0d", mism_kv);

    if (mism_kv !== 0) begin
        kv_oidx = first_bad_kv;
        h_i     = kv_oidx / (HEAD_DIM * HEAD_DIM);
        d_out_i = (kv_oidx / HEAD_DIM) % HEAD_DIM;
        d_k_i   = kv_oidx % HEAD_DIM;
        $display("  [FAIL] kv_buf mismatch count=%0d first_bad=%0d h=%0d d_out=%0d d_k=%0d dut=%h gold=%h",
                 mism_kv, first_bad_kv, h_i, d_out_i, d_k_i,
                 u_dut.kv_buf[first_bad_kv], GOLD_KV[first_bad_kv]);
    end else
        $display("  [PASS] attn_kv matches numpy _care_kv_q88");

    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d", cycle_cnt, busy);
    $finish;
end

endmodule
