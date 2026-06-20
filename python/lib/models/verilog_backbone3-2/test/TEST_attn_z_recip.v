`timescale 1ns/10ps

`include "recip_lut_seed.v"
`include "recip_nr.v"
`include "care_attention/attn_z_recip.v"

// =============================================================================
// TEST_attn_z_recip.v -- block0 S_Z_RECIP (token 0 slice, N_TOKENS=1)
//
// DUT: attn_z_recip reads qkm from behavioral Sram_qkm, writes zr in-place.
// Input qkm: TB computes preprocess chain (split + k_mean + qk_mean) from
//   backbone_blocks_0_attn_after_qkv_{q,k}_bi.txt
// Expect zr: _recip_nr_q88_fixed(max(qkm,1)) bit-accurate vs recip_nr.v
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_attn_z_recip.v +incdir+.. +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|attn_z_recip done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif

module TEST_attn_z_recip;

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

wire        sram_qkm_ceb_o;
wire        sram_qkm_web_o;
wire [13:0] sram_qkm_addr_o;
wire [15:0] sram_qkm_din_o;
wire [15:0] sram_qkm_q_i;

reg [15:0] Q_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] K_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] Q_IN     [0:HD_ELEMS-1];
reg [15:0] K_IN     [0:HD_ELEMS-1];
reg [15:0] Q_SPLIT  [0:HD_ELEMS-1];
reg [15:0] K_SPLIT  [0:HD_ELEMS-1];
reg [15:0] GOLD_KM  [0:KM_ELEMS-1];
reg [15:0] GOLD_QKM [0:QKM_ELEMS-1];
reg [15:0] GOLD_ZR  [0:QKM_ELEMS-1];
reg [15:0] QKM_MEM  [0:QKM_ELEMS-1];
reg [15:0] ZR_MEM   [0:QKM_ELEMS-1];

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
integer x_eps;

reg [13:0] qkm_raddr_q;

reg [31:0] mism_zr;
reg [31:0] first_bad_zr;
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

function [3:0] recip_msb_k;
    input integer x;
    integer bit_i;
    reg [3:0] k;
    begin
        k = 4'd0;
        if (x < 1)
            x = 1;
        for (bit_i = 15; bit_i >= 0; bit_i = bit_i - 1) begin
            if ((x >= (1 << bit_i)) && (k == 0))
                k = bit_i[3:0];
        end
        recip_msb_k = k;
    end
endfunction

function signed [15:0] recip_lut_y0;
    input [3:0] k;
    begin
        case (k)
            4'd15: recip_lut_y0 = 16'sd2;
            4'd14: recip_lut_y0 = 16'sd3;
            4'd13: recip_lut_y0 = 16'sd5;
            4'd12: recip_lut_y0 = 16'sd11;
            4'd11: recip_lut_y0 = 16'sd21;
            4'd10: recip_lut_y0 = 16'sd43;
            4'd9 : recip_lut_y0 = 16'sd85;
            4'd8 : recip_lut_y0 = 16'sd171;
            4'd7 : recip_lut_y0 = 16'sd341;
            4'd6 : recip_lut_y0 = 16'sd683;
            4'd5 : recip_lut_y0 = 16'sd1365;
            4'd4 : recip_lut_y0 = 16'sd2731;
            4'd3 : recip_lut_y0 = 16'sd5461;
            4'd2 : recip_lut_y0 = 16'sd10922;
            4'd1 : recip_lut_y0 = 16'sd21845;
            default: recip_lut_y0 = 16'sd32767;
        endcase
    end
endfunction

function signed [15:0] trunc_q88_slice32;
    input integer v;
    reg signed [31:0] vv;
    begin
        vv = v;
        trunc_q88_slice32 = vv[23:8];
    end
endfunction

function signed [15:0] recip_nr_golden;
    input integer x_in;
    integer x;
    integer y;
    integer c;
    begin
        x = x_in;
        if (x < 1)
            x = 1;
        y = recip_lut_y0(recip_msb_k(x));
        c = 512 - trunc_q88_slice32(x * y);
        y = trunc_q88_slice32(y * c);
        if (y > 32767)
            y = 32767;
        else if (y < -32768)
            y = -32768;
        recip_nr_golden = y;
    end
endfunction

attn_z_recip #(
    .NUM_HEADS (NUM_HEADS),
    .N_TOKENS  (N_TOKENS)
) u_dut (
    .clk             (clk),
    .reset           (reset),
    .start           (start),
    .busy            (busy),
    .done            (done),
    .sram_qkm_ceb_o  (sram_qkm_ceb_o),
    .sram_qkm_web_o  (sram_qkm_web_o),
    .sram_qkm_addr_o (sram_qkm_addr_o),
    .sram_qkm_din_o  (sram_qkm_din_o),
    .sram_qkm_q_i    (sram_qkm_q_i)
);

assign sram_qkm_q_i = QKM_MEM[qkm_raddr_q[4:0]];

always #(CYCLE/2.0) clk = ~clk;

always @(negedge clk) begin
    if (!sram_qkm_ceb_o && sram_qkm_web_o)
        qkm_raddr_q <= sram_qkm_addr_o;
end

always @(posedge clk) begin
    if (!sram_qkm_ceb_o && !sram_qkm_web_o)
        ZR_MEM[sram_qkm_addr_o[4:0]] <= sram_qkm_din_o;
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

        prod_q = $signed(Q_IN[load_i]) * S_Q88;
        prod_k = $signed(K_IN[load_i]) * S_Q88;
        Q_SPLIT[load_i] = relu6_sat16(sat16_32((prod_q + 128) >>> 8));
        K_SPLIT[load_i] = relu6_sat16(sat16_32((prod_k + 128) >>> 8));
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
        QKM_MEM[load_i]  = GOLD_QKM[load_i];
        x_eps = GOLD_QKM[load_i];
        if (x_eps < 1)
            x_eps = 1;
        GOLD_ZR[load_i] = recip_nr_golden(x_eps);
        ZR_MEM[load_i]  = 16'hxxxx;
    end

    $display("[TB] attn_z_recip token %0d N_TOKENS=%0d QKM_ELEMS=%0d",
             TOK_IDX, N_TOKENS, QKM_ELEMS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);

    clk         = 1'b0;
    reset       = 1'b1;
    start       = 1'b0;
    qkm_raddr_q = 14'd0;

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    mism_zr      = 32'd0;
    first_bad_zr = 32'hFFFF_FFFF;

    for (cmp_i = 0; cmp_i < QKM_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_ZR[cmp_i] !== ZR_MEM[cmp_i]) begin
            mism_zr = mism_zr + 32'd1;
            if (first_bad_zr == 32'hFFFF_FFFF)
                first_bad_zr = cmp_i;
        end
    end

    $display("---- attn_z_recip done @ cycle %0d ----", cycle_cnt);
    $display("[TB] zr_mism=%0d", mism_zr);

    if (mism_zr !== 0) begin
        $display("  [FAIL] zr mismatch count=%0d first_bad=%0d dut=%h gold=%h qkm_in=%h",
                 mism_zr, first_bad_zr,
                 ZR_MEM[first_bad_zr], GOLD_ZR[first_bad_zr], GOLD_QKM[first_bad_zr]);
    end else
        $display("  [PASS] attn_z_recip matches numpy _recip_nr_q88_fixed");

    $finish;
end

initial begin
    #(CYCLE * 200_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d", cycle_cnt, busy);
    $finish;
end

endmodule
