`timescale 1ns/10ps

`include "common/vec_mac8.v"
`include "linear_vec8.v"
`include "care_attention/attn_proj.v"

// =============================================================================
// TEST_attn_proj.v -- block0 S_PROJ (token 0 slice, N_TOKENS=1)
//
// DUT: attn_proj reads ao from Sram_ao + linear_vec8 32->32 -> y stream.
// Input ao: TB preloads numpy attention_forward ao (km/kv over N_TOKENS_GOLD=320,
// token TOK_IDX slice). DUT attn_proj stays N_TOKENS=1.
// Expect: backbone_blocks_0_after_attn_attn_out_bi.txt (token slice)
// Weight: backbone_blocks_0_attn_proj_{weight,bias}_bi.txt
//
// VCS:
//   cd python/lib/models/verilog_backbone3/test
//   vcs TEST_attn_proj.v +incdir+.. +incdir+../common +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|attn_proj done' simv.log
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "../Memory2/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "../Memory2/Weight"
`endif

module TEST_attn_proj;

parameter CYCLE        = 2.0;
parameter EMBED_DIM    = 32;
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
parameter KV_ELEMS      = NUM_HEADS * HEAD_DIM * HEAD_DIM;
parameter QKM_ELEMS     = NUM_HEADS * N_TOKENS;
parameter AO_ELEMS      = N_TOKENS * EMBED_DIM;
parameter PROJ_W_DEPTH  = EMBED_DIM * EMBED_DIM;
parameter PROJ_B_DEPTH  = EMBED_DIM;

reg         clk;
reg         reset;
reg         start;

wire signed [15:0] wgt_i;
wire signed [15:0] bias_i;
wire [12:0]        wgt_addr_o;
wire [7:0]         vm_b_addr_o;
wire               busy;
wire               done;
wire signed [15:0] y_o;
wire               y_valid;
wire [6:0]         y_neu_o;
wire [8:0]         px_tok_o;

wire        sram_ao_ceb_o;
wire        sram_ao_web_o;
wire [13:0] sram_ao_addr_o;
wire [15:0] sram_ao_q_i;

reg [15:0] Q_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] K_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] V_IN_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] ATTN_ALL [0:N_TOKENS_GOLD*EMBED_DIM-1];
reg [15:0] W_MEM [0:PROJ_W_DEPTH-1];
reg [15:0] B_MEM [0:PROJ_B_DEPTH-1];

reg [15:0] Q_SPLIT_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] K_SPLIT_ALL [0:HD_ELEMS_GOLD-1];
reg [15:0] V_ALL       [0:HD_ELEMS_GOLD-1];
reg [15:0] GOLD_KM  [0:NUM_HEADS*HEAD_DIM-1];
reg [15:0] GOLD_KV  [0:KV_ELEMS-1];
reg [15:0] GOLD_ZR  [0:QKM_ELEMS-1];
reg [15:0] GOLD_OUT [0:AO_ELEMS-1];
reg [15:0] AO_MEM   [0:AO_ELEMS-1];
reg [15:0] OUT_MEM  [0:AO_ELEMS-1];

integer load_i;
integer cmp_i;
integer h_i;
integer n_i;
integer d_i;
integer d_out_i;
integer d_k_i;
integer prod_q;
integer prod_k;
integer k_sum;
integer km_scaled;
integer qkm_acc;
integer kv_acc;
integer kv_scaled;
integer dot_acc;
integer dot_sat;
integer zr_val;
integer ao_val;
integer x_eps;
integer out_base;

reg [13:0] ao_raddr_q;
reg [12:0] w_addr_rom_q;
reg [7:0]  b_addr_rom_q;

reg [31:0] mism_out;
reg [31:0] first_bad_out;
reg [31:0] y_recv_cnt;
reg [31:0] cycle_cnt;

function signed [15:0] sat16_32;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767) out = 16'sh7FFF;
        else if (v < -32768) out = -16'sh8000;
        else out = v;
        sat16_32 = out;
    end
endfunction

function signed [15:0] sat16_33;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767) out = 16'sh7FFF;
        else if (v < -32768) out = -16'sh8000;
        else out = v;
        sat16_33 = out;
    end
endfunction

function signed [15:0] sat16_48;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767) out = 16'sh7FFF;
        else if (v < -32768) out = -16'sh8000;
        else out = v;
        sat16_48 = out;
    end
endfunction

function signed [15:0] sat16_49;
    input integer v;
    reg signed [15:0] out;
    begin
        if (v > 32767) out = 16'sh7FFF;
        else if (v < -32768) out = -16'sh8000;
        else out = v;
        sat16_49 = out;
    end
endfunction

function signed [15:0] relu6_sat16;
    input signed [15:0] v;
    begin
        if (v < 16'sd0) relu6_sat16 = 16'sd0;
        else if (v > RELU6_MAX[15:0]) relu6_sat16 = RELU6_MAX[15:0];
        else relu6_sat16 = v;
    end
endfunction

function integer flat_hd_gold_idx;
    input integer h;
    input integer n;
    input integer d;
    begin
        flat_hd_gold_idx = h * N_TOKENS_GOLD * HEAD_DIM + n * HEAD_DIM + d;
    end
endfunction

function integer kv_store_idx;
    input integer h;
    input integer d_out;
    input integer d_k;
    begin
        kv_store_idx = h * HEAD_DIM * HEAD_DIM + d_out * HEAD_DIM + d_k;
    end
endfunction

function integer ao_flat_idx;
    input integer n;
    input integer h;
    input integer d_out;
    begin
        ao_flat_idx = n * EMBED_DIM + h * HEAD_DIM + d_out;
    end
endfunction

function integer qkm_max1;
    input integer v;
    begin
        if (v < 1) qkm_max1 = 1;
        else qkm_max1 = v;
    end
endfunction

function [3:0] recip_msb_k;
    input integer x;
    integer bit_i;
    reg [3:0] k;
    begin
        k = 4'd0;
        if (x < 1) x = 1;
        for (bit_i = 15; bit_i >= 0; bit_i = bit_i - 1)
            if ((x >= (1 << bit_i)) && (k == 0)) k = bit_i[3:0];
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
        if (x < 1) x = 1;
        y = recip_lut_y0(recip_msb_k(x));
        c = 512 - trunc_q88_slice32(x * y);
        y = trunc_q88_slice32(y * c);
        if (y > 32767) y = 32767;
        else if (y < -32768) y = -32768;
        recip_nr_golden = y;
    end
endfunction

attn_proj #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_dut (
    .clk             (clk),
    .reset           (reset),
    .start           (start),
    .wgt_i           (wgt_i),
    .bias_i          (bias_i),
    .wgt_addr_o      (wgt_addr_o),
    .busy            (busy),
    .done            (done),
    .y_o             (y_o),
    .y_valid         (y_valid),
    .y_neu_o         (y_neu_o),
    .px_tok_o        (px_tok_o),
    .sram_ao_ceb_o   (sram_ao_ceb_o),
    .sram_ao_web_o   (sram_ao_web_o),
    .sram_ao_addr_o  (sram_ao_addr_o),
    .sram_ao_q_i     (sram_ao_q_i)
);

assign vm_b_addr_o = u_dut.u_lin_proj.u_vec_mac8.b_addr_o;
assign wgt_i  = $signed(W_MEM[w_addr_rom_q]);
assign bias_i = $signed(B_MEM[b_addr_rom_q]);
assign sram_ao_q_i = AO_MEM[ao_raddr_q[4:0]];

always #(CYCLE/2.0) clk = ~clk;

always @(negedge clk) begin
    if (!sram_ao_ceb_o && sram_ao_web_o)
        ao_raddr_q <= sram_ao_addr_o;
    w_addr_rom_q <= wgt_addr_o;
    b_addr_rom_q <= vm_b_addr_o;
end

always @(negedge clk) begin
    if (y_valid)
        OUT_MEM[px_tok_o * EMBED_DIM + y_neu_o] <= y_o;
end

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

initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_q_bi.txt"}, Q_IN_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_k_bi.txt"}, K_IN_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_v_bi.txt"}, V_IN_ALL);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_attn_attn_out_bi.txt"}, ATTN_ALL);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_proj_weight_bi.txt"}, W_MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_proj_bias_bi.txt"},   B_MEM);

    out_base = TOK_IDX * EMBED_DIM;
    for (load_i = 0; load_i < AO_ELEMS; load_i = load_i + 1)
        GOLD_OUT[load_i] = ATTN_ALL[out_base + load_i];

    for (load_i = 0; load_i < HD_ELEMS_GOLD; load_i = load_i + 1) begin
        prod_q = $signed(Q_IN_ALL[load_i]) * S_Q88;
        prod_k = $signed(K_IN_ALL[load_i]) * S_Q88;
        Q_SPLIT_ALL[load_i] = relu6_sat16(sat16_32((prod_q + 128) >>> 8));
        K_SPLIT_ALL[load_i] = relu6_sat16(sat16_32((prod_k + 128) >>> 8));
        V_ALL[load_i] = V_IN_ALL[load_i];
    end

    for (load_i = 0; load_i < NUM_HEADS * HEAD_DIM; load_i = load_i + 1) begin
        h_i = load_i / HEAD_DIM;
        d_i = load_i % HEAD_DIM;
        k_sum = 0;
        for (n_i = 0; n_i < N_TOKENS_GOLD; n_i = n_i + 1)
            k_sum = k_sum + $signed(K_SPLIT_ALL[flat_hd_gold_idx(h_i, n_i, d_i)]);
        km_scaled = k_sum * RCP_N_NUM;
        GOLD_KM[load_i] = sat16_48((km_scaled + 32768) >>> RCP_N_SHIFT);
    end

    for (h_i = 0; h_i < NUM_HEADS; h_i = h_i + 1) begin
        qkm_acc = 0;
        for (d_i = 0; d_i < HEAD_DIM; d_i = d_i + 1)
            qkm_acc = qkm_acc +
                $signed(Q_SPLIT_ALL[flat_hd_gold_idx(h_i, TOK_IDX, d_i)]) *
                $signed(GOLD_KM[h_i * HEAD_DIM + d_i]);
        x_eps = qkm_max1(sat16_33((qkm_acc + 128) >>> 8));
        if (x_eps < 1) x_eps = 1;
        GOLD_ZR[h_i] = recip_nr_golden(x_eps);
    end

    for (load_i = 0; load_i < KV_ELEMS; load_i = load_i + 1) begin
        h_i     = load_i / (HEAD_DIM * HEAD_DIM);
        d_out_i = (load_i / HEAD_DIM) % HEAD_DIM;
        d_k_i   = load_i % HEAD_DIM;
        kv_acc  = 0;
        for (n_i = 0; n_i < N_TOKENS_GOLD; n_i = n_i + 1)
            kv_acc = kv_acc +
                $signed(K_SPLIT_ALL[flat_hd_gold_idx(h_i, n_i, d_out_i)]) *
                $signed(V_ALL[flat_hd_gold_idx(h_i, n_i, d_k_i)]);
        kv_scaled = kv_acc * RCP_N_NUM;
        GOLD_KV[load_i] = sat16_48((kv_scaled + KV_Q88_ROUND) >>> (RCP_N_SHIFT + 8));
    end

    for (h_i = 0; h_i < NUM_HEADS; h_i = h_i + 1) begin
        zr_val = GOLD_ZR[h_i];
        for (d_out_i = 0; d_out_i < HEAD_DIM; d_out_i = d_out_i + 1) begin
            dot_acc = 0;
            for (d_k_i = 0; d_k_i < HEAD_DIM; d_k_i = d_k_i + 1)
                dot_acc = dot_acc +
                    $signed(Q_SPLIT_ALL[flat_hd_gold_idx(h_i, TOK_IDX, d_k_i)]) *
                    $signed(GOLD_KV[kv_store_idx(h_i, d_k_i, d_out_i)]);
            dot_sat = sat16_49((dot_acc + 128) >>> 8);
            ao_val  = sat16_32((dot_sat * zr_val + 128) >>> 8);
            AO_MEM[ao_flat_idx(0, h_i, d_out_i)] = ao_val;
        end
    end

    for (load_i = 0; load_i < AO_ELEMS; load_i = load_i + 1)
        OUT_MEM[load_i] = 16'hxxxx;

    $display("[TB] attn_proj token %0d N_TOKENS=%0d", TOK_IDX, N_TOKENS);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk          = 1'b0;
    reset        = 1'b1;
    start        = 1'b0;
    ao_raddr_q   = 14'd0;
    w_addr_rom_q = 13'd0;
    b_addr_rom_q = 8'd0;

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

    for (cmp_i = 0; cmp_i < AO_ELEMS; cmp_i = cmp_i + 1) begin
        if (GOLD_OUT[cmp_i] !== OUT_MEM[cmp_i]) begin
            mism_out = mism_out + 32'd1;
            if (first_bad_out == 32'hFFFF_FFFF)
                first_bad_out = cmp_i;
        end
    end

    $display("---- attn_proj done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv_cnt=%0d out_mism=%0d", y_recv_cnt, mism_out);

    if (mism_out !== 0)
        $display("  [FAIL] attn_out mismatch count=%0d first_bad=%0d dut=%h gold=%h",
                 mism_out, first_bad_out, OUT_MEM[first_bad_out], GOLD_OUT[first_bad_out]);
    else
        $display("  [PASS] attn_proj matches backbone_blocks_0_after_attn_attn_out");

    $finish;
end

initial begin
    #(CYCLE * 2_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d", cycle_cnt, busy);
    $finish;
end

endmodule
