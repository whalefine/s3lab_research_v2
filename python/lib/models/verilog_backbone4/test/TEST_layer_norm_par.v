`timescale 1ns/10ps

// Compile from the module dir (cwd = verilog_backbone4/):
//   vcs test/TEST_layer_norm_par.v +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\[PASS\]|\[FAIL\]|TIMEOUT|done' simv.log
//   # full-length cycle measurement: add +define+N_TEST=320
`include "layer_norm_par.v"
`include "inv_sqrt_nr.v"
`include "inv_sqrt_lut_seed.v"

// =============================================================================
// TEST_layer_norm_par.v -- block0 LayerNorm (norm1), dim192/Q7.7, LN_LANES=32
//
// DUT: layer_norm_par  FEAT_DIM=192  LN_LANES=32  (P-lane, one-pass variance).
//
// Golden (numpy trunk dim192/Q7.7, 14-bit two's complement per line):
//   x    : Activation/merged_tokens_bi.txt                    (token-major t*192+f)
//   g/b  : Weight/backbone_blocks_0_norm1_{weight,bias}_bi.txt (per-feature)
//   gold : Activation/backbone_blocks_0_after_norm1_out_bi.txt (FLOAT-math golden)
//
// PASS/FAIL reference = SELF-REFERENCE fixed-point oracle that replicates the
//   DUT datapath BIT-EXACTLY, including the Q8.8 inv_sqrt_nr (seed LUT + 3 NR)
//   ported below.  The numpy norm1 golden used true float mean/var/1-sqrt and is
//   reported as informational max-abs error only (it can NOT bit-match the RTL).
//
// x bank model: LN_LANES banks share one addr (tok*ROWS + row); bank p returns
//   feature row*LANES+p = MERGED[tok*192 + row*32 + p].  addr latched on negedge,
//   data valid at posedge T+1 (CLK=~clk contract, same as linear_ws).
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif
`ifndef GOLDEN_WGT
`define GOLDEN_WGT "./TXT_File/Weight"
`endif
`ifndef N_TEST
`define N_TEST 8
`endif

module TEST_layer_norm_par;

parameter CYCLE      = 2.0;
parameter FEAT_DIM   = 192;
parameter N_TOK_FULL = 320;
parameter LN_LANES   = 32;
parameter ROWS       = FEAT_DIM / LN_LANES;   // 6
parameter DATA_W     = 16;
parameter FRAC       = 7;
parameter SAT_MAX    = 8191;
parameter SAT_MIN    = -8192;
parameter S16_MAX    = 32767;
parameter S16_MIN    = -32768;
parameter MEAN_SHIFT = 22;
parameter MEAN_RCP   = 21845;
parameter VAR_SHIFT  = 40;
parameter VAR_RCP    = 233018;

parameter FEAT_AW    = 8;
parameter X_AW       = 12;
parameter TOK_AW     = 10;
parameter ROW_AW     = 4;

parameter N_TEST     = `N_TEST;
parameter NX         = N_TOK_FULL * FEAT_DIM;   // 61440

reg clk, reset, start;

wire [FEAT_AW-1:0]        g_addr_o;
wire [FEAT_AW-1:0]        b_addr_o;
wire signed [DATA_W-1:0]  g_i, b_i;
wire                      x_rd_en;
wire [X_AW-1:0]           x_rd_addr;
wire [LN_LANES*DATA_W-1:0] x_q;
wire                      busy, done;
wire [LN_LANES*DATA_W-1:0] y_o;
wire                      y_valid;
wire [TOK_AW-1:0]         y_tok_o;
wire [ROW_AW-1:0]         y_row_o;

// 14-bit golden storage; sign-extend on use
reg [13:0] MERGED [0:NX-1];
reg [13:0] GOLD   [0:NX-1];
reg [13:0] GMEM   [0:FEAT_DIM-1];
reg [13:0] BMEM   [0:FEAT_DIM-1];

reg signed [DATA_W-1:0] REF [0:N_TEST*FEAT_DIM-1];

// posedge registered-data read outputs (CLK=clk contract; no negedge latch)
reg signed [DATA_W-1:0]   g_i_r, b_i_r;
reg [LN_LANES*DATA_W-1:0] x_q_r;     // registered x bus
reg [LN_LANES*DATA_W-1:0] x_q_comb;  // combinational bus from current x_rd_addr

reg [31:0] y_recv_cnt, mism_cnt, cycle_cnt, max_float_diff;
integer    first_bad_idx;
reg signed [DATA_W-1:0] first_bad_ref, first_bad_got;

integer t, f, pp, feat, idx, dd, beat_mism;
integer xbp, xbi;
reg signed [DATA_W-1:0] gotv, refv, gv;

// oracle scratch
reg signed [63:0] S1, S2, num, var_pre, mtmp, vtmp, p1, p2, ci, yv;
reg signed [DATA_W-1:0] meanv, var_q77, inv77, cistd, wci;

`ifdef LN_DBG
// debug: oracle token0 stats (one-shot localisation vs DUT internals)
reg signed [63:0] orc_S1, orc_S2, orc_num, orc_varpre;
reg signed [DATA_W-1:0] orc_mean, orc_var, orc_inv;
reg dbg_shot;
reg [5:0] dbg_nc;
`endif

// -------------------------------------------------------------------------
// DUT
// -------------------------------------------------------------------------
layer_norm_par #(
    .FEAT_DIM (FEAT_DIM),
    .N_TOKENS (N_TEST),
    .LN_LANES (LN_LANES)
) u_dut (
    .clk       (clk),
    .reset     (reset),
    .start     (start),
    .g_addr_o  (g_addr_o),
    .g_i       (g_i),
    .b_addr_o  (b_addr_o),
    .b_i       (b_i),
    .x_rd_en   (x_rd_en),
    .x_rd_addr (x_rd_addr),
    .x_q       (x_q),
    .busy      (busy),
    .done      (done),
    .y_o       (y_o),
    .y_valid   (y_valid),
    .y_tok_o   (y_tok_o),
    .y_row_o   (y_row_o)
);

// -------------------------------------------------------------------------
// ROM / bank read models (CLK=clk posedge registered-data; no negedge latch)
//   posedge T: addr (g_addr_o/b_addr_o/x_rd_addr) stable
//   posedge T+1: q register updated  ->  data valid at T+1 (=+1 vs old ~clk model)
// -------------------------------------------------------------------------
assign g_i = g_i_r;
assign b_i = b_i_r;
assign x_q = x_q_r;

// build the LN_LANES-wide x bus from the CURRENT (un-latched) x_rd_addr
always @(*) begin
    for (xbp = 0; xbp < LN_LANES; xbp = xbp + 1) begin
        xbi = (x_rd_addr / ROWS) * FEAT_DIM + (x_rd_addr % ROWS) * LN_LANES + xbp;
        x_q_comb[xbp*DATA_W +: DATA_W] = {{2{MERGED[xbi][13]}}, MERGED[xbi]};
    end
end

always #(CYCLE/2.0) clk = ~clk;

// posedge synchronous read: register memory contents addressed in the prev cycle
always @(posedge clk) begin
    g_i_r <= {{2{GMEM[g_addr_o][13]}}, GMEM[g_addr_o]};
    b_i_r <= {{2{BMEM[b_addr_o][13]}}, BMEM[b_addr_o]};
    x_q_r <= x_q_comb;
end

always @(posedge clk) begin
    if (reset) cycle_cnt <= 32'd0;
    else       cycle_cnt <= cycle_cnt + 32'd1;
end

// -------------------------------------------------------------------------
// inv_sqrt oracle (bit-exact replica of inv_sqrt_nr / inv_sqrt_lut_seed)
// -------------------------------------------------------------------------
function [15:0] seed_lut;
    input [15:0] v;
    reg [3:0] k;
    begin
        casez (v)
            16'b1???????????????: k = 4'd15;
            16'b01??????????????: k = 4'd14;
            16'b001?????????????: k = 4'd13;
            16'b0001????????????: k = 4'd12;
            16'b00001???????????: k = 4'd11;
            16'b000001??????????: k = 4'd10;
            16'b0000001?????????: k = 4'd9;
            16'b00000001????????: k = 4'd8;
            16'b000000001???????: k = 4'd7;
            16'b0000000001??????: k = 4'd6;
            16'b00000000001?????: k = 4'd5;
            16'b000000000001????: k = 4'd4;
            16'b0000000000001???: k = 4'd3;
            16'b00000000000001??: k = 4'd2;
            16'b000000000000001?: k = 4'd1;
            default:              k = 4'd0;
        endcase
        case (k)        // Q7.7: y0 = round(128/sqrt(1.5*2^(k-7)))
            4'd15: seed_lut = 16'd7;
            4'd14: seed_lut = 16'd9;
            4'd13: seed_lut = 16'd13;
            4'd12: seed_lut = 16'd18;
            4'd11: seed_lut = 16'd26;
            4'd10: seed_lut = 16'd37;
            4'd9:  seed_lut = 16'd52;
            4'd8:  seed_lut = 16'd74;
            4'd7:  seed_lut = 16'd105;
            4'd6:  seed_lut = 16'd148;
            4'd5:  seed_lut = 16'd209;
            4'd4:  seed_lut = 16'd296;
            4'd3:  seed_lut = 16'd418;
            4'd2:  seed_lut = 16'd591;
            4'd1:  seed_lut = 16'd836;
            4'd0:  seed_lut = 16'd1182;
            default: seed_lut = 16'd105;
        endcase
    end
endfunction

function signed [15:0] inv_sqrt_oracle;
    input signed [15:0] vin;
    reg signed [15:0] y, ysq, term, coeff;
    reg signed [31:0] t1, t2, t3;
    integer it;
    begin
        y = seed_lut(vin);
        for (it = 0; it < 3; it = it + 1) begin
            t1    = $signed(y) * $signed(y) + 32'sd64;      ysq  = t1[22:7];
            t2    = $signed(vin) * $signed(ysq) + 32'sd128; term = t2[23:8];
            coeff = 16'sd192 - term;
            t3    = $signed(y) * coeff + 32'sd64;           y    = t3[22:7];
        end
        inv_sqrt_oracle = y;
    end
endfunction

function signed [15:0] clampS16;
    input signed [63:0] v;
    begin
        clampS16 = (v > S16_MAX) ? S16_MAX[15:0] :
                   (v < S16_MIN) ? S16_MIN[15:0] : v[15:0];
    end
endfunction

// -------------------------------------------------------------------------
// scoring: each y_valid beat carries LN_LANES features of one (tok,row)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt     = 32'd0;
        mism_cnt       = 32'd0;
        max_float_diff = 32'd0;
        first_bad_idx  = -1;
        first_bad_ref  = 16'sd0;
        first_bad_got  = 16'sd0;
    end else if (y_valid) begin
        beat_mism = 0;
        for (pp = 0; pp < LN_LANES; pp = pp + 1) begin
            feat = y_row_o * LN_LANES + pp;
            idx  = y_tok_o * FEAT_DIM + feat;
            gotv = $signed(y_o[pp*DATA_W +: DATA_W]);
            refv = REF[idx];
            if (gotv !== refv) begin
                beat_mism = beat_mism + 1;
                if (first_bad_idx == -1) begin
                    first_bad_idx = idx;
                    first_bad_ref = refv;
                    first_bad_got = gotv;
                end
            end
            gv = {{2{GOLD[idx][13]}}, GOLD[idx]};
            dd = gotv - gv;
            if (dd < 0) dd = -dd;
            if (dd > max_float_diff) max_float_diff = dd[31:0];
        end
        y_recv_cnt = y_recv_cnt + LN_LANES;
        mism_cnt   = mism_cnt + beat_mism;
    end
end

`ifdef LN_DBG
// -------------------------------------------------------------------------
// DEBUG (one-shot, compile with +define+LN_DBG): localise divergence on token0
// (accum / stats / inv_sqrt). Samples DUT internals the first cycle it enters
// S_NORM (=4'd9), where token0's mean/var/inv_std are still held.
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        dbg_shot <= 1'b0;
    end else if (!dbg_shot && u_dut.state == 4'd9) begin
        dbg_shot <= 1'b1;
        $display("[LN_DBG] tok0 S1   dut=%0d orc=%0d", $signed(u_dut.S1),  orc_S1);
        $display("[LN_DBG] tok0 S2   dut=%0d orc=%0d", $signed(u_dut.S2),  orc_S2);
        $display("[LN_DBG] tok0 num  dut=%0d orc=%0d", $signed(u_dut.num), orc_num);
        $display("[LN_DBG] tok0 vpre dut=%0d orc=%0d", $signed(u_dut.var_pre), orc_varpre);
        $display("[LN_DBG] tok0 mean dut=%0d orc=%0d", $signed(u_dut.mean_q77), orc_mean);
        $display("[LN_DBG] tok0 var  dut=%0d orc=%0d", $signed(u_dut.var_q77),  orc_var);
        $display("[LN_DBG] tok0 inv  dut=%0d orc=%0d", $signed(u_dut.inv_std_q77), orc_inv);
        $display("[LN_DBG] tok0 f0 gbuf=%0d bbuf=%0d REF=%0d GOLD=%0d",
                 $signed(u_dut.gbuf[0]), $signed(u_dut.bbuf[0]),
                 $signed(REF[0]), $signed({{2{GOLD[0][13]}}, GOLD[0]}));
    end
end

// DEBUG (trace): token0 NORM pipeline, lane0, first ~12 beats. Localises which
// stage (ci -> *invstd -> *gamma -> +beta) blows up to the 8191 saturation.
always @(posedge clk) begin
    if (reset) begin
        dbg_nc <= 6'd0;
    end else if (u_dut.tok_cnt == {TOK_AW{1'b0}} && u_dut.state == 4'd9 && dbg_nc < 6'd12) begin
        dbg_nc <= dbg_nc + 6'd1;
        $display("[LN_TR] nc=%0d v4=%b rn3=%0d rn4=%0d ci0=%0d cistd0=%0d wci0=%0d y0=%0d",
                 u_dut.norm_cnt, u_dut.v4, u_dut.rn3, u_dut.rn4,
                 $signed(u_dut.ci_n[0]), $signed(u_dut.cistd_n[0]),
                 $signed(u_dut.wci_n[0]), $signed(u_dut.y_o[15:0]));
    end
end
`endif

// -------------------------------------------------------------------------
// Stimulus + self-reference build
// -------------------------------------------------------------------------
initial begin
    $readmemb({`GOLDEN_ACT, "/merged_tokens_bi.txt"}, MERGED);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"}, GOLD);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm1_weight_bi.txt"}, GMEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_norm1_bias_bi.txt"},   BMEM);

    // ---- self-reference fixed-point oracle (bit-exact to DUT) ----
    for (t = 0; t < N_TEST; t = t + 1) begin
        S1 = 64'sd0;
        S2 = 64'sd0;
        for (f = 0; f < FEAT_DIM; f = f + 1) begin
            ci = $signed({{2{MERGED[t*FEAT_DIM+f][13]}}, MERGED[t*FEAT_DIM+f]});
            S1 = S1 + ci;
            S2 = S2 + ci * ci;
        end
        // mean = round(S1 / 192)
        mtmp  = (S1 * $signed(MEAN_RCP) + (64'sd1 <<< (MEAN_SHIFT-1))) >>> MEAN_SHIFT;
        meanv = clampS16(mtmp);
        // var = round((192*S2 - S1*S1) / (192*192*128))
        num     = $signed(FEAT_DIM) * S2 - S1 * S1;
        var_pre = num * $signed(VAR_RCP);
        vtmp    = (var_pre + (64'sd1 <<< (VAR_SHIFT-1))) >>> VAR_SHIFT;
        var_q77 = (vtmp > SAT_MAX) ? SAT_MAX[15:0] :
                  (vtmp < 64'sd1)  ? 16'sd1 : vtmp[15:0];
        inv77   = inv_sqrt_oracle(var_q77);   // Q7.7-native, no shift
`ifdef LN_DBG
        if (t == 0) begin
            orc_S1 = S1; orc_S2 = S2; orc_num = num; orc_varpre = var_pre;
            orc_mean = meanv; orc_var = var_q77; orc_inv = inv77;
        end
`endif
        for (f = 0; f < FEAT_DIM; f = f + 1) begin
            ci    = $signed({{2{MERGED[t*FEAT_DIM+f][13]}}, MERGED[t*FEAT_DIM+f]}) -
                    $signed({{48{meanv[15]}}, meanv});
            p1    = ci * $signed({{48{inv77[15]}}, inv77});
            cistd = clampS16((p1 + 64'sd64) >>> FRAC);
            p2    = $signed({{48{cistd[15]}}, cistd}) *
                    $signed({{2{GMEM[f][13]}}, GMEM[f]});
            wci   = clampS16((p2 + 64'sd64) >>> FRAC);
            yv    = $signed({{48{wci[15]}}, wci}) +
                    $signed({{2{BMEM[f][13]}}, BMEM[f]});
            REF[t*FEAT_DIM+f] = (yv > SAT_MAX) ? SAT_MAX[15:0] :
                                (yv < SAT_MIN) ? SAT_MIN[15:0] : yv[15:0];
        end
    end

    $display("[TB] layer_norm_par dim192/Q7.7  N_TEST=%0d LN_LANES=%0d", N_TEST, LN_LANES);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);

    clk      = 1'b0;
    reset    = 1'b1;
    start    = 1'b0;
    g_i_r    = {DATA_W{1'b0}};
    b_i_r    = {DATA_W{1'b0}};
    x_q_r    = {(LN_LANES*DATA_W){1'b0}};
    x_q_comb = {(LN_LANES*DATA_W){1'b0}};

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- layer_norm_par done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d", y_recv_cnt, N_TEST*FEAT_DIM, mism_cnt);
    $display("[TB] info: max |dut - numpy_float_golden| (Q7.7 LSB) = %0d", max_float_diff);

    if (y_recv_cnt !== N_TEST*FEAT_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, N_TEST*FEAT_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] self-ref mismatch count=%0d first_bad idx=%0d ref=%0d got=%0d",
                 mism_cnt, first_bad_idx, first_bad_ref, first_bad_got);
    else
        $display("  [PASS] layer_norm_par matches Q7.7 self-reference (all %0d outputs)",
                 N_TEST*FEAT_DIM);

    $finish;
end

initial begin
    #(CYCLE * 50_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d", cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
