`timescale 1ns/10ps

// Compile from the module dir (cwd = verilog_backbone4/); include the generated
// SRAM macro in the VCS command line (NOT via `include):
//   vcs test/TEST_sglatrack_top.v Sram_1920.v +lint=TFIPC-L +notimingcheck \
//       +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\[PASS\]|\[FAIL\]|TIMEOUT|done' simv.log
//   # full-length: add +define+N_TEST=320
//
// WHY +notimingcheck (REQUIRED): Sram_1920 is a CLK=clk posedge macro fed a
//   streaming read address from posedge FFs inside layer_norm_par.  In zero-delay
//   RTL sim the address/CEB transition AT the sampling posedge, so the macro's
//   $hold(posedge CLK, negedge A / posedge CEB, limit:1) checks fire and the
//   TSMC notifier corrupts Q -> X (=> got=x).  These are spurious zero-delay
//   violations (real silicon has clk-to-q margin; resolved by SDF at gate level).
//   +notimingcheck disables the macro timing checks so Q is not X-corrupted.
//   NOTE: +define+TSMC_CM_NO_WARNING only mutes the warning text; it does NOT
//   stop the notifier, so it alone is not enough.
`include "sglatrack_top.v"
`include "layer_norm_par.v"
`include "inv_sqrt_nr.v"
`include "inv_sqrt_lut_seed.v"

// =============================================================================
// TEST_sglatrack_top.v -- backbone4 top (CURRENT STAGE: block0 norm1 LayerNorm)
//   connected to REAL activation SRAM macros (4 x Sram_1920, 112-bit),
//   dim192/Q7.7, LN_LANES=32.
//
// DUT: sglatrack_top  (4x Sram_1920 x banks + layer_norm_par).
//
// Golden (numpy trunk dim192/Q7.7, 14-bit two's complement per line):
//   x    : Activation/merged_tokens_bi.txt                    (token-major t*192+f)
//   g/b  : Weight/backbone_blocks_0_norm1_{weight,bias}_bi.txt (per-feature)
//   gold : Activation/backbone_blocks_0_after_norm1_out_bi.txt (FLOAT-math golden)
//
// PASS/FAIL = self-reference fixed-point oracle (bit-exact to DUT datapath,
//   incl. Q7.7 inv_sqrt_nr).  numpy float golden reported as info max-abs error.
//
// x SRAM: 4 banks share one word addr (tok*ROWS+row); bank b stores features
//   {row*32+b*8+s}.  TB preloads via pre_we/pre_addr/pre_wdata (448-bit row),
//   then asserts start.  Macro CLK=clk, posedge, 1-cycle latency.
//
// g/b ROM: posedge registered-data model (CLK=clk contract; 2-phase preload in
//   layer_norm_par absorbs the 1-cycle latency).
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

module TEST_sglatrack_top;

parameter CYCLE      = 2.0;
parameter FEAT_DIM   = 192;
parameter N_TOK_FULL = 320;
parameter LN_LANES   = 32;
parameter ROWS       = FEAT_DIM / LN_LANES;   // 6
parameter DATA_W     = 16;
parameter MACRO_W    = 14;
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
parameter BANK_AW    = 11;
parameter PRE_W      = LN_LANES * MACRO_W;    // 448 = 4 banks x 112

parameter N_TEST     = `N_TEST;
parameter NX         = N_TOK_FULL * FEAT_DIM;   // 61440

reg clk, reset, start;

// preload write
reg                       pre_we;
reg  [BANK_AW-1:0]        pre_addr;
reg  [PRE_W-1:0]          pre_wdata;

// gamma/beta ROM (posedge registered)
wire [FEAT_AW-1:0]        g_addr_o, b_addr_o;
wire signed [DATA_W-1:0]  g_i, b_i;
reg  signed [DATA_W-1:0]  g_i_r, b_i_r;

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

reg [31:0] y_recv_cnt, mism_cnt, cycle_cnt, max_float_diff;
integer    first_bad_idx;
reg signed [DATA_W-1:0] first_bad_ref, first_bad_got;

integer t, f, row, lane, pp, feat, idx, dd, beat_mism;
reg signed [DATA_W-1:0] gotv, refv, gv;

// oracle scratch
reg signed [63:0] S1, S2, num, var_pre, mtmp, vtmp, p1, p2, ci, yv;
reg signed [DATA_W-1:0] meanv, var_q77, inv77, cistd, wci;

// -------------------------------------------------------------------------
// DUT
// -------------------------------------------------------------------------
sglatrack_top #(
    .FEAT_DIM (FEAT_DIM),
    .N_TOKENS (N_TEST),
    .LN_LANES (LN_LANES)
) u_dut (
    .clk       (clk),
    .reset     (reset),
    .start     (start),
    .pre_we    (pre_we),
    .pre_addr  (pre_addr),
    .pre_wdata (pre_wdata),
    .g_addr_o  (g_addr_o),
    .g_i       (g_i),
    .b_addr_o  (b_addr_o),
    .b_i       (b_i),
    .busy      (busy),
    .done      (done),
    .y_o       (y_o),
    .y_valid   (y_valid),
    .y_tok_o   (y_tok_o),
    .y_row_o   (y_row_o)
);

// -------------------------------------------------------------------------
// gamma/beta ROM (CLK=clk posedge registered-data; 1-cycle latency)
// -------------------------------------------------------------------------
assign g_i = g_i_r;
assign b_i = b_i_r;

always #(CYCLE/2.0) clk = ~clk;

always @(posedge clk) begin
    g_i_r <= {{2{GMEM[g_addr_o][13]}}, GMEM[g_addr_o]};
    b_i_r <= {{2{BMEM[b_addr_o][13]}}, BMEM[b_addr_o]};
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
        mtmp  = (S1 * $signed(MEAN_RCP) + (64'sd1 <<< (MEAN_SHIFT-1))) >>> MEAN_SHIFT;
        meanv = clampS16(mtmp);
        num     = $signed(FEAT_DIM) * S2 - S1 * S1;
        var_pre = num * $signed(VAR_RCP);
        vtmp    = (var_pre + (64'sd1 <<< (VAR_SHIFT-1))) >>> VAR_SHIFT;
        var_q77 = (vtmp > SAT_MAX) ? SAT_MAX[15:0] :
                  (vtmp < 64'sd1)  ? 16'sd1 : vtmp[15:0];
        inv77   = inv_sqrt_oracle(var_q77);
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

    $display("[TB] sglatrack_top dim192/Q7.7  N_TEST=%0d LN_LANES=%0d banks=4xSram_1920(112b)", N_TEST, LN_LANES);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);

    clk      = 1'b0;
    reset    = 1'b1;
    start    = 1'b0;
    pre_we   = 1'b0;
    pre_addr = {BANK_AW{1'b0}};
    pre_wdata= {PRE_W{1'b0}};
    g_i_r    = {DATA_W{1'b0}};
    b_i_r    = {DATA_W{1'b0}};

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    // ---- preload activation into the 4 SRAM banks (one row = 32 features) ----
    for (t = 0; t < N_TEST; t = t + 1) begin
        for (row = 0; row < ROWS; row = row + 1) begin
            @(negedge clk);
            pre_we   = 1'b1;
            pre_addr = t*ROWS + row;
            for (lane = 0; lane < LN_LANES; lane = lane + 1)
                pre_wdata[lane*MACRO_W +: MACRO_W] =
                    MERGED[t*FEAT_DIM + row*LN_LANES + lane];
        end
    end
    @(negedge clk);
    pre_we   = 1'b0;
    pre_addr = {BANK_AW{1'b0}};
    pre_wdata= {PRE_W{1'b0}};

    // ---- run ----
    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- sglatrack_top done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d", y_recv_cnt, N_TEST*FEAT_DIM, mism_cnt);
    $display("[TB] info: max |dut - numpy_float_golden| (Q7.7 LSB) = %0d", max_float_diff);

    if (y_recv_cnt !== N_TEST*FEAT_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, N_TEST*FEAT_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] self-ref mismatch count=%0d first_bad idx=%0d ref=%0d got=%0d",
                 mism_cnt, first_bad_idx, first_bad_ref, first_bad_got);
    else
        $display("  [PASS] sglatrack_top matches Q7.7 self-reference (all %0d outputs)",
                 N_TEST*FEAT_DIM);

    $finish;
end

initial begin
    #(CYCLE * 50_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d", cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
