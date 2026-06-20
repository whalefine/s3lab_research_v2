`timescale 1ns/10ps

// Compile from the module dir (cwd = verilog_backbone4/):
//   vcs test/TEST_linear_par.v +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\[PASS\]|\[FAIL\]|TIMEOUT|done' simv.log
//   # full-length cycle measurement: add +define+N_TEST=320
`include "linear_par.v"

// =============================================================================
// TEST_linear_par.v -- block0 MLP fc1 (192->768) on the B2 engine (dim192/Q7.7)
//
// DUT: linear_par IN_DIM=192 OUT_DIM=768 LANES=16 P=32 (feature-parallel,
//   pipelined adder tree), weight-stationary.
//
// Golden (numpy trunk dim192/Q7.7, 14-bit two's complement per line):
//   x   : Activation/backbone_blocks_0_after_norm2_out_bi.txt   (token-major t*192+f)
//   W/B : Weight/backbone_blocks_0_mlp_fc1_{weight,bias}_bi.txt  (neuron-major)
//
// PASS/FAIL = SELF-REFERENCE integer MAC (truncating, matches linear_ws):
//   y = sat14( (sum_f x_int*w_int) >>> 7 + bias_int ).  Bit-exact target for the
//   reorganized P-parallel adder-tree summation (full precision, no rounding).
//   (No fc1 float golden node exists, so no numpy info diff here.)
//
// x SRAM model: P=32 features/word, token-major.  word a = features [a*32, a*32+31]
//   = NORM2[a*32 +: 32]  (since 192 = 6*32).  CLK=clk posedge, 1-cycle latency.
// W/B ROM model: CLK=clk posedge registered-data (2-phase preload absorbs latency).
//
// y stream: one beat per token = LANES=16 neurons (neu_base..neu_base+15) for
//   token y_tok_o; lane l -> neuron (y_neu_base_o + l).
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

module TEST_linear_par;

parameter CYCLE      = 2.0;
parameter IN_DIM     = 192;
parameter OUT_DIM    = 768;
parameter LANES      = 16;
parameter P          = 32;
parameter N_TOK_FULL = 320;
parameter N_TEST     = `N_TEST;
parameter DATA_W     = 16;
parameter ACC_W      = 48;
parameter FRAC_BITS  = 7;
parameter SAT_MAX    = 8191;
parameter SAT_MIN    = -8192;
parameter XW_AW      = 14;
parameter W_ADDR_W   = 20;
parameter B_ADDR_W   = 10;
parameter NEU_AW     = 10;
parameter TOK_AW     = 10;

parameter NX = N_TOK_FULL * IN_DIM;   // 61440
parameter NW = OUT_DIM * IN_DIM;      // 147456
parameter NB = OUT_DIM;               // 768

reg clk, reset, start;

wire                       x_rd_en;
wire [XW_AW-1:0]           x_rd_addr;
wire [P*DATA_W-1:0]        x_q;
wire [W_ADDR_W-1:0]        w_addr_o;
wire signed [DATA_W-1:0]   w_i;
wire [B_ADDR_W-1:0]        b_addr_o;
wire signed [DATA_W-1:0]   b_i;
wire                       busy, done;
wire signed [LANES*DATA_W-1:0] y_o;
wire                       y_valid;
wire [TOK_AW-1:0]          y_tok_o;
wire [NEU_AW-1:0]          y_neu_base_o;

// 14-bit golden storage (Q7.7); sign-extend to 16-bit on use
reg [13:0] NORM2 [0:NX-1];
reg [13:0] WMEM  [0:NW-1];
reg [13:0] BMEM  [0:NB-1];

reg signed [DATA_W-1:0] REF_Y [0:N_TEST*OUT_DIM-1];

// posedge synchronous read-data registers (CLK = clk, no negedge)
reg signed [DATA_W-1:0] w_i_r;
reg signed [DATA_W-1:0] b_i_r;
reg [P*DATA_W-1:0]      x_q_r;

reg [31:0] y_recv_cnt, mism_cnt, cycle_cnt;
integer    first_bad_tok, first_bad_neu;
reg signed [DATA_W-1:0] first_bad_ref, first_bad_got;

integer t, n, f, lane, neu, beat_mism, xp;
reg signed [ACC_W-1:0] accv;
reg signed [DATA_W-1:0] xv, wv, bv, gotv, refv;

// -------------------------------------------------------------------------
// DUT
// -------------------------------------------------------------------------
linear_par #(
    .IN_DIM   (IN_DIM),
    .OUT_DIM  (OUT_DIM),
    .N_TOKENS (N_TEST),
    .LANES    (LANES),
    .P        (P)
) u_dut (
    .clk          (clk),
    .reset        (reset),
    .start        (start),
    .x_rd_en      (x_rd_en),
    .x_rd_addr    (x_rd_addr),
    .x_q          (x_q),
    .w_addr_o     (w_addr_o),
    .w_i          (w_i),
    .b_addr_o     (b_addr_o),
    .b_i          (b_i),
    .busy         (busy),
    .done         (done),
    .y_o          (y_o),
    .y_valid      (y_valid),
    .y_tok_o      (y_tok_o),
    .y_neu_base_o (y_neu_base_o)
);

// -------------------------------------------------------------------------
// ROM / x read models: CLK=clk posedge synchronous read (registered data out)
// x word a = 32 features [a*32 +: 32] (token-major; 192 = 6*32)
// -------------------------------------------------------------------------
assign w_i = w_i_r;
assign b_i = b_i_r;
assign x_q = x_q_r;

always #(CYCLE/2.0) clk = ~clk;

always @(posedge clk) begin
    w_i_r <= {{2{WMEM[w_addr_o][13]}}, WMEM[w_addr_o]};
    b_i_r <= {{2{BMEM[b_addr_o][13]}}, BMEM[b_addr_o]};
    for (xp = 0; xp < P; xp = xp + 1)
        x_q_r[xp*DATA_W +: DATA_W] <=
            {{2{NORM2[x_rd_addr*P + xp][13]}}, NORM2[x_rd_addr*P + xp]};
end

always @(posedge clk) begin
    if (reset) cycle_cnt <= 32'd0;
    else       cycle_cnt <= cycle_cnt + 32'd1;
end

// -------------------------------------------------------------------------
// Compare each y beat (LANES neurons) against self-reference
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt    <= 32'd0;
        mism_cnt      <= 32'd0;
        first_bad_tok <= -1;
        first_bad_neu <= -1;
        first_bad_ref <= 16'sd0;
        first_bad_got <= 16'sd0;
    end else if (y_valid) begin
        beat_mism = 0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            neu  = y_neu_base_o + lane;
            gotv = $signed(y_o[lane*DATA_W +: DATA_W]);
            refv = REF_Y[y_tok_o*OUT_DIM + neu];
            if (gotv !== refv) begin
                beat_mism = beat_mism + 1;
                if (first_bad_neu == -1) begin
                    first_bad_tok <= y_tok_o;
                    first_bad_neu <= neu;
                    first_bad_ref <= refv;
                    first_bad_got <= gotv;
                end
            end
        end
        y_recv_cnt <= y_recv_cnt + LANES;
        mism_cnt   <= mism_cnt + beat_mism;
        // ---- DEBUG: dump first beats lane-by-lane (sim-only) -------------
        // diagnoses self-ref FAIL pattern: which lanes / tokens are wrong,
        // is got a saturate (8191)?  enable with +define+LP_DBG
`ifdef LP_DBG
        if (y_recv_cnt < 32'd64) begin
            for (lane = 0; lane < LANES; lane = lane + 1)
                $display("[LPDBG] tok=%0d nbase=%0d lane=%0d neu=%0d ref=%0d got=%0d",
                         y_tok_o, y_neu_base_o, lane, y_neu_base_o+lane,
                         REF_Y[y_tok_o*OUT_DIM + y_neu_base_o + lane],
                         $signed(y_o[lane*DATA_W +: DATA_W]));
        end
`endif
    end
end

// -------------------------------------------------------------------------
// DEBUG: dump DUT internals at token0/group0 emit cycle (acc_val&&acc_last).
// distinguishes "sum wrong" (sum_full huge) vs "shift/sat wrong" (sum_full ok
// but shr_b huge).  Also dump group0 w_buf for a PASS lane(0) vs FAIL lane(1).
// enable with +define+LP_DBG2
// -------------------------------------------------------------------------
`ifdef LP_DBG2
integer di;
reg dbg_done2;
always @(posedge clk) begin
    if (reset) dbg_done2 <= 1'b0;
    else if (!dbg_done2 && u_dut.acc_val && u_dut.acc_last &&
             u_dut.acc_tok == 0 && u_dut.neu_base == 0) begin
        dbg_done2 <= 1'b1;
        for (di = 0; di < LANES; di = di + 1)
            $display("[LPDBG2] lane=%0d acc=%0d a5=%0d sum_full=%0d shr_b=%0d sat=%0d",
                     di, u_dut.acc[di], u_dut.a5[di], u_dut.sum_full[di],
                     u_dut.shr_b[di], u_dut.sat_lane_w[di]);
        $display("[LPDBG2] grp0 w_buf[0][0..2]=%0d %0d %0d  w_buf[1][0..2]=%0d %0d %0d",
                 u_dut.LANE[0].w_buf[0], u_dut.LANE[0].w_buf[1], u_dut.LANE[0].w_buf[2],
                 u_dut.LANE[1].w_buf[0], u_dut.LANE[1].w_buf[1], u_dut.LANE[1].w_buf[2]);
        $display("[LPDBG2] exp w[0][0..2]=%0d %0d %0d  w[1][0..2]=%0d %0d %0d",
                 $signed({{2{WMEM[0][13]}},WMEM[0]}), $signed({{2{WMEM[1][13]}},WMEM[1]}),
                 $signed({{2{WMEM[2][13]}},WMEM[2]}),
                 $signed({{2{WMEM[IN_DIM][13]}},WMEM[IN_DIM]}),
                 $signed({{2{WMEM[IN_DIM+1][13]}},WMEM[IN_DIM+1]}),
                 $signed({{2{WMEM[IN_DIM+2][13]}},WMEM[IN_DIM+2]}));
    end
end
`endif

// -------------------------------------------------------------------------
// Stimulus + self-reference build
// -------------------------------------------------------------------------
initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm2_out_bi.txt"}, NORM2);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_weight_bi.txt"}, WMEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_bias_bi.txt"},   BMEM);

    // self-reference integer MAC (Q7.7): y = sat14( (sum x*w) >>> 7 + bias )
    for (t = 0; t < N_TEST; t = t + 1) begin
        for (n = 0; n < OUT_DIM; n = n + 1) begin
            accv = {ACC_W{1'b0}};
            for (f = 0; f < IN_DIM; f = f + 1) begin
                xv = {{2{NORM2[t*IN_DIM+f][13]}}, NORM2[t*IN_DIM+f]};
                wv = {{2{WMEM[n*IN_DIM+f][13]}}, WMEM[n*IN_DIM+f]};
                accv = accv + xv * wv;
            end
            accv = accv >>> FRAC_BITS;
            bv   = {{2{BMEM[n][13]}}, BMEM[n]};
            accv = accv + {{(ACC_W-DATA_W){bv[DATA_W-1]}}, bv};
            REF_Y[t*OUT_DIM+n] = (accv > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
                                 (accv < SAT_MIN) ? SAT_MIN[DATA_W-1:0] :
                                 accv[DATA_W-1:0];
        end
    end

    $display("[TB] linear_par fc1 dim192/Q7.7  N_TEST=%0d LANES=%0d P=%0d", N_TEST, LANES, P);
    $display("[TB] GOLDEN_ACT=%s GOLDEN_WGT=%s", `GOLDEN_ACT, `GOLDEN_WGT);

    clk   = 1'b0;
    reset = 1'b1;
    start = 1'b0;
    w_i_r = {DATA_W{1'b0}};
    b_i_r = {DATA_W{1'b0}};
    x_q_r = {(P*DATA_W){1'b0}};

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- linear_par done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d", y_recv_cnt, N_TEST*OUT_DIM, mism_cnt);

    if (y_recv_cnt !== N_TEST*OUT_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, N_TEST*OUT_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] self-ref mismatch count=%0d first_bad tok=%0d neu=%0d ref=%0d got=%0d",
                 mism_cnt, first_bad_tok, first_bad_neu, first_bad_ref, first_bad_got);
    else
        $display("  [PASS] linear_par matches Q7.7 self-reference (all %0d outputs)",
                 N_TEST*OUT_DIM);

    // ---- DEBUG: after done, w_buf holds the LAST group (neu OUT_DIM-LANES..)
    // check lane0 and lane15 weights/bias actually landed (hierarchical, sim-only)
`ifdef LP_DBG
    $display("[LPDBG] --- last-group w/b load check (base neu=%0d) ---", OUT_DIM-LANES);
    $display("[LPDBG] w_buf[0][0]    =%0d exp=%0d", u_dut.LANE[0].w_buf[0],
             $signed({{2{WMEM[(OUT_DIM-LANES)*IN_DIM][13]}}, WMEM[(OUT_DIM-LANES)*IN_DIM]}));
    $display("[LPDBG] w_buf[15][0]   =%0d exp=%0d", u_dut.LANE[15].w_buf[0],
             $signed({{2{WMEM[(OUT_DIM-1)*IN_DIM][13]}}, WMEM[(OUT_DIM-1)*IN_DIM]}));
    $display("[LPDBG] w_buf[15][191] =%0d exp=%0d", u_dut.LANE[15].w_buf[191],
             $signed({{2{WMEM[(OUT_DIM-1)*IN_DIM+191][13]}}, WMEM[(OUT_DIM-1)*IN_DIM+191]}));
    $display("[LPDBG] bias[0]=%0d exp=%0d  bias[15]=%0d exp=%0d",
             u_dut.bias_reg[0],  $signed({{2{BMEM[OUT_DIM-LANES][13]}}, BMEM[OUT_DIM-LANES]}),
             u_dut.bias_reg[15], $signed({{2{BMEM[OUT_DIM-1][13]}}, BMEM[OUT_DIM-1]}));
`endif

    $finish;
end

initial begin
    #(CYCLE * 50_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d", cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
