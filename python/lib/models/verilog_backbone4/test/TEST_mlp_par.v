`timescale 1ns/10ps

// Compile from the module dir (cwd = verilog_backbone4/):
//   vcs test/TEST_mlp_par.v +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\[PASS\]|\[FAIL\]|mism|info' simv.log
//   # smaller/faster: default N_TEST=8 (hidden reg array sized N_TEST*HIDDEN)
`include "mlp_par.v"

// =============================================================================
// TEST_mlp_par.v -- block0 MLP (192 -> 768 -> ReLU -> 192) on the B2 engine
//   reused for BOTH fc1 and fc2 (single MAC array, two passes).  dim192/Q7.7.
//
// Golden (numpy trunk dim192/Q7.7, 14-bit two's complement per line):
//   x      : Activation/backbone_blocks_0_after_norm2_out_bi.txt   (t*192+f)
//   W1/B1  : Weight/backbone_blocks_0_mlp_fc1_{weight,bias}_bi.txt  (neuron-major)
//   W2/B2  : Weight/backbone_blocks_0_mlp_fc2_{weight,bias}_bi.txt  (neuron-major)
//   mlpout : Activation/backbone_blocks_0_mlp_after_mlp_out_bi.txt  (t*192+n)  [INFO]
//
// PASS/FAIL = integer SELF-REFERENCE (truncating, matches linear_ws/linear_par):
//   h1[n1] = sat14( (sum_f x*w1) >>> 7 + b1 ) ; h2 = max(0,h1) ;
//   y [n2] = sat14( (sum_k h2*w2) >>> 7 + b2 ).
//   (numpy mlp golden uses float matmul + round -> a few LSB off; reported as
//    info only, like linear_ws's "max |dut - numpy_float_golden|".)
//
// x SRAM model : P=32 features/word, token-major (EMBED=192 = 6*32).
// W/B ROM model: CLK=clk posedge registered (2-phase preload absorbs latency).
// y stream     : one beat/token = LANES=16 neurons (fc2 out=192 -> 12 groups).
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

module TEST_mlp_par;

parameter CYCLE      = 2.0;
parameter EMBED      = 192;
parameter HIDDEN     = 768;
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
parameter W1_AW      = 20;
parameter W2_AW      = 20;
parameter B_AW       = 10;
parameter NEU_AW     = 10;
parameter TOK_AW     = 10;

parameter NX  = N_TOK_FULL * EMBED;    // 61440  (x)
parameter NW1 = HIDDEN * EMBED;        // 147456 (fc1 w)
parameter NB1 = HIDDEN;                // 768
parameter NW2 = EMBED  * HIDDEN;       // 147456 (fc2 w)
parameter NB2 = EMBED;                 // 192
parameter NG  = N_TOK_FULL * EMBED;    // 61440  (mlp golden)

reg clk, reset, start;

wire                       x_rd_en;
wire [XW_AW-1:0]           x_rd_addr;
wire [P*DATA_W-1:0]        x_q;
wire [W1_AW-1:0]           w1_addr_o;
wire signed [DATA_W-1:0]   w1_i;
wire [B_AW-1:0]            b1_addr_o;
wire signed [DATA_W-1:0]   b1_i;
wire [W2_AW-1:0]           w2_addr_o;
wire signed [DATA_W-1:0]   w2_i;
wire [B_AW-1:0]            b2_addr_o;
wire signed [DATA_W-1:0]   b2_i;
wire                       busy, done;
wire signed [LANES*DATA_W-1:0] y_o;
wire                       y_valid;
wire [TOK_AW-1:0]          y_tok_o;
wire [NEU_AW-1:0]          y_neu_base_o;

// 14-bit golden storage (Q7.7); sign-extend to 16-bit on use
reg [13:0] NORM2 [0:NX-1];
reg [13:0] W1MEM [0:NW1-1];
reg [13:0] B1MEM [0:NB1-1];
reg [13:0] W2MEM [0:NW2-1];
reg [13:0] B2MEM [0:NB2-1];
reg [13:0] GMLP  [0:NG-1];

// self-reference results
reg signed [DATA_W-1:0] H2    [0:N_TEST*HIDDEN-1];  // ReLU(fc1) hidden
reg signed [DATA_W-1:0] REF_Y [0:N_TEST*EMBED-1];   // fc2 output

// posedge synchronous read-data registers (CLK=clk, no negedge)
reg signed [DATA_W-1:0] w1_i_r, b1_i_r, w2_i_r, b2_i_r;
reg [P*DATA_W-1:0]      x_q_r;

reg [31:0] y_recv_cnt, mism_cnt, cycle_cnt;
integer    first_bad_tok, first_bad_neu;
reg signed [DATA_W-1:0] first_bad_ref, first_bad_got;
integer    max_gdiff, gdiff;

integer t, n, f, k, lane, neu, beat_mism, xp;
reg signed [ACC_W-1:0]  accv;
reg signed [DATA_W-1:0] xv, wv, bv, h1v, gotv, refv, gv;

// -------------------------------------------------------------------------
// DUT
// -------------------------------------------------------------------------
mlp_par #(
    .EMBED    (EMBED),
    .HIDDEN   (HIDDEN),
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
    .w1_addr_o    (w1_addr_o),
    .w1_i         (w1_i),
    .b1_addr_o    (b1_addr_o),
    .b1_i         (b1_i),
    .w2_addr_o    (w2_addr_o),
    .w2_i         (w2_i),
    .b2_addr_o    (b2_addr_o),
    .b2_i         (b2_i),
    .busy         (busy),
    .done         (done),
    .y_o          (y_o),
    .y_valid      (y_valid),
    .y_tok_o      (y_tok_o),
    .y_neu_base_o (y_neu_base_o)
);

// -------------------------------------------------------------------------
// ROM / x read models: CLK=clk posedge synchronous read (registered data out)
// -------------------------------------------------------------------------
assign w1_i = w1_i_r;
assign b1_i = b1_i_r;
assign w2_i = w2_i_r;
assign b2_i = b2_i_r;
assign x_q  = x_q_r;

always #(CYCLE/2.0) clk = ~clk;

always @(posedge clk) begin
    w1_i_r <= {{2{W1MEM[w1_addr_o][13]}}, W1MEM[w1_addr_o]};
    b1_i_r <= {{2{B1MEM[b1_addr_o][13]}}, B1MEM[b1_addr_o]};
    w2_i_r <= {{2{W2MEM[w2_addr_o][13]}}, W2MEM[w2_addr_o]};
    b2_i_r <= {{2{B2MEM[b2_addr_o][13]}}, B2MEM[b2_addr_o]};
    for (xp = 0; xp < P; xp = xp + 1)
        x_q_r[xp*DATA_W +: DATA_W] <=
            {{2{NORM2[x_rd_addr*P + xp][13]}}, NORM2[x_rd_addr*P + xp]};
end

always @(posedge clk) begin
    if (reset) cycle_cnt <= 32'd0;
    else       cycle_cnt <= cycle_cnt + 32'd1;
end

// -------------------------------------------------------------------------
// Compare each y beat (fc2 output) against self-reference; track golden diff
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt    <= 32'd0;
        mism_cnt      <= 32'd0;
        first_bad_tok <= -1;
        first_bad_neu <= -1;
        first_bad_ref <= 16'sd0;
        first_bad_got <= 16'sd0;
        max_gdiff     <= 0;
    end else if (y_valid) begin
        beat_mism = 0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            neu  = y_neu_base_o + lane;
            gotv = $signed(y_o[lane*DATA_W +: DATA_W]);
            refv = REF_Y[y_tok_o*EMBED + neu];
            gv   = {{2{GMLP[y_tok_o*EMBED + neu][13]}}, GMLP[y_tok_o*EMBED + neu]};
            gdiff = (gotv > gv) ? (gotv - gv) : (gv - gotv);
            if (gdiff > max_gdiff) max_gdiff <= gdiff;
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
    end
end

// -------------------------------------------------------------------------
// Stimulus + self-reference build
// -------------------------------------------------------------------------
initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm2_out_bi.txt"}, NORM2);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_weight_bi.txt"},  W1MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc1_bias_bi.txt"},    B1MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_weight_bi.txt"},  W2MEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_mlp_fc2_bias_bi.txt"},    B2MEM);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_mlp_after_mlp_out_bi.txt"}, GMLP);

    // ---- self-reference (Q7.7 truncating MAC) -----------------------------
    // pass1: h2[t][n1] = ReLU( sat14( (sum_f x*w1) >>> 7 + b1 ) )
    for (t = 0; t < N_TEST; t = t + 1) begin
        for (n = 0; n < HIDDEN; n = n + 1) begin
            accv = {ACC_W{1'b0}};
            for (f = 0; f < EMBED; f = f + 1) begin
                xv = {{2{NORM2[t*EMBED+f][13]}}, NORM2[t*EMBED+f]};
                wv = {{2{W1MEM[n*EMBED+f][13]}}, W1MEM[n*EMBED+f]};
                accv = accv + xv * wv;
            end
            accv = accv >>> FRAC_BITS;
            bv   = {{2{B1MEM[n][13]}}, B1MEM[n]};
            accv = accv + {{(ACC_W-DATA_W){bv[DATA_W-1]}}, bv};
            h1v  = (accv > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
                   (accv < SAT_MIN) ? SAT_MIN[DATA_W-1:0] : accv[DATA_W-1:0];
            H2[t*HIDDEN+n] = (h1v < 0) ? 16'sd0 : h1v;   // ReLU
        end
    end
    // pass2: y[t][n2] = sat14( (sum_k h2*w2) >>> 7 + b2 )
    for (t = 0; t < N_TEST; t = t + 1) begin
        for (n = 0; n < EMBED; n = n + 1) begin
            accv = {ACC_W{1'b0}};
            for (k = 0; k < HIDDEN; k = k + 1) begin
                wv   = {{2{W2MEM[n*HIDDEN+k][13]}}, W2MEM[n*HIDDEN+k]};
                accv = accv + H2[t*HIDDEN+k] * wv;
            end
            accv = accv >>> FRAC_BITS;
            bv   = {{2{B2MEM[n][13]}}, B2MEM[n]};
            accv = accv + {{(ACC_W-DATA_W){bv[DATA_W-1]}}, bv};
            REF_Y[t*EMBED+n] = (accv > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
                               (accv < SAT_MIN) ? SAT_MIN[DATA_W-1:0] : accv[DATA_W-1:0];
        end
    end

    $display("[TB] mlp_par dim192/Q7.7  N_TEST=%0d LANES=%0d P=%0d (fc1 192->768 ReLU, fc2 768->192)",
             N_TEST, LANES, P);
    $display("[TB] GOLDEN_ACT=%s GOLDEN_WGT=%s", `GOLDEN_ACT, `GOLDEN_WGT);

    clk    = 1'b0;
    reset  = 1'b1;
    start  = 1'b0;
    w1_i_r = {DATA_W{1'b0}};
    b1_i_r = {DATA_W{1'b0}};
    w2_i_r = {DATA_W{1'b0}};
    b2_i_r = {DATA_W{1'b0}};
    x_q_r  = {(P*DATA_W){1'b0}};

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- mlp_par done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d", y_recv_cnt, N_TEST*EMBED, mism_cnt);
    $display("[TB] info: max |dut - numpy_float_golden| (Q7.7 LSB) = %0d", max_gdiff);

    if (y_recv_cnt !== N_TEST*EMBED)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, N_TEST*EMBED);
    else if (mism_cnt !== 0)
        $display("  [FAIL] self-ref mismatch count=%0d first_bad tok=%0d neu=%0d ref=%0d got=%0d",
                 mism_cnt, first_bad_tok, first_bad_neu, first_bad_ref, first_bad_got);
    else
        $display("  [PASS] mlp_par matches Q7.7 self-reference (all %0d outputs)",
                 N_TEST*EMBED);

    $finish;
end

// timeout guard
initial begin
    #(CYCLE * 4000000);
    $display("  [FAIL] TIMEOUT");
    $finish;
end

endmodule
