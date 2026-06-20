`timescale 1ns/10ps

// Compile from the module dir (cwd = verilog_backbone4/), same convention as
// backbone3: `vcs test/TEST_linear_ws.v ...`. The bare include below resolves
// relative to that cwd.
`include "linear_ws.v"

// =============================================================================
// TEST_linear_ws.v -- block0 QKV weight-stationary linear unit test (dim192/Q7.7)
//
// DUT: linear_ws IN_DIM=192 OUT_DIM=576 LANES=64 (QKV), neuron-parallel WS.
//
// Golden (numpy trunk dim192/Q7.7, 14-bit two's complement per line):
//   x   : Activation/backbone_blocks_0_after_norm1_out_bi.txt   (token-major)
//   W/B : Weight/backbone_blocks_0_attn_qkv_{weight,bias}_bi.txt (neuron-major)
//   q/k/v (informational only): Activation/backbone_blocks_0_attn_after_qkv_{q,k,v}_bi.txt
//
// PASS/FAIL reference = SELF-REFERENCE integer MAC using the SAME Q7.7 weights
//   the DUT reads from ROM:  y = sat14( (sum_f x*w) >>> 7 + bias ).
//   This is independent of the numpy q/k/v golden, which used RAW float weights
//   (qkv_w not quantized before linear) and therefore can NOT bit-match any RTL.
//   The q/k/v float golden is reported as informational max-abs error only.
//
// dim192 QKV layout: linear out neuron n in [0,576):
//   qkv_sel = n / 192 (0=q,1=k,2=v); loc=n%192; head=loc/64; dim=loc%64
//   numpy q/k/v saved (B,H,N,d) head-major: flat = head*320*64 + tok*64 + dim
//
// ROM/x read model: CLK=clk posedge synchronous read (registered data out).
//   addr@T sampled at posedge -> data valid @T+1; DUT consumes streaming x +1 later.
//
// Run a small token slice for fast golden check (full N=320 only needed for a
// real cycle/throughput number):  +define+N_TEST=320
//
// VCS (run from the module dir so include + golden paths + ./simv cwd all match):
//   cd python/lib/models/verilog_backbone4
//   vcs test/TEST_linear_ws.v +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//   grep -E '\[PASS\]|\[FAIL\]|TIMEOUT|linear_ws done' simv.log
//   # full-length cycle measurement: add +define+N_TEST=320
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

module TEST_linear_ws;

parameter CYCLE      = 2.0;
parameter IN_DIM     = 192;
parameter OUT_DIM    = 576;
parameter LANES      = 64;
parameter N_TOK_FULL = 320;   // golden file token count (q/k/v stride)
parameter N_TEST     = `N_TEST;
parameter HEAD_DIM   = 64;
parameter DATA_W     = 16;
parameter ACC_W      = 36;
parameter FRAC_BITS  = 7;
parameter SAT_MAX    = 8191;
parameter SAT_MIN    = -8192;
parameter X_AW       = 17;
parameter W_ADDR_W   = 18;
parameter B_ADDR_W   = 10;
parameter NEU_AW     = 10;
parameter TOK_AW     = 10;

parameter NX = N_TOK_FULL * IN_DIM;   // 61440
parameter NW = OUT_DIM * IN_DIM;      // 110592
parameter NB = OUT_DIM;               // 576

reg         clk;
reg         reset;
reg         start;

wire        x_rd_en;
wire [X_AW-1:0]  x_rd_addr;
wire signed [DATA_W-1:0] x_q;
wire [W_ADDR_W-1:0] w_addr_o;
wire signed [DATA_W-1:0] w_i;
wire [B_ADDR_W-1:0] b_addr_o;
wire signed [DATA_W-1:0] b_i;
wire        busy;
wire        done;
wire signed [DATA_W-1:0] y_o;
wire        y_valid;
wire [TOK_AW-1:0] y_tok_o;
wire [NEU_AW-1:0] y_neu_o;

// 14-bit golden storage (Q7.7); sign-extend to 16-bit on use
reg [13:0] NORM1 [0:NX-1];
reg [13:0] WMEM  [0:NW-1];
reg [13:0] BMEM  [0:NB-1];
reg [13:0] QG    [0:NX-1];
reg [13:0] KG    [0:NX-1];
reg [13:0] VG    [0:NX-1];

reg signed [DATA_W-1:0] REF_Y [0:N_TEST*OUT_DIM-1];

// posedge synchronous SRAM/ROM read-data registers (CLK = clk, no negedge)
reg signed [DATA_W-1:0] w_i_r;
reg signed [DATA_W-1:0] b_i_r;
reg signed [DATA_W-1:0] x_q_r;

reg [31:0] y_recv_cnt;
reg [31:0] mism_cnt;
reg [31:0] first_bad_tok;
reg [31:0] first_bad_neu;
reg [31:0] cycle_cnt;
reg [31:0] max_float_diff;

integer t, n, f;
reg signed [ACC_W-1:0] accv;
reg signed [DATA_W-1:0] xv, wv, bv;

integer qsel, loc, hd, dm, gflat, gdiff;
reg signed [DATA_W-1:0] gval, dut_val;

// -------------------------------------------------------------------------
// DUT
// -------------------------------------------------------------------------
linear_ws #(
    .IN_DIM   (IN_DIM),
    .OUT_DIM  (OUT_DIM),
    .N_TOKENS (N_TEST),
    .LANES    (LANES)
) u_dut (
    .clk       (clk),
    .reset     (reset),
    .start     (start),
    .x_rd_en   (x_rd_en),
    .x_rd_addr (x_rd_addr),
    .x_q       (x_q),
    .w_addr_o  (w_addr_o),
    .w_i       (w_i),
    .b_addr_o  (b_addr_o),
    .b_i       (b_i),
    .busy      (busy),
    .done      (done),
    .y_o       (y_o),
    .y_valid   (y_valid),
    .y_tok_o   (y_tok_o),
    .y_neu_o   (y_neu_o)
);

// ROM / x read models: CLK=clk posedge synchronous read (registered data out).
// addr sampled at posedge -> data valid next cycle; 2-phase preloads capture in
// phase1, streaming x is consumed +1 cycle later by the DUT (see linear_ws MAC).
assign w_i = w_i_r;
assign b_i = b_i_r;
assign x_q = x_q_r;

always #(CYCLE/2.0) clk = ~clk;

always @(posedge clk) begin
    w_i_r <= {{2{WMEM[w_addr_o][13]}}, WMEM[w_addr_o]};
    b_i_r <= {{2{BMEM[b_addr_o][13]}}, BMEM[b_addr_o]};
    x_q_r <= {{2{NORM1[x_rd_addr][13]}}, NORM1[x_rd_addr]};
end

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

// -------------------------------------------------------------------------
// Compare each y against self-reference; track float-golden max diff (info)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        y_recv_cnt     <= 32'd0;
        mism_cnt       <= 32'd0;
        first_bad_tok  <= 32'hFFFF_FFFF;
        first_bad_neu  <= 32'hFFFF_FFFF;
        max_float_diff <= 32'd0;
    end else if (y_valid) begin
        y_recv_cnt <= y_recv_cnt + 32'd1;
        // self-reference (Q7.7 weights) = PASS/FAIL oracle
        if (REF_Y[y_tok_o*OUT_DIM + y_neu_o] !== y_o) begin
            mism_cnt <= mism_cnt + 32'd1;
            if (first_bad_neu == 32'hFFFF_FFFF) begin
                first_bad_tok <= {22'b0, y_tok_o};
                first_bad_neu <= {22'b0, y_neu_o};
            end
        end
        // ---- DEBUG: dump first outputs to reveal mismatch pattern (sim-only)
        // diagnoses: self-ref FAIL — is got == ref shifted by neuron/feature?
        // enable with +define+WS_DBG; prints first 24 received y (tok/neu/ref/got)
`ifdef WS_DBG
        if (y_recv_cnt < 32'd24)
            $display("[WSDBG] tok=%0d neu=%0d ref=%0d got=%0d",
                     y_tok_o, y_neu_o,
                     REF_Y[y_tok_o*OUT_DIM + y_neu_o], y_o);
`endif
        // informational: distance to numpy float-weight golden (head-major)
        qsel = y_neu_o / IN_DIM;
        loc  = y_neu_o % IN_DIM;
        hd   = loc / HEAD_DIM;
        dm   = loc % HEAD_DIM;
        gflat = hd*N_TOK_FULL*HEAD_DIM + y_tok_o*HEAD_DIM + dm;
        if (qsel == 0)      gval = {{2{QG[gflat][13]}}, QG[gflat]};
        else if (qsel == 1) gval = {{2{KG[gflat][13]}}, KG[gflat]};
        else                gval = {{2{VG[gflat][13]}}, VG[gflat]};
        dut_val = y_o;
        gdiff = dut_val - gval;
        if (gdiff < 0) gdiff = -gdiff;
        if (gdiff > max_float_diff)
            max_float_diff <= gdiff[31:0];
    end
end

// -------------------------------------------------------------------------
// Stimulus + self-reference build
// -------------------------------------------------------------------------
initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_after_norm1_out_bi.txt"}, NORM1);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_weight_bi.txt"}, WMEM);
    $readmemb({`GOLDEN_WGT, "/backbone_blocks_0_attn_qkv_bias_bi.txt"},   BMEM);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_q_bi.txt"}, QG);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_k_bi.txt"}, KG);
    $readmemb({`GOLDEN_ACT, "/backbone_blocks_0_attn_after_qkv_v_bi.txt"}, VG);

    // self-reference integer MAC (Q7.7): y = sat14( (sum x*w) >>> 7 + bias )
    for (t = 0; t < N_TEST; t = t + 1) begin
        for (n = 0; n < OUT_DIM; n = n + 1) begin
            accv = {ACC_W{1'b0}};
            for (f = 0; f < IN_DIM; f = f + 1) begin
                xv = {{2{NORM1[t*IN_DIM+f][13]}}, NORM1[t*IN_DIM+f]};
                wv = {{2{WMEM[n*IN_DIM+f][13]}}, WMEM[n*IN_DIM+f]};
                accv = accv + xv * wv;
            end
            accv = accv >>> FRAC_BITS;
            bv   = {{2{BMEM[n][13]}}, BMEM[n]};
            accv = accv + {{(ACC_W-DATA_W){bv[DATA_W-1]}}, bv};
            if (accv > SAT_MAX)
                REF_Y[t*OUT_DIM+n] = SAT_MAX[DATA_W-1:0];
            else if (accv < SAT_MIN)
                REF_Y[t*OUT_DIM+n] = SAT_MIN[DATA_W-1:0];
            else
                REF_Y[t*OUT_DIM+n] = accv[DATA_W-1:0];
        end
    end

    $display("[TB] linear_ws QKV dim192/Q7.7  N_TEST=%0d LANES=%0d", N_TEST, LANES);
    $display("[TB] GOLDEN_ACT=%s", `GOLDEN_ACT);
    $display("[TB] GOLDEN_WGT=%s", `GOLDEN_WGT);

    clk      = 1'b0;
    reset    = 1'b1;
    start    = 1'b0;
    w_i_r    = {DATA_W{1'b0}};
    b_i_r    = {DATA_W{1'b0}};
    x_q_r    = {DATA_W{1'b0}};

    #(CYCLE);
    reset = 1'b0;
    #(CYCLE);

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done === 1'b1);
    @(posedge clk);

    $display("---- linear_ws done @ cycle %0d ----", cycle_cnt);
    $display("[TB] y_recv=%0d expect=%0d mism=%0d", y_recv_cnt, N_TEST*OUT_DIM, mism_cnt);
    $display("[TB] info: max |dut - numpy_float_golden| (Q7.7 LSB) = %0d", max_float_diff);

    if (y_recv_cnt !== N_TEST*OUT_DIM)
        $display("  [FAIL] y_valid count mismatch (got %0d expect %0d)",
                 y_recv_cnt, N_TEST*OUT_DIM);
    else if (mism_cnt !== 0)
        $display("  [FAIL] self-ref mismatch count=%0d first_bad tok=%0d neu=%0d",
                 mism_cnt, first_bad_tok, first_bad_neu);
    else
        $display("  [PASS] linear_ws matches Q7.7 self-reference (all %0d outputs)",
                 N_TEST*OUT_DIM);

    // ---- DEBUG: verify last-group weight/bias actually landed in DUT reg file
    // (hierarchical refs, sim-only). After done, w_lane holds the LAST group:
    // neurons [OUT_DIM-LANES .. OUT_DIM-1].  Enable with +define+WS_DBG.
`ifdef WS_DBG
    $display("[WSDBG] --- last-group w/b load check (base neu=%0d) ---", OUT_DIM-LANES);
    $display("[WSDBG] wlane[0][0]   =%0d exp=%0d", u_dut.w_lane[0][0],
             $signed({{2{WMEM[(OUT_DIM-LANES)*IN_DIM][13]}},   WMEM[(OUT_DIM-LANES)*IN_DIM]}));
    $display("[WSDBG] wlane[0][1]   =%0d exp=%0d", u_dut.w_lane[0][1],
             $signed({{2{WMEM[(OUT_DIM-LANES)*IN_DIM+1][13]}}, WMEM[(OUT_DIM-LANES)*IN_DIM+1]}));
    $display("[WSDBG] wlane[1][0]   =%0d exp=%0d", u_dut.w_lane[1][0],
             $signed({{2{WMEM[(OUT_DIM-LANES+1)*IN_DIM][13]}}, WMEM[(OUT_DIM-LANES+1)*IN_DIM]}));
    $display("[WSDBG] wlane[%0d][191]=%0d exp=%0d", LANES-1, u_dut.w_lane[LANES-1][191],
             $signed({{2{WMEM[(OUT_DIM-1)*IN_DIM+191][13]}},   WMEM[(OUT_DIM-1)*IN_DIM+191]}));
    $display("[WSDBG] bias[0]=%0d exp=%0d  bias[%0d]=%0d exp=%0d",
             u_dut.bias_reg[0],       $signed({{2{BMEM[OUT_DIM-LANES][13]}}, BMEM[OUT_DIM-LANES]}),
             LANES-1, u_dut.bias_reg[LANES-1], $signed({{2{BMEM[OUT_DIM-1][13]}},   BMEM[OUT_DIM-1]}));
`endif

    $finish;
end

initial begin
    #(CYCLE * 50_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d y_recv=%0d", cycle_cnt, busy, y_recv_cnt);
    $finish;
end

endmodule
