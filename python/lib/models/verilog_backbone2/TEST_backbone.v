`timescale 1ns/10ps

// =============================================================================
// TEST_backbone.v -- backbone_top end-to-end test (blocks 0..6 + backbone_norm)
//
// Flow:
//   1. Load template + search post-embedding from TXT_File/Activation/
//   2. reset; sel_block_i = 6; pulse start
//   3. backbone_top runs blocks 0..START_LAYER, adaptive block 6, backbone_norm
//   4. Compare y_o stream vs golden; print [PASS] or [FAIL] at done
//   5. On done (or timeout): $toggle_stop / $toggle_report -> backbone_top_rtl.saif
//
// Golden: ./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt
// Run simv from a directory where ./TXT_File/Activation/ resolves.
//
// VCS (run from directory where ./TXT_File/Activation/ resolves):
//
//   vcs verilog_backbone2/*.v memory/*.v \
//     <path>/Sram_tok1.v <path>/Sram_tok2.v \
//     <path>/Sram_x.v <path>/Sram_q.v <path>/Sram_k.v <path>/Sram_v.v <path>/Sram_qkm.v \
//     +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//
//   ./simv | tee simv.log
//   # -> backbone_top_rtl.saif, backbone_tb.fsdb
//
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|backbone_top done' simv.log
//
//   Block0 only (fast debug): +define+CHECK_BLOCK0_ONLY
// =============================================================================

module TEST_backbone;

parameter CYCLE = 2.0;

parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;
parameter FEAT_SZ     = FEAT_H * FEAT_W;
parameter TEMPL_TOT   = LENS_Z  * EMBED_DIM;
parameter SRCH_TOT    = FEAT_SZ * EMBED_DIM;
parameter TOK_TOTAL   = TEMPL_TOT + SRCH_TOT;
parameter N_TOKENS    = TOK_TOTAL / EMBED_DIM;

reg         clk, reset, start;
reg  [3:0]  sel_block_i;
reg  signed [15:0] data_in;
reg                data_valid;

wire        busy;
wire        done;
wire signed [15:0] y_o;
wire        y_valid;

backbone_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_DUT (
    .clk        (clk),
    .reset      (reset),
    .start      (start),
    .sel_block_i(sel_block_i),
    .x_i        (data_in),
    .x_valid    (data_valid),
    .busy       (busy),
    .done       (done),
    .y_o        (y_o),
    .y_valid    (y_valid)
);

wire tb_stream_gate =
    !u_DUT.tok_replay &&
    (u_DUT.u_tb.state == 4'd1);

reg [15:0] GOLD_BB [0:TOK_TOTAL-1];
reg [13:0] rtl_bb_cnt;
reg [31:0] bb_mism;
reg [31:0] bb_first_bad;

always #(CYCLE/2.0) clk = ~clk;

reg [31:0] cycle_cnt;
always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

reg [15:0] TEMPL_MEM [0:TEMPL_TOT-1];
reg [15:0] SRCH_MEM  [0:SRCH_TOT-1];
reg [13:0] tok_cnt;

always @(posedge clk) begin
    if (reset) begin
        tok_cnt    <= 14'd0;
        data_in    <= 16'sd0;
        data_valid <= 1'b0;
    end else if ((start || busy) && tb_stream_gate) begin
        if (tok_cnt < TEMPL_TOT) begin
            data_valid <= 1'b1;
            data_in    <= TEMPL_MEM[tok_cnt];
            tok_cnt    <= tok_cnt + 14'd1;
        end else if (tok_cnt < TOK_TOTAL) begin
            data_valid <= 1'b1;
            data_in    <= SRCH_MEM[tok_cnt - TEMPL_TOT];
            tok_cnt    <= tok_cnt + 14'd1;
        end else begin
            data_valid <= 1'b0;
            data_in    <= 16'sd0;
        end
    end else if (start || busy) begin
        data_valid <= 1'b0;
        data_in    <= 16'sd0;
    end
end

always @(posedge clk) begin
    if (reset) begin
        rtl_bb_cnt   <= 14'd0;
        bb_mism      <= 32'd0;
        bb_first_bad <= 32'hFFFF_FFFF;
    end else if (y_valid) begin
        if (GOLD_BB[rtl_bb_cnt] !== y_o[15:0]) begin
            bb_mism <= bb_mism + 32'd1;
            if (bb_first_bad == 32'hFFFF_FFFF)
                bb_first_bad <= {18'b0, rtl_bb_cnt};
        end
        rtl_bb_cnt <= rtl_bb_cnt + 14'd1;
    end
end

always @(negedge clk) begin

    if (done) begin
        $display("\n---- backbone_top done @ cycle %0d ----", cycle_cnt);
        $display("  sel_block_i = %0d", sel_block_i);
        $display("  y_valid samples = %0d (expect %0d)", rtl_bb_cnt, TOK_TOTAL);
        if (rtl_bb_cnt != TOK_TOTAL)
            $display("  [FAIL] sample count mismatch");
        else if (bb_mism == 0)
            $display("  [PASS] backbone_after_norm_backbone_out matches golden (%0d elems)",
                     TOK_TOTAL);
        else
            $display("  [FAIL] backbone_after_norm_backbone_out mismatches = %0d / %0d  first_bad_idx = %0d",
                     bb_mism, TOK_TOTAL, bb_first_bad);
        $toggle_stop();
        $toggle_report("backbone_top_rtl.saif", 1.0e-9, "u_DUT");
        $finish;
    end
end

initial begin
    // $fsdbDumpfile("backbone_tb.fsdb");
    // $fsdbDumpvars;
    // $fsdbDumpMDA;

    $set_toggle_region("u_DUT");

    $toggle_start();
    $readmemb("./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt", GOLD_BB);
    $readmemb("./TXT_File/Activation/template_post_embed_input_bi.txt", TEMPL_MEM);
    $readmemb("./TXT_File/Activation/search_post_embed_input_bi.txt", SRCH_MEM);
    $display("[TB] Loaded inputs from ./TXT_File/Activation/  template(%0d) search(%0d)",
             TEMPL_TOT, SRCH_TOT);

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 0;
    tok_cnt    = 0;
    rtl_bb_cnt = 0;
    bb_mism    = 0;
    bb_first_bad = 32'hFFFF_FFFF;

    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    #(CYCLE * 500_000_000);
    $display("[TB] TIMEOUT: backbone_top did not finish (cycle %0d)", cycle_cnt);
    $toggle_stop();
    $toggle_report("backbone_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
