`timescale 1ns/10ps
`include "sglatrack_top.v"
// =============================================================================
// TEST_backbone.v -- sglatrack_top(backbone-only) end-to-end test (Plan B)
//
// Flow:
//   1. Load template + search post-embedding from TXT_File/Activation/
//   2. reset; sel_block_i = 6; pulse start
//   3. sglatrack_top runs blocks 0..START_LAYER, adaptive block 6, backbone_norm
//   4. On done: 2-phase readback Sram_tok1 vs golden (no y_o stream)
//   5. On done (or timeout): $toggle_stop / $toggle_report -> backbone_top_rtl.saif
//
// Golden: ./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt
// Run simv from a directory where ./TXT_File/Activation/ resolves.
//
// VCS (run from directory where ./TXT_File/Activation/ resolves):
//
//   vcs verilog_backbone2/*.v memory/*.v \
//     <path>/Sram_tok1.v <path>/Sram_tok2.v \
//     <path>/Sram_q.v <path>/Sram_k.v <path>/Sram_v.v <path>/Sram_qkm.v \
//     +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//
//   ./simv | tee simv.log
//
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|backbone_top done|readback done' simv.log
// =============================================================================

module TEST_backbone;

// must write X.0, can't write x
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

reg               tok1_readback;
reg [13:0]        tok1_readback_addr;

wire        busy;
wire        x_ready;
wire        done;
wire signed [15:0] y_o;
wire        y_valid;
wire [15:0] tok1_readback_q;

sglatrack_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_DUT (
    .clk               (clk),
    .reset             (reset),
    .start             (start),
    .sel_block_i       (sel_block_i),
    .data_in           (data_in),
    .data_valid        (data_valid),
    .busy              (busy),
    .x_ready           (x_ready),
    .done              (done),
    .data_o            (y_o),
    .data_o_valid      (y_valid),
    .tok1_readback     (tok1_readback),
    .tok1_readback_addr(tok1_readback_addr),
    .tok1_readback_q   (tok1_readback_q)
);

reg [15:0] GOLD_BB [0:TOK_TOTAL-1];

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
    end else if ((start || busy) && x_ready) begin
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

// 2-phase read Sram_tok1 (CLK=~clk): posedge T ADDR, posedge T+1 sample Q
task readback_check_sram_tok1;
    input [31:0] expect_cnt;
    integer      idx;
    reg [15:0]   sample;
    reg [31:0]   mism;
    reg [31:0]   first_bad;
    begin
        mism      = 32'd0;
        first_bad = 32'hFFFF_FFFF;
        tok1_readback      <= 1'b0;
        tok1_readback_addr <= 14'd0;
        @(posedge clk);
        for (idx = 0; idx < expect_cnt; idx = idx + 1) begin
            @(posedge clk);
            tok1_readback      <= 1'b1;
            tok1_readback_addr <= idx[13:0];
            @(posedge clk);
            sample = tok1_readback_q;
            tok1_readback      <= 1'b0;
            if (GOLD_BB[idx] !== sample) begin
                mism = mism + 32'd1;
                if (first_bad == 32'hFFFF_FFFF)
                    first_bad = idx;
            end
        end
        tok1_readback      <= 1'b0;
        tok1_readback_addr <= 14'd0;
        $display("[TB] Sram_tok1 readback done (%0d words, token-major)", expect_cnt);
        if (mism == 0)
            $display("  [PASS] backbone_after_norm_backbone_out matches golden (%0d elems)",
                     expect_cnt);
        else
            $display("  [FAIL] backbone_after_norm_backbone_out mismatches = %0d / %0d  first_bad_idx = %0d",
                     mism, expect_cnt, first_bad);
    end
endtask

reg done_seen;

always @(posedge clk) begin
    if (reset)
        done_seen <= 1'b0;
    else if (done)
        done_seen <= 1'b1;
end

initial begin
    $fsdbDumpfile("sglatrack_top.fsdb");
    $fsdbDumpvars;
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
    tok1_readback      = 1'b0;
    tok1_readback_addr = 14'd0;
    done_seen  = 1'b0;
    cycle_cnt  = 0;
    tok_cnt    = 0;

    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    wait (done_seen === 1'b1);
    @(posedge clk);
    $display("\n---- backbone_top done @ cycle %0d ----", cycle_cnt);
    $display("  sel_block_i = %0d", sel_block_i);
    readback_check_sram_tok1(TOK_TOTAL);
    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

initial begin
    #(CYCLE * 500_000_000);
    $display("[TB] TIMEOUT: backbone_top did not finish (cycle %0d)", cycle_cnt);
    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
