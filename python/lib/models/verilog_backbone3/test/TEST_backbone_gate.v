`timescale 1ns/10ps
`include "sglatrack_top_syn.v"
`include "memory2/Sram_tok1.v"
`include "memory2/Sram_tok2.v"
`include "memory2/Sram_q.v"
`include "memory2/Sram_16384.v"
`include "memory2/rom_backbone_blocks_0_3_attn_qkv_weight.v"
`include "memory2/rom_backbone_blocks_0_3_mlp_fc1_weight.v"
`include "memory2/rom_backbone_blocks_0_3_mlp_fc2_weight.v"
`include "memory2/rom_backbone_blocks_0_6_attn_proj_bias.v"
`include "memory2/rom_backbone_blocks_0_6_attn_proj_weight.v"
`include "memory2/rom_backbone_blocks_0_6_attn_qkv_bias.v"
`include "memory2/rom_backbone_blocks_0_6_mlp_fc1_bias.v"
`include "memory2/rom_backbone_blocks_0_6_mlp_fc2_bias.v"
`include "memory2/rom_backbone_blocks_0_6_norm1_bias.v"
`include "memory2/rom_backbone_blocks_0_6_norm1_weight.v"
`include "memory2/rom_backbone_blocks_0_6_norm2_bias.v"
`include "memory2/rom_backbone_blocks_0_6_norm2_weight.v"
`include "memory2/rom_backbone_blocks_4_6_attn_qkv_weight.v"
`include "memory2/rom_backbone_blocks_4_6_mlp_fc1_weight.v"
`include "memory2/rom_backbone_blocks_4_6_mlp_fc2_weight.v"
`include "memory2/rom_backbone_norm_bias.v"
`include "memory2/rom_backbone_norm_weight.v"
`include "memory2/rom_box_head_shared_conv1_2_folded_bias.v"
`include "memory2/rom_box_head_shared_conv1_folded_weight1.v"
`include "memory2/rom_box_head_shared_conv1_folded_weight2.v"
`include "memory2/rom_box_head_shared_conv2_folded_weight1.v"
`include "memory2/rom_box_head_shared_conv2_folded_weight2.v"
`include "memory2/rom_box_head_shared_conv2_folded_weight3.v"
`include "memory2/rom_box_head_tail_ctr_offset_size_bias.v"
`include "memory2/rom_box_head_tail_ctr_offset_size_weight.v"
`include "tcbn16ffcllbwp7d5t20p96cpd.v"
// =============================================================================
// TEST_backbone_gate.v -- sglatrack_top(backbone-only) GLS final-check
//
// Flat netlist: no hierarchy tap. After done, read Sram_tok1 via top tok1_readback_*.
// Requires sglatrack_top_syn from RTL that keeps readback mux (verilog_backbone3
// sglatrack_top.v Plan A; do NOT tie tok1_readback_q off under +define+SYNTHESIS).
// SDC: set_false_path on tok1_readback* (see verilog_backbone3/synthesis/sglatrack_top.sdc).
//
// Golden: ./TXT_File/Activation/
//
// VCS (from sim root with sglatrack_top_syn.v, memory2/, stdcell .v):
//   vcs test/TEST_backbone_gate.v \
//       +lint=TFIPC-L +define+TSMC_CM_NO_WARNING \
//       +notimingcheck +delay_mode_zero \
//       -debug_access+all -debug_region+cell | tee runvcs_gate.log
//   ./simv | tee simv_gate.log
//
// read_saif (if used in synthesis): -instance u_DUT  (not TEST_backbone/u_DUT)
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

module TEST_backbone_gate;

parameter CYCLE = 3.0;

parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;
parameter FEAT_SZ     = FEAT_H * FEAT_W;
parameter TEMPL_TOT   = LENS_Z  * EMBED_DIM;
parameter SRCH_TOT    = FEAT_SZ * EMBED_DIM;
parameter TOK_TOTAL   = TEMPL_TOT + SRCH_TOT;
parameter N_TOKENS    = TOK_TOTAL / EMBED_DIM;
parameter TOK_FLAT    = TOK_TOTAL;

reg         clk, reset, start;
reg  [3:0]  sel_block_i;
reg  signed [15:0] data_in;
reg                data_valid;

reg                tok1_readback;
reg [13:0]         tok1_readback_addr;

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

reg [15:0] READ_TOK1 [0:TOK_FLAT-1];
reg [15:0] GOLD_BB   [0:TOK_FLAT-1];

reg [31:0] cycle_cnt;
reg [15:0] TEMPL_MEM [0:TEMPL_TOT-1];
reg [15:0] SRCH_MEM  [0:SRCH_TOT-1];
reg [13:0] tok_cnt;
reg        done_seen;

always #(CYCLE/2.0) clk = ~clk;

always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

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

// SRAM read contract (1P, CLK = ~clk):
//   posedge T:   tok1_readback_addr
//   posedge T+1: tok1_readback_q valid for addr@T
task readback_tok1_all;
    input [31:0] word_cnt;
    integer      idx;
    begin
        tok1_readback = 1'b1;
        for (idx = 0; idx < word_cnt; idx = idx + 1) begin
            tok1_readback_addr = idx[13:0];
            @(posedge clk);
            @(posedge clk);
            READ_TOK1[idx] = tok1_readback_q;
        end
        tok1_readback      = 1'b0;
        tok1_readback_addr = 14'd0;
    end
endtask

task compare_final_bb_norm;
    input [255:0] tag;
    input [31:0]  expect_cnt;
    integer       idx;
    reg [31:0]    mism;
    reg [31:0]    first_bad;
    reg [15:0]    rtl_bad;
    reg [15:0]    gold_bad;
    begin
        mism      = 32'd0;
        first_bad = 32'hFFFF_FFFF;
        for (idx = 0; idx < expect_cnt; idx = idx + 1) begin
            if (GOLD_BB[idx] !== READ_TOK1[idx]) begin
                mism = mism + 32'd1;
                if (first_bad == 32'hFFFF_FFFF) begin
                    first_bad = idx;
                    rtl_bad   = READ_TOK1[idx];
                    gold_bad  = GOLD_BB[idx];
                end
            end
        end
        $display("[FINAL] %0s @ cycle %0d  (tok1_readback %0d words)",
                tag, cycle_cnt, expect_cnt);
        if (mism == 0)
            $display("  [PASS] %0s", tag);
        else
            $display("  [FAIL] %0s  mismatches = %0d / %0d  first_bad_idx = %0d  RTL = %04h  GOLD = %04h",
                    tag, mism, expect_cnt, first_bad, rtl_bad, gold_bad);
    end
endtask

always @(posedge clk) begin
    if (reset)
        done_seen <= 1'b0;
    else if (done)
        done_seen <= 1'b1;
end

initial begin
`ifdef DUMP_FSDB
    $fsdbDumpfile("sglatrack_top_gate.fsdb");
    $fsdbDumpvars;
`endif

    $set_toggle_region("u_DUT");
    $toggle_start();

    $readmemb({`GOLDEN_ACT, "/backbone_after_norm_backbone_out_bi.txt"}, GOLD_BB);
    $readmemb({`GOLDEN_ACT, "/template_post_embed_input_bi.txt"},        TEMPL_MEM);
    $readmemb({`GOLDEN_ACT, "/search_post_embed_input_bi.txt"},         SRCH_MEM);

    $display("[TB] backbone E2E final-check (GLS)  N_TOKENS=%0d  TOK_FLAT=%0d", N_TOKENS, TOK_FLAT);
    $display("[TB] Golden dir: %s", `GOLDEN_ACT);
    $display("[TB] Compare: backbone_after_norm_backbone_out_bi.txt vs tok1_readback_q");
    $display("[TB] CYCLE=%0.1f ns  (match synthesis clock period)", CYCLE);

    clk                = 0;
    reset              = 1;
    start              = 0;
    data_in            = 16'sd0;
    data_valid         = 1'b0;
    tok1_readback      = 1'b0;
    tok1_readback_addr = 14'd0;
    done_seen          = 1'b0;
    cycle_cnt          = 0;
    tok_cnt            = 0;

    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    $display("[TB] start pulsed @ cycle %0d", cycle_cnt);

    wait (done_seen === 1'b1);
    @(posedge clk);
    $display("\n---- backbone_top done @ cycle %0d ----", cycle_cnt);
    $display("  sel_block_i = %0d", sel_block_i);

    readback_tok1_all(TOK_FLAT);
    compare_final_bb_norm("BACKBONE_NORM backbone_after_norm_backbone_out", TOK_FLAT);

    $toggle_stop();
    $toggle_report("sglatrack_top_gate.saif", 1.0e-9, "u_DUT");
    $finish;
end

initial begin
    #(CYCLE * 80_000_000);
    $display("[TB] TIMEOUT: backbone_top did not finish (cycle %0d)", cycle_cnt);
    $toggle_stop();
    $toggle_report("sglatrack_top_gate.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
