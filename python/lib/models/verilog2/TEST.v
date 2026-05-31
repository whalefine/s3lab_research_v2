`timescale 1ns/10ps

// =============================================================================
// TEST.v -- verilog2 full-chain testbench (sglatrack_top)
//
// DUT: sglatrack_top (RTL pulled in below via `include "sglatrack_top.v")
// VCS (cwd = verilog2/ so `include resolves; or add +incdir+.../verilog2):
//   vcs TEST.v memory/Sram_tok1.v memory/Sram_tok2.v memory/Sram_q.v \
//        memory/Sram_k.v memory/Sram_v.v memory/Sram_qkm.v memory/rom_*.v \
//        +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
// Do NOT also pass sglatrack_top.v or verilog2/*.v (duplicate module).
// Compile memory/: Sram_tok1.v Sram_tok2.v Sram_q.v Sram_k.v Sram_v.v Sram_qkm.v (one each)
//
// Inputs (golden activation only, not weights):
//   template_post_embed_input_bi.txt
//   search_post_embed_input_bi.txt
//
// Weights: internal ROM in backbone_top / conv / tail (memory/ at VCS compile)
//
// Golden bbox (Q8.8, box_head_after_cal_bbox_bbox_bi.txt):
//   cx=0x007F cy=0x007E w=0x0024 h=0x0057
//
// Backbone norm (Plan B): compare golden at norm SRAM write beat (no post-read).
//   Golden: ./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt
//   Do NOT force-read Sram_tok1 after bb_done (TSMC macro $hold on CEB vs CLK).
// =============================================================================

`include "sglatrack_top.v"

module TEST;

// must write X.0, can't write x
// if you write 2 must be wrong, you should write 2.0
parameter CYCLE = 2.0;
// FSDB: sim time = CYCLE * FSDB_START_MULT (ns). Must be < done (~cycle 69M -> ~138M ns).
// Old 100_000_000 -> 200M ns dumpon AFTER $finish -> fsdbreport shows NV for all signals.
parameter [31:0] FSDB_START_MULT     = 32'd28_000_000;  // 56M ns; before head phase
parameter [31:0] PROGRESS_STEP_MULT = 32'd10_000_000;

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

wire [15:0] cx_out, cy_out, w_out, h_out;
wire busy;
wire done;

sglatrack_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS),
    .FEAT_H    (FEAT_H),
    .FEAT_W    (FEAT_W)
) u_DUT (
    .clk        (clk),
    .reset      (reset),
    .start      (start),
    .sel_block_i(sel_block_i),
    .data_in    (data_in),
    .data_valid (data_valid),
    .busy       (busy),
    .done       (done),
    .cx_o       (cx_out),
    .cy_o       (cy_out),
    .w_o        (w_out),
    .h_o        (h_out)
);

// Feed tokens only during backbone block 0 S_LOAD_X (before tok_replay)
wire tb_stream_gate =
    !u_DUT.u_backbone.tok_replay &&
    (u_DUT.u_backbone.u_tb.state == 4'd1);

// Plan B: backbone norm golden check at write beat (see bb_norm_wr below)
wire             bb_done    = u_DUT.u_backbone.done;

// Plan B: backbone norm golden check at Sram_tok1 write beat (same as [BB_NORM_WR] din)
localparam BB_ST_NORM = 3'd3;

wire bb_norm_wr = (u_DUT.u_backbone.state == BB_ST_NORM) &&
                  u_DUT.u_backbone.bn_wr_do;
wire [13:0] bb_wr_flat = u_DUT.u_backbone.sram_tok1_addr_o;
wire [15:0] bb_wr_din  = u_DUT.u_backbone.sram_tok1_din_o;

reg [31:0] bb_wr_cnt;
reg [31:0] bb_wr_mism;

always #(CYCLE/2.0) clk = ~clk;

reg [31:0] cycle_cnt;
reg [31:0] prog_time_mult;

always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

// Progress heartbeat every #(CYCLE * PROGRESS_STEP_MULT); sim time labeled as CYCLE * N
initial begin : progress_monitor
    prog_time_mult = 32'd0;
    forever begin
        #(CYCLE * PROGRESS_STEP_MULT);
        prog_time_mult = prog_time_mult + PROGRESS_STEP_MULT;
        $display("[TB] sim_time CYCLE * %0d  (cycle_cnt=%0d busy=%0d done=%0d)",
                 prog_time_mult, cycle_cnt, busy, done);
    end
end

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

reg [15:0] GOLD_BB [0:TOK_TOTAL-1];
reg [31:0] bb_first_bad;
reg        bb_rpt_done;
reg        bb_done_d1;

wire bb_done_rise = bb_done & ~bb_done_d1;

// Compare golden at norm write beat (non-destructive; no SRAM force/readback).
always @(posedge clk) begin
    if (reset) begin
        bb_wr_cnt     <= 32'd0;
        bb_wr_mism    <= 32'd0;
        bb_first_bad  <= 32'hFFFF_FFFF;
        bb_rpt_done   <= 1'b0;
        bb_done_d1    <= 1'b0;
    end else begin
        bb_done_d1 <= bb_done;

        if (bb_norm_wr) begin
            if (bb_wr_din !== GOLD_BB[bb_wr_flat]) begin
                if (bb_wr_mism < 32'd8)
                    $display("  [TB_BB_WR] mismatch flat=%0d rtl=0x%04h golden=0x%04h cycle=%0d",
                             bb_wr_flat, bb_wr_din, GOLD_BB[bb_wr_flat], cycle_cnt);
                if (bb_first_bad == 32'hFFFF_FFFF)
                    bb_first_bad <= {18'b0, bb_wr_flat};
                bb_wr_mism <= bb_wr_mism + 32'd1;
            end
            bb_wr_cnt <= bb_wr_cnt + 32'd1;
        end

        if (bb_done_rise && !bb_rpt_done) begin
            bb_rpt_done <= 1'b1;
            $display("\n---- backbone_after_norm @ cycle %0d (write-beat check, Plan B) ----",
                     cycle_cnt);
            $display("  sel_block_i=%0d  norm_wr_cnt=%0d expect=%0d",
                     sel_block_i, bb_wr_cnt, TOK_TOTAL);
            if (bb_wr_cnt != TOK_TOTAL)
                $display("  [FAIL] norm write count mismatch (missing or extra beats)");
            else if (bb_wr_mism == 32'd0)
                $display("  [PASS] backbone_after_norm write data matches golden (%0d elems)",
                         TOK_TOTAL);
            else
                $display("  [FAIL] backbone_after_norm write mismatches=%0d / %0d first_bad=%0d",
                         bb_wr_mism, TOK_TOTAL, bb_first_bad);
        end
    end
end

always @(negedge clk) begin
    if (done) begin
        $display("\n---- sglatrack_top (verilog2) done @ cycle %0d ----", cycle_cnt);
        $display("  Tokens: template=%0d search=%0d total=%0d", TEMPL_TOT, SRCH_TOT, TOK_TOTAL);
        $display("  sel_block_i = %0d", sel_block_i);
        $display("\n  Predicted bbox (Q8.8 hex | signed/256):");
        $display("    cx = 0x%04h  (%f)", cx_out, $itor($signed(cx_out)) / 256.0);
        $display("    cy = 0x%04h  (%f)", cy_out, $itor($signed(cy_out)) / 256.0);
        $display("    w  = 0x%04h  (%f)", w_out,  $itor($signed(w_out))  / 256.0);
        $display("    h  = 0x%04h  (%f)", h_out,  $itor($signed(h_out))  / 256.0);

        $display("\n  Golden bbox (Activation/box_head_after_cal_bbox_bbox_bi.txt):");
        $display("    cx = 0x007F  (0.496094)");
        $display("    cy = 0x007E  (0.492188)");
        $display("    w  = 0x0024  (0.140625)");
        $display("    h  = 0x0057  (0.339844)");

        if (($signed(cx_out) - $signed(16'sh007F)) <= 2 &&
            ($signed(16'sh007F) - $signed(cx_out)) <= 2 &&
            ($signed(cy_out) - $signed(16'sh007E)) <= 2 &&
            ($signed(16'sh007E) - $signed(cy_out)) <= 2 &&
            ($signed(w_out)  - $signed(16'sh0024)) <= 2 &&
            ($signed(16'sh0024) - $signed(w_out))  <= 2 &&
            ($signed(h_out)  - $signed(16'sh0057)) <= 2 &&
            ($signed(16'sh0057) - $signed(h_out))  <= 2)
            $display("\n  [PASS] bbox matches golden within +-2 LSB");
        else
            $display("\n  [FAIL] bbox differs from golden (+-2 LSB)");

        $toggle_stop();
        $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
        $finish;
    end
end

initial begin
    $fsdbDumpfile("sglatrack_tb.fsdb");
    // Must scope TEST.u_DUT; bare $fsdbDumpvars only dumps TEST tb regs (no /TEST/u_DUT/* in FSDB)
    $fsdbDumpvars(0, TEST.u_DUT);
    $fsdbDumpoff;
end

initial begin
    #(CYCLE * FSDB_START_MULT);
    $fsdbDumpon;
end

initial begin
    $set_toggle_region("u_DUT");
    $toggle_start();

    $readmemb("./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt", GOLD_BB);
    $readmemb("./TXT_File/Activation/template_post_embed_input_bi.txt", TEMPL_MEM);
    $readmemb("./TXT_File/Activation/search_post_embed_input_bi.txt", SRCH_MEM);
    $display("[TB] Loaded template(%0d) + search(%0d) + backbone_norm golden(%0d)",
             TEMPL_TOT, SRCH_TOT, TOK_TOTAL);
    $display("[TB] Weights: ROM macros in DUT (memory/ at compile); no Weight/*.txt in RTL");

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 32'd0;
    tok_cnt    = 14'd0;
    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    $display("[TB] Started sglatrack_top sel_block_i=%0d", sel_block_i);

    #(CYCLE * 70_000_000);
    $display("[TB] TIMEOUT: verilog2 full chain did not finish within 70M cycles");
    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
