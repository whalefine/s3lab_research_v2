`timescale 1ns/10ps

// =============================================================================
// TEST.v -- verilog3 full-chain E2E (backbone3 + head3 via sglatrack_top)
//
// Inputs: template_post_embed + search_post_embed (streamed during backbone S_LOAD_IN).
// No tok1_preload: backbone writes Sram_tok1; head reads it after phase_sel=1.
//
// Golden:
//   template_post_embed_input_bi.txt
//   search_post_embed_input_bi.txt
//   backbone_after_norm_backbone_out_bi.txt  (checked at norm write beat, Plan B)
//   box_head_after_cal_bbox_bbox_bi.txt      (final bbox)
//
// VCS (cwd = dir containing memory/ and TXT_File/; or adjust paths):
//   vcs python/lib/models/verilog3/TEST.v \
//       +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   ./simv | tee simv.log
//
// SAIF (power): on done or timeout -> sglatrack_top_rtl.saif (scope u_DUT)
//   read_saif: -instance u_DUT
//
// Do NOT pass verilog3/*.v twice if TEST pulls sglatrack_top via `include.
// =============================================================================

`include "sglatrack_top.v"

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

module TEST;

parameter CYCLE = 2.0;
parameter [31:0] FSDB_START_MULT     = 32'd28_000_000;
parameter [31:0] PROGRESS_STEP_MULT  = 32'd10_000_000;
parameter BBOX_TOL_LSB = 2;

parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;
parameter FEAT_SZ     = FEAT_H * FEAT_W;
parameter TEMPL_TOT   = LENS_Z  * EMBED_DIM;
parameter SRCH_TOT    = FEAT_SZ * EMBED_DIM;
parameter TOK_TOTAL   = TEMPL_TOT + SRCH_TOT;
parameter N_TOKENS    = TOK_TOTAL / EMBED_DIM;

localparam BB_S_LOAD_IN       = 4'd1;
localparam BB_S_BACKBONE_NORM = 4'd4;

reg         clk, reset, start;
reg  [3:0]  sel_block_i;
reg  signed [15:0] data_in;
reg                data_valid;

wire [15:0] cx_out, cy_out, w_out, h_out;
wire        busy;
wire        done;

sglatrack_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS),
    .FEAT_H    (FEAT_H),
    .FEAT_W    (FEAT_W)
) u_DUT (
    .clk         (clk),
    .reset       (reset),
    .start       (start),
    .sel_block_i (sel_block_i),
    .data_in     (data_in),
    .data_valid  (data_valid),
    .busy        (busy),
    .done        (done),
    .cx_o        (cx_out),
    .cy_o        (cy_out),
    .w_o         (w_out),
    .h_o         (h_out)
);

// Stream merged tokens only during backbone S_LOAD_IN (no tok1_preload).
wire tb_stream_gate = (u_DUT.u_backbone.state == BB_S_LOAD_IN);

wire        bb_done     = u_DUT.u_backbone.done;
wire        bb_norm_wr   = (u_DUT.u_backbone.state == BB_S_BACKBONE_NORM) &&
                           u_DUT.u_backbone.bn_wr_en;
wire [13:0] bb_wr_flat   = u_DUT.u_backbone.bn_wr_addr;
wire [15:0] bb_wr_din    = u_DUT.u_backbone.bn_wr_din;

always #(CYCLE/2.0) clk = ~clk;

reg [31:0] cycle_cnt;
reg [31:0] prog_time_mult;

always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

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
    end else if (tb_stream_gate && u_DUT.u_backbone.x_ready) begin
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
    end else begin
        data_valid <= 1'b0;
        data_in    <= 16'sd0;
    end
end

reg [15:0] GOLD_BB   [0:TOK_TOTAL-1];
reg [15:0] bbox_gold [0:3];
reg [31:0] bb_wr_cnt;
reg [31:0] bb_wr_mism;
reg [31:0] bb_first_bad;
reg        bb_rpt_done;
reg        bb_done_d1;

wire bb_done_rise = bb_done & ~bb_done_d1;

always @(posedge clk) begin
    if (reset) begin
        bb_wr_cnt    <= 32'd0;
        bb_wr_mism   <= 32'd0;
        bb_first_bad <= 32'hFFFF_FFFF;
        bb_rpt_done  <= 1'b0;
        bb_done_d1   <= 1'b0;
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
            $display("\n---- backbone_after_norm @ cycle %0d (write-beat check) ----",
                     cycle_cnt);
            $display("  sel_block_i=%0d  norm_wr_cnt=%0d expect=%0d",
                     sel_block_i, bb_wr_cnt, TOK_TOTAL);
            if (bb_wr_cnt != TOK_TOTAL)
                $display("  [FAIL] norm write count mismatch");
            else if (bb_wr_mism == 32'd0)
                $display("  [PASS] backbone_after_norm write data matches golden (%0d elems)",
                         TOK_TOTAL);
            else
                $display("  [FAIL] backbone_after_norm mismatches=%0d first_bad=%0d",
                         bb_wr_mism, bb_first_bad);
        end
    end
end

initial begin
    $fsdbDumpfile("sglatrack_top.fsdb");
    $fsdbDumpvars(0, TEST.u_DUT);

    $set_toggle_region("u_DUT");
    $toggle_start();

    $readmemb({`GOLDEN_ACT, "/backbone_after_norm_backbone_out_bi.txt"}, GOLD_BB);
    $readmemb({`GOLDEN_ACT, "/template_post_embed_input_bi.txt"},        TEMPL_MEM);
    $readmemb({`GOLDEN_ACT, "/search_post_embed_input_bi.txt"},         SRCH_MEM);
    $readmemb({`GOLDEN_ACT, "/box_head_after_cal_bbox_bbox_bi.txt"},     bbox_gold);

    $display("[TB] verilog3 full-chain E2E");
    $display("[TB] template=%0d search=%0d tokens=%0d", TEMPL_TOT, SRCH_TOT, TOK_TOTAL);
    $display("[TB] Golden dir: %s", `GOLDEN_ACT);

    clk         = 1'b0;
    reset       = 1'b1;
    start       = 1'b0;
    data_in     = 16'sd0;
    data_valid  = 1'b0;
    cycle_cnt   = 32'd0;
    tok_cnt     = 14'd0;
    sel_block_i = 4'd6;

    #(CYCLE) reset = 1'b1;
    #(CYCLE) reset = 1'b0;

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    $display("[TB] Started sglatrack_top sel_block_i=%0d", sel_block_i);

    wait (done === 1'b1);
    @(posedge clk);

    $display("\n---- verilog3 full-chain done @ cycle %0d ----", cycle_cnt);
    $display("  Predicted bbox (Q8.8 hex | float/256):");
    $display("    cx = 0x%04h  (%f)", cx_out, $itor($signed(cx_out)) / 256.0);
    $display("    cy = 0x%04h  (%f)", cy_out, $itor($signed(cy_out)) / 256.0);
    $display("    w  = 0x%04h  (%f)", w_out,  $itor($signed(w_out))  / 256.0);
    $display("    h  = 0x%04h  (%f)", h_out,  $itor($signed(h_out))  / 256.0);
    $display("  Golden bbox:");
    $display("    cx = 0x%04h  (%f)", bbox_gold[0], $itor($signed(bbox_gold[0])) / 256.0);
    $display("    cy = 0x%04h  (%f)", bbox_gold[1], $itor($signed(bbox_gold[1])) / 256.0);
    $display("    w  = 0x%04h  (%f)", bbox_gold[2], $itor($signed(bbox_gold[2])) / 256.0);
    $display("    h  = 0x%04h  (%f)", bbox_gold[3], $itor($signed(bbox_gold[3])) / 256.0);

    if (($signed(cx_out) - $signed(bbox_gold[0])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[0]) - $signed(cx_out)) <= BBOX_TOL_LSB &&
        ($signed(cy_out) - $signed(bbox_gold[1])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[1]) - $signed(cy_out)) <= BBOX_TOL_LSB &&
        ($signed(w_out)  - $signed(bbox_gold[2])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[2]) - $signed(w_out))  <= BBOX_TOL_LSB &&
        ($signed(h_out)  - $signed(bbox_gold[3])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[3]) - $signed(h_out))  <= BBOX_TOL_LSB)
        $display("\n  [PASS] bbox matches golden within +-%0d LSB", BBOX_TOL_LSB);
    else
        $display("\n  [FAIL] bbox differs from golden (+- %0d LSB)", BBOX_TOL_LSB);

    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

initial begin
    #(CYCLE * 120_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d", cycle_cnt, busy);
    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
