`timescale 1ns/10ps

// =============================================================================
// TEST.v -- verilog2 full-chain testbench (sglatrack_top)
//
// DUT: backbone_top (verilog_backbone2) + head_top (verilog_head2 conv/tail/cal_bbox)
//
// Inputs (golden activation only, not weights):
//   template_post_embed_input_bi.txt
//   search_post_embed_input_bi.txt
//
// Weights: internal ROM in backbone_top / conv / tail (memory/ at VCS compile)
//
// Golden bbox (Q8.8, box_head_after_cal_bbox_bbox_bi.txt):
//   cx=0x007F cy=0x007E w=0x0023 h=0x0054
// =============================================================================

module TEST;

parameter CYCLE = 2.0;
parameter [31:0] FSDB_START_MULT     = 32'd100_000_000;
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

wire tb_stream_gate =
    !u_DUT.u_backbone.tok_replay &&
    (u_DUT.u_backbone.u_tb.state == 4'd1);

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
        $display("    cx = 0x007F  cy = 0x007E  w = 0x0023  h = 0x0054");

        if (($signed(cx_out) - $signed(16'sh007F)) <= 2 &&
            ($signed(16'sh007F) - $signed(cx_out)) <= 2 &&
            ($signed(cy_out) - $signed(16'sh007E)) <= 2 &&
            ($signed(16'sh007E) - $signed(cy_out)) <= 2 &&
            ($signed(w_out)  - $signed(16'sh0023)) <= 2 &&
            ($signed(16'sh0023) - $signed(w_out))  <= 2 &&
            ($signed(h_out)  - $signed(16'sh0054)) <= 2 &&
            ($signed(16'sh0054) - $signed(h_out))  <= 2)
            $display("\n  [PASS] bbox matches golden within +-2 LSB");
        else
            $display("\n  [FAIL] bbox differs from golden (+-2 LSB)");

        $toggle_stop();
        $toggle_report("sglatrack_rtl2.saif", 1.0e-9, "u_DUT");
        $finish;
    end
end

initial begin
    $fsdbDumpfile("sglatrack_tb2.fsdb");
    $fsdbDumpvars(1, TEST.u_DUT);
    $fsdbDumpvars(0, TEST.u_DUT.u_head);
    $fsdbDumpoff;
end

initial begin
    #(CYCLE * FSDB_START_MULT);
    $fsdbDumpon;
end

initial begin
    $set_toggle_region("u_DUT");
    $toggle_start();

    $readmemb("./TXT_File/Activation/template_post_embed_input_bi.txt", TEMPL_MEM);
    $readmemb("./TXT_File/Activation/search_post_embed_input_bi.txt", SRCH_MEM);
    $display("[TB] Loaded template(%0d) + search(%0d) activation inputs", TEMPL_TOT, SRCH_TOT);
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

    #(CYCLE * 150_000_000);
    $display("[TB] TIMEOUT: verilog2 full chain did not finish within 150M cycles");
    $toggle_stop();
    $toggle_report("sglatrack_rtl2.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
