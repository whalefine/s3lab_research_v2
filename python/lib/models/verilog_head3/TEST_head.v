// =============================================================================
// TEST_head.v -- verilog_head2 head-only test via sglatrack_top (Plan B)
//
// Preloads backbone_after_norm into Sram_tok1 (token-major), then starts head_top.
// head_top Plan A: conv1 reads Sram_tok1 directly (token-major, no S_FILL).
// Compares final bbox vs box_head_after_cal_bbox_bbox_bi.txt only.
//
// VCS (DUT pulls RTL + memory via `include in sglatrack_top.v; TB only extra file):
//   vcs verilog_head2/sglatrack_top.v verilog_head2/TEST_head.v \
//     +lint=TFIPC-L +define+TSMC_CM_NO_WARNING | tee runvcs.log
//   Compile root must contain memory/ (Sram_* + rom_box_head_*). Or +incdir if paths differ.
//   ./simv | tee simv.log
//
//   grep -E '\\[PASS\\]|\\[FAIL\\]|TIMEOUT|Head-only done' simv.log
// =============================================================================

`timescale 1ns/1ps
`include "sglatrack_top.v"

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

module TEST_head ;

parameter DATA_W       = 16 ;
parameter IN_CH        = 32 ;
parameter N_TOKENS     = 320 ;
parameter FEAT_H       = 16 ;
parameter FEAT_W       = 16 ;
parameter TOT_VALS     = N_TOKENS * IN_CH ;
parameter BBOX_LEN     = 4 ;
parameter BBOX_TOL_LSB = 0 ;  // bit-exact h (u_siz ROUND_Y + tail h-ch sigmoid +1)

// must write X.0, can't write x
parameter CYCLE = 2.0 ;

reg clk ;
reg rst_n ;

initial begin
    clk = 1'b0 ;
end
always #(CYCLE/2.0) clk = ~clk ;

reg [DATA_W-1:0] raw_in    [0:TOT_VALS-1] ;
reg [DATA_W-1:0] bbox_gold [0:BBOX_LEN-1] ;

reg               kick_start ;
reg               head_start ;
reg               tok1_preload ;
reg [13:0]        tok1_preload_addr ;
reg [15:0]        tok1_preload_din ;

wire              head_busy ;
wire              head_done ;
wire [15:0]       cx_o, cy_o, w_o, h_o ;

sglatrack_top #(
    .DATA_W   (DATA_W  ),
    .IN_CH    (IN_CH   ),
    .FEAT_H   (FEAT_H  ),
    .FEAT_W   (FEAT_W  ),
    .N_TOKENS (N_TOKENS)
) u_dut (
    .clk               (clk               ),
    .reset             (~rst_n            ),
    .start             (head_start        ),
    .busy              (head_busy         ),
    .done              (head_done         ),
    .tok1_preload      (tok1_preload      ),
    .tok1_preload_addr (tok1_preload_addr ),
    .tok1_preload_din  (tok1_preload_din  ),
    .cx_o              (cx_o              ),
    .cy_o              (cy_o              ),
    .w_o               (w_o               ),
    .h_o               (h_o               )
);

reg [31:0] cycle_cnt ;

always @(posedge clk) begin
    if (!rst_n)
        cycle_cnt <= 32'd0 ;
    else
        cycle_cnt <= cycle_cnt + 1 ;
end

// One-cycle start pulse after Sram_tok1 preload completes
always @(posedge clk) begin
    if (!rst_n) begin
        head_start        <= 1'b0 ;
        tok1_preload      <= 1'b0 ;
        tok1_preload_addr <= 14'd0 ;
        tok1_preload_din  <= 16'd0 ;
    end else if (kick_start)
        head_start <= 1'b1 ;
    else
        head_start <= 1'b0 ;
end

// Preload Sram_tok1: one write per posedge (SRAM CLK=~clk, write sampled at negedge)
task preload_sram_tok1 ;
    integer idx ;
    begin
        tok1_preload <= 1'b0 ;
        tok1_preload_addr <= 14'd0 ;
        tok1_preload_din  <= 16'd0 ;
        @(posedge clk) ;
        for (idx = 0; idx < TOT_VALS; idx = idx + 1) begin
            tok1_preload      <= 1'b1 ;
            tok1_preload_addr <= idx[13:0] ;
            tok1_preload_din  <= raw_in[idx] ;
            @(posedge clk) ;
        end
        tok1_preload      <= 1'b0 ;
        tok1_preload_addr <= 14'd0 ;
        tok1_preload_din  <= 16'd0 ;
        @(posedge clk) ;
        $display("[TB] Sram_tok1 preload done (%0d words, token-major)", TOT_VALS);
    end
endtask

initial begin
    $fsdbDumpfile("sglatrack_top.fsdb");
    $fsdbDumpvars;
    $fsdbDumpMDA;

    $readmemb({`GOLDEN_ACT, "/backbone_after_norm_backbone_out_bi.txt"}, raw_in    ) ;
    $readmemb({`GOLDEN_ACT, "/box_head_after_cal_bbox_bbox_bi.txt"     }, bbox_gold) ;

    $set_toggle_region("u_dut");
    $toggle_start();

    rst_n      = 1'b0 ;
    kick_start = 1'b0 ;
    #25 ;
    @(posedge clk) ;
    rst_n = 1'b1 ;
    @(posedge clk) ;
    @(posedge clk) ;

    preload_sram_tok1 ;

    kick_start = 1'b1 ;
    @(posedge clk) ;
    kick_start = 1'b0 ;

    @(posedge head_done) ;

    $display("\n---- Head-only done @ cycle %0d ----", cycle_cnt);
    $display("\n  Predicted bbox (Q8.8 hex | float/256):");
    $display("    cx = 0x%04h  (%f)", cx_o, $itor($signed(cx_o)) / 256.0);
    $display("    cy = 0x%04h  (%f)", cy_o, $itor($signed(cy_o)) / 256.0);
    $display("    w  = 0x%04h  (%f)", w_o,  $itor($signed(w_o))  / 256.0);
    $display("    h  = 0x%04h  (%f)", h_o,  $itor($signed(h_o))  / 256.0);
    $display("\n  Golden bbox  (box_head_after_cal_bbox_bbox_bi.txt):");
    $display("    cx = 0x%04h  (%f)", bbox_gold[0], $itor($signed(bbox_gold[0])) / 256.0);
    $display("    cy = 0x%04h  (%f)", bbox_gold[1], $itor($signed(bbox_gold[1])) / 256.0);
    $display("    w  = 0x%04h  (%f)", bbox_gold[2], $itor($signed(bbox_gold[2])) / 256.0);
    $display("    h  = 0x%04h  (%f)", bbox_gold[3], $itor($signed(bbox_gold[3])) / 256.0);
    if (($signed(cx_o) - $signed(bbox_gold[0])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[0]) - $signed(cx_o)) <= BBOX_TOL_LSB &&
        ($signed(cy_o) - $signed(bbox_gold[1])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[1]) - $signed(cy_o)) <= BBOX_TOL_LSB &&
        ($signed(w_o)  - $signed(bbox_gold[2])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[2]) - $signed(w_o))  <= BBOX_TOL_LSB &&
        ($signed(h_o)  - $signed(bbox_gold[3])) <= BBOX_TOL_LSB &&
        ($signed(bbox_gold[3]) - $signed(h_o))  <= BBOX_TOL_LSB)
        $display("\n  [PASS] bbox matches golden within +-%0d LSB", BBOX_TOL_LSB);
    else
        $display("\n  [FAIL] bbox differs from golden (+- %0d LSB)", BBOX_TOL_LSB);

    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_dut");
    $finish ;
end

initial begin
    #(CYCLE * 500_000_000);
    $display("[TB] TIMEOUT: head_top did not finish (cycle %0d head_busy=%0d)",
             cycle_cnt, head_busy) ;
    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_dut");
    $finish ;
end

endmodule
