// =============================================================================
// TEST_head_gate.v -- sglatrack_top(head-only) GLS final-check
//
// Preloads backbone_after_norm into Sram_tok1 (token-major), then starts head_top.
// reset/start: negedge+INPUT_SETUP (FF async CN/SN hold under SDF).
// tok1_preload: posedge NBA (Sram_tok1 CLK=~clk; must NOT toggle A/D at sys negedge).
//
// Golden: ./TXT_File/Activation/
//
// Zero-delay functional GLS (no SDF):
//   vcs TEST_head_gate.v +notimingcheck +delay_mode_zero ...
//
// With SDF (compile +define+GATE_SDF; do NOT use +delay_mode_zero):
//   vcs TEST_head_gate.v +define+GATE_SDF +define+TSMC_CM_NO_WARNING ...
//   ./simv | tee simv_gate.log
// TB: reset/start on negedge+INPUT_SETUP; SRAM preload on posedge (see preload task).
//
// Optional: +define+GATE_SDF_FILE=\"./sglatrack_top.sdf\"
// =============================================================================

`timescale 1ns/1ps
`include "sglatrack_top_syn.v"
`include "./memory2/Sram_q.v"
`include "./memory2/Sram_k.v"
`include "./memory2/Sram_qkm.v"
`include "./memory2/Sram_tok1.v"
`include "./memory2/Sram_tok2.v"
`include "./memory2/Sram_v.v"
`include "./memory2/Sram_x.v"
`include "./memory2/rom_backbone_blocks_0_3_attn_qkv_weight.v"
`include "./memory2/rom_backbone_blocks_0_3_mlp_fc1_weight.v"
`include "./memory2/rom_backbone_blocks_0_3_mlp_fc2_weight.v"
`include "./memory2/rom_backbone_blocks_0_6_attn_proj_bias.v"
`include "./memory2/rom_backbone_blocks_0_6_attn_proj_weight.v"
`include "./memory2/rom_backbone_blocks_0_6_attn_qkv_bias.v"
`include "./memory2/rom_backbone_blocks_0_6_mlp_fc1_bias.v"
`include "./memory2/rom_backbone_blocks_0_6_mlp_fc2_bias.v"
`include "./memory2/rom_backbone_blocks_0_6_norm1_bias.v"
`include "./memory2/rom_backbone_blocks_0_6_norm1_weight.v"
`include "./memory2/rom_backbone_blocks_0_6_norm2_bias.v"
`include "./memory2/rom_backbone_blocks_0_6_norm2_weight.v"
`include "./memory2/rom_backbone_blocks_4_6_attn_qkv_weight.v"
`include "./memory2/rom_backbone_blocks_4_6_mlp_fc1_weight.v"
`include "./memory2/rom_backbone_blocks_4_6_mlp_fc2_weight.v"
`include "./memory2/rom_backbone_norm_bias.v"
`include "./memory2/rom_backbone_norm_weight.v"
`include "./memory2/rom_box_head_shared_conv1_2_folded_bias.v"
`include "./memory2/rom_box_head_shared_conv1_folded_weight1.v"
`include "./memory2/rom_box_head_shared_conv1_folded_weight2.v"
`include "./memory2/rom_box_head_shared_conv2_folded_weight1.v"
`include "./memory2/rom_box_head_shared_conv2_folded_weight2.v"
`include "./memory2/rom_box_head_shared_conv2_folded_weight3.v"
`include "./memory2/rom_box_head_tail_ctr_offset_size_bias.v"
`include "./memory2/rom_box_head_tail_ctr_offset_size_weight.v"
`include "tcbn16ffcllbwp7d5t20p96cpd.v"

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

`ifndef GATE_SDF_FILE
`define GATE_SDF_FILE "sglatrack_top.sdf"
`endif

module TEST_head_gate;

parameter DATA_W       = 16;
parameter IN_CH        = 32;
parameter N_TOKENS     = 320;
parameter FEAT_H       = 16;
parameter FEAT_W       = 16;
parameter TOT_VALS     = N_TOKENS * IN_CH;
parameter BBOX_LEN     = 4;
parameter BBOX_TOL_LSB = 2;

// must write X.0, can't write x; align with synthesis CLK_PERIOD / SDF
parameter CYCLE = 5.0;

// After negedge, wait before changing inputs (setup for DUT posedge FF under SDF)
`ifdef GATE_SDF
parameter INPUT_SETUP = 1.25;
`else
parameter INPUT_SETUP = 0.0;
`endif

reg               clk;
reg               reset;
reg               start;
reg               tok1_preload;
reg [13:0]        tok1_preload_addr;
reg [15:0]        tok1_preload_din;

wire              head_busy;
wire              head_done;
wire [15:0]       cx_o;
wire [15:0]       cy_o;
wire [15:0]       w_o;
wire [15:0]       h_o;

reg [DATA_W-1:0]  raw_in    [0:TOT_VALS-1];
reg [DATA_W-1:0]  bbox_gold [0:BBOX_LEN-1];

reg [31:0]        cycle_cnt;
reg               done_seen;

sglatrack_top #(
    .DATA_W   (DATA_W  ),
    .IN_CH    (IN_CH   ),
    .FEAT_H   (FEAT_H  ),
    .FEAT_W   (FEAT_W  ),
    .N_TOKENS (N_TOKENS)
) u_DUT (
    .clk               (clk               ),
    .reset             (reset             ),
    .start             (start             ),
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

always #(CYCLE/2.0) clk = ~clk;

always @(posedge clk) begin
    if (reset)
        cycle_cnt <= 32'd0;
    else
        cycle_cnt <= cycle_cnt + 32'd1;
end

always @(posedge clk) begin
    if (reset)
        done_seen <= 1'b0;
    else if (head_done)
        done_seen <= 1'b1;
end

// Preload Sram_tok1: posedge NBA, one write per cycle (same as TEST_head.v RTL).
// Sram_tok1 CLK=~clk samples at sys negedge; changing A/D after sys negedge+INPUT_SETUP
// violates macro $hold (posedge CLK vs negedge D/A). NBA after sys posedge is safe.
task preload_sram_tok1;
    integer idx;
    begin
        tok1_preload      <= 1'b0;
        tok1_preload_addr <= 14'd0;
        tok1_preload_din  <= 16'd0;
        @(posedge clk);
        for (idx = 0; idx < TOT_VALS; idx = idx + 1) begin
            tok1_preload      <= 1'b1;
            tok1_preload_addr <= idx[13:0];
            tok1_preload_din  <= raw_in[idx];
            @(posedge clk);
        end
        tok1_preload      <= 1'b0;
        tok1_preload_addr <= 14'd0;
        tok1_preload_din  <= 16'd0;
        @(posedge clk);
        $display("[TB] Sram_tok1 preload done (%0d words, token-major)", TOT_VALS);
    end
endtask

initial begin
`ifdef GATE_SDF
    $display("[TB] SDF annotate: %s -> u_DUT", `GATE_SDF_FILE);
    $sdf_annotate(`GATE_SDF_FILE, u_DUT);
    $display("[TB] SDF annotate done (timing checks enabled; TB INPUT_SETUP=%0.2f ns)", INPUT_SETUP);
`endif

    $fsdbDumpfile("sglatrack_top_gate.fsdb");
    $fsdbDumpvars;

    $readmemb({`GOLDEN_ACT, "/backbone_after_norm_backbone_out_bi.txt"}, raw_in);
    $readmemb({`GOLDEN_ACT, "/box_head_after_cal_bbox_bbox_bi.txt"     }, bbox_gold);

    $display("[TB] head-only final-check (GLS)  TOT_VALS=%0d", TOT_VALS);
    $display("[TB] Golden dir: %s", `GOLDEN_ACT);
    $display("[TB] CYCLE=%0.1f ns  reset/start=negedge+%0.2f ns  preload=posedge NBA",
             CYCLE, INPUT_SETUP);
`ifdef GATE_SDF
    $display("[TB] GATE_SDF enabled  file=%s", `GATE_SDF_FILE);
`else
    $display("[TB] GATE_SDF disabled");
`endif

    clk                = 1'b0;
    reset              = 1'b1;
    start              = 1'b0;
    tok1_preload       = 1'b0;
    tok1_preload_addr  = 14'd0;
    tok1_preload_din   = 16'd0;
    cycle_cnt          = 32'd0;
    done_seen          = 1'b0;

    // Clock + reset release (async reset recovery under SDF)
    #(CYCLE) reset = 1'b1;
`ifdef GATE_SDF
    repeat (20) @(posedge clk);
`endif
    @(negedge clk);
    #(INPUT_SETUP);
    reset = 1'b0;

    preload_sram_tok1();

    @(negedge clk);
    #(INPUT_SETUP);
    start = 1'b1;
    @(negedge clk);
    #(INPUT_SETUP);
    start = 1'b0;

    $display("[TB] start pulsed @ cycle %0d", cycle_cnt);

    wait (done_seen === 1'b1);
    @(posedge clk);

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

    $finish;
end

initial begin
    // $sdf_annotate("sglatrack_top.sdf", u_DUT);
    #(CYCLE * 500_000_000);
    $display("[TB] TIMEOUT: head_top did not finish (cycle %0d head_busy=%0d)",
             cycle_cnt, head_busy);
    $finish;
end

endmodule
