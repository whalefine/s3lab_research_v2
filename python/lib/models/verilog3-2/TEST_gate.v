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
// TEST_gate.v -- verilog3 full-chain GLS final-check (backbone3 + head3)
//
// Flat netlist: sglatrack_top_syn.v + memory2 SRAM/ROM + stdcell.
// Stimulus on negedge clk + INPUT_SETUP (SDF: avoids FF $hold on async controls).
// Token stream uses u_DUT.u_backbone.x_ready (synthesis must preserve u_backbone).
//
// Golden:
//   template_post_embed_input_bi.txt
//   search_post_embed_input_bi.txt
//   box_head_after_cal_bbox_bbox_bi.txt
//
// Zero-delay functional GLS (no SDF):
//   vcs TEST_gate.v +notimingcheck +delay_mode_zero +define+TSMC_CM_NO_WARNING ...
//
// With SDF (compile +define+GATE_SDF; do NOT use +delay_mode_zero):
//   vcs TEST_gate.v +define+GATE_SDF +define+TSMC_CM_NO_WARNING ...
//   ./simv | tee simv_gate.log
//
// Optional: +define+GATE_SDF_FILE=\"./sglatrack_top.sdf\"
// SAIF: sglatrack_top_gate.saif  (read_saif: -instance u_DUT)
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

`define GATE_SDF_FILE "sglatrack_top.sdf"

module TEST_gate;

// Align with synthesis/sglatrack_top.sdc CLK_PERIOD
parameter CYCLE = 3.0;

`ifdef GATE_SDF
parameter INPUT_SETUP = 0.75;
`else
parameter INPUT_SETUP = 0.0;
`endif

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

reg         clk, reset, start;
reg  [3:0]  sel_block_i;
reg  signed [15:0] data_in;
reg                data_valid;

wire [15:0] cx_out, cy_out, w_out, h_out;
wire        busy;
wire        done;
wire        x_ready;

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
    .x_ready     (x_ready),
    .cx_o        (cx_out),
    .cy_o        (cy_out),
    .w_o         (w_out),
    .h_o         (h_out)
);

always #(CYCLE/2.0) clk = ~clk;

reg [31:0] cycle_cnt;
reg [15:0] TEMPL_MEM [0:TEMPL_TOT-1];
reg [15:0] SRCH_MEM  [0:SRCH_TOT-1];
reg [15:0] bbox_gold [0:3];
reg [13:0] tok_cnt;
reg        done_seen;

always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

always @(posedge clk) begin
    if (reset)
        done_seen <= 1'b0;
    else if (done)
        done_seen <= 1'b1;
end

// negedge + INPUT_SETUP: stable before DUT posedge FF under SDF
always @(negedge clk) begin
    #(INPUT_SETUP);
    if (reset) begin
        tok_cnt    = 14'd0;
        data_in    = 16'sd0;
        data_valid = 1'b0;
    end else if ((start || busy) && x_ready) begin
        if (tok_cnt < TEMPL_TOT) begin
            data_valid = 1'b1;
            data_in    = TEMPL_MEM[tok_cnt];
            tok_cnt    = tok_cnt + 14'd1;
        end else if (tok_cnt < TOK_TOTAL) begin
            data_valid = 1'b1;
            data_in    = SRCH_MEM[tok_cnt - TEMPL_TOT];
            tok_cnt    = tok_cnt + 14'd1;
        end else begin
            data_valid = 1'b0;
            data_in    = 16'sd0;
        end
    end else if (start || busy) begin
        data_valid = 1'b0;
        data_in    = 16'sd0;
    end
end

always @(posedge clk) begin
    if (cycle_cnt % 2000000 == 0) begin
        $display("[DBG] cycle=%0d x_ready=%b busy=%b tok_cnt=%0d done=%b", 
                 cycle_cnt, x_ready, busy, tok_cnt, done);
    end
end

initial begin
`ifdef GATE_SDF
    $display("[TB] SDF annotate: %s -> u_DUT", `GATE_SDF_FILE);
    $sdf_annotate(`GATE_SDF_FILE, u_DUT);
    $display("[TB] SDF annotate done (timing checks enabled; TB INPUT_SETUP=%0.2f ns)", INPUT_SETUP);
`endif

    $fsdbDumpfile("sglatrack_top_gate.fsdb");
    $fsdbDumpvars;

    $set_toggle_region("u_DUT");
    $toggle_start();

    $readmemb({`GOLDEN_ACT, "/template_post_embed_input_bi.txt"},        TEMPL_MEM);
    $readmemb({`GOLDEN_ACT, "/search_post_embed_input_bi.txt"},         SRCH_MEM);
    $readmemb({`GOLDEN_ACT, "/box_head_after_cal_bbox_bbox_bi.txt"},     bbox_gold);

    $display("[TB] verilog3 full-chain GLS final-check");
    $display("[TB] template=%0d search=%0d tokens=%0d", TEMPL_TOT, SRCH_TOT, TOK_TOTAL);
    $display("[TB] Golden dir: %s", `GOLDEN_ACT);
    $display("[TB] CYCLE=%0.1f ns  stimulus=negedge+%0.2f ns", CYCLE, INPUT_SETUP);
`ifdef GATE_SDF
    $display("[TB] GATE_SDF enabled  file=%s", `GATE_SDF_FILE);
`else
    $display("[TB] GATE_SDF disabled (zero-delay GLS)");
`endif

    clk         = 1'b0;
    reset       = 1'b1;
    start       = 1'b0;
    data_in     = 16'sd0;
    data_valid  = 1'b0;
    cycle_cnt   = 32'd0;
    tok_cnt     = 14'd0;
    done_seen   = 1'b0;
    sel_block_i = 4'd6;

    #(CYCLE) reset = 1'b1;
`ifdef GATE_SDF
    repeat (20) @(posedge clk);
`endif
    @(negedge clk);
    #(INPUT_SETUP);
    reset = 1'b0;

    @(negedge clk);
    #(INPUT_SETUP);
    start = 1'b1;
    @(negedge clk);
    #(INPUT_SETUP);
    start = 1'b0;

    $display("[TB] Started sglatrack_top sel_block_i=%0d @ cycle %0d", sel_block_i, cycle_cnt);

    wait (done_seen === 1'b1);
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
    $toggle_report("sglatrack_top_gate.saif", 1.0e-9, "u_DUT");
    $finish;
end

initial begin
    #(CYCLE * 12_000_000);
    $display("[TB] TIMEOUT @ cycle %0d busy=%0d done=%0d", cycle_cnt, busy, done);
    $toggle_stop();
    $toggle_report("sglatrack_top_gate.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
