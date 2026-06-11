`timescale 1ns/10ps
`include "sglatrack_top.v"
// =============================================================================
// TEST_backbone.v -- sglatrack_top(backbone-only) E2E with per-block stage checks
//
// Shadow SRAM mirrors macro tok1/tok2/q (same model as TEST_transformer_block).
// Compare at transformer_block state edges without advancing extra cycles
// (avoids readback-during-run corruption / xxxx on macro Q).
//
// Stages per block: NORM1(tok1) ATTN(tok1) RES1(tok2) NORM2(q) MLP(q) BLOCK(tok1)
// Backbone: LOAD_IN(merged,tok1) -> blocks 0..6 -> BACKBONE_NORM(tok1)
//
// Golden: ./TXT_File/Activation/
// =============================================================================

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

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
parameter TOK_FLAT    = TOK_TOTAL;

reg         clk, reset, start;
reg  [3:0]  sel_block_i;
reg  signed [15:0] data_in;
reg                data_valid;

reg [3:0] bb_state_d;
reg [3:0] tb_state_d;
reg       tb_done_d;

wire        busy;
wire        x_ready;
wire        done;
wire signed [15:0] y_o;
wire        y_valid;

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
    .tok1_readback     (1'b0),
    .tok1_readback_addr(14'd0),
    .tok1_readback_q   ()
);

// Hierarchy shortcuts
wire [3:0] bb_state     = u_DUT.u_backbone.state;
wire [3:0] bb_block_idx = u_DUT.u_backbone.block_idx;
wire [3:0] tb_state     = u_DUT.u_backbone.u_tb.state;
wire       tb_done      = u_DUT.u_backbone.u_tb.done;

// Macro tap (shadow update)
wire        t1_ceb = u_DUT.sram_tok1_ceb_mac;
wire        t1_web = u_DUT.sram_tok1_web_mac;
wire [13:0] t1_addr = u_DUT.sram_tok1_addr_mac;
wire [15:0] t1_din  = u_DUT.sram_tok1_din_mac;

wire        t2_ceb = u_DUT.sram_tok2_ceb_mac;
wire        t2_web = u_DUT.sram_tok2_web_mac;
wire [13:0] t2_addr = u_DUT.sram_tok2_addr_mac;
wire [15:0] t2_din  = u_DUT.sram_tok2_din_mac;

wire        q_ceb = u_DUT.sram_q_ceb_mac;
wire        q_web = u_DUT.sram_q_web_mac;
wire [13:0] q_addr = u_DUT.sram_q_addr_mac;
wire [15:0] q_din  = u_DUT.sram_q_din_mac;

// backbone_top FSM
localparam BB_S_IDLE          = 4'd0;
localparam BB_S_LOAD_IN       = 4'd1;
localparam BB_S_RUN_FIXED     = 4'd2;
localparam BB_S_RUN_SELECTED  = 4'd3;
localparam BB_S_BACKBONE_NORM = 4'd4;
localparam BB_S_DONE          = 4'd5;

// transformer_block FSM (must match transformer_block.v)
localparam TB_S_NORM1     = 4'd2;
localparam TB_S_ATTN_FEED= 4'd3;
localparam TB_S_ATTN_WAIT= 4'd4;
localparam TB_S_RES1      = 4'd5;
localparam TB_S_NORM2     = 4'd6;
localparam TB_S_MLP_FEED  = 4'd7;
localparam TB_S_MLP_WAIT  = 4'd8;
localparam TB_S_RES2      = 4'd9;
localparam TB_S_DONE      = 4'd10;

// Shadow SRAM (behavioral mirror of CLK=~clk macros)
reg [15:0] SHADOW_TOK1 [0:TOK_FLAT-1];
reg [15:0] SHADOW_TOK2 [0:TOK_FLAT-1];
reg [15:0] SHADOW_Q    [0:TOK_FLAT-1];

reg [13:0] t1_raddr_q, t2_raddr_q, q_raddr_q;

reg [15:0] GOLD_STAGE [0:TOK_FLAT-1];
reg [15:0] GOLD_MERGED [0:TOK_FLAT-1];
reg [15:0] GOLD_BB     [0:TOK_FLAT-1];

reg [31:0] cycle_cnt;
reg [15:0] TEMPL_MEM [0:TEMPL_TOT-1];
reg [15:0] SRCH_MEM  [0:SRCH_TOT-1];
reg [13:0] tok_cnt;
reg        done_seen;

// SRAM read addr latch (negedge, CLK=~clk macro)
always @(negedge clk) begin
    if (!t1_ceb && t1_web)
        t1_raddr_q <= t1_addr;
    if (!t2_ceb && t2_web)
        t2_raddr_q <= t2_addr;
    if (!q_ceb && q_web)
        q_raddr_q <= q_addr;
end

always @(posedge clk) begin
    if (!t1_ceb && !t1_web)
        SHADOW_TOK1[t1_addr] <= t1_din;
    if (!t2_ceb && !t2_web)
        SHADOW_TOK2[t2_addr] <= t2_din;
    if (!q_ceb && !q_web)
        SHADOW_Q[q_addr] <= q_din;
end

always #(CYCLE/2.0) clk = ~clk;

// Cycle counter
always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

// Stimulus: stream template then search tokens into DUT
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

// ---------------------------------------------------------------------------
// Golden load: stg 0=norm1 1=attn 2=res1 3=norm2 4=mlp 5=block_out
// ---------------------------------------------------------------------------
task load_block_golden;
    input [3:0] blk;
    input [2:0] stg;
    reg [1023:0] path;
    begin
        case (stg)
            3'd0: $sformat(path, "%s/backbone_blocks_%0d_after_norm1_out_bi.txt",
                        `GOLDEN_ACT, blk);
            3'd1: $sformat(path, "%s/backbone_blocks_%0d_after_attn_attn_out_bi.txt",
                        `GOLDEN_ACT, blk);
            3'd2: $sformat(path, "%s/backbone_blocks_%0d_after_residual_add1_out_bi.txt",
                        `GOLDEN_ACT, blk);
            3'd3: $sformat(path, "%s/backbone_blocks_%0d_after_norm2_out_bi.txt",
                        `GOLDEN_ACT, blk);
            3'd4: $sformat(path, "%s/backbone_blocks_%0d_mlp_after_mlp_out_bi.txt",
                        `GOLDEN_ACT, blk);
            3'd5: $sformat(path, "%s/backbone_blocks_%0d_after_block_out_bi.txt",
                        `GOLDEN_ACT, blk);
            default: path = "";
        endcase
        $readmemb(path, GOLD_STAGE);
    end
endtask

// mem_sel: 0=tok1 1=tok2 2=q
task compare_shadow;
    input [1:0]  mem_sel;
    input [3:0]  blk;
    input [2:0]  stg;
    input [255:0] stg_name;
    input [31:0]  expect_cnt;
    integer       idx;
    reg [15:0]    rtl_val;
    reg [15:0]    gold_val;
    reg [31:0]    mism;
    reg [31:0]    first_bad;
    reg [15:0]    rtl_bad;
    reg [15:0]    gold_bad;
    begin
        load_block_golden(blk, stg);
        mism      = 32'd0;
        first_bad = 32'hFFFF_FFFF;
        rtl_bad   = 16'h0000;
        gold_bad  = 16'h0000;
        for (idx = 0; idx < expect_cnt; idx = idx + 1) begin
            case (mem_sel)
                2'd0: rtl_val = SHADOW_TOK1[idx];
                2'd1: rtl_val = SHADOW_TOK2[idx];
                2'd2: rtl_val = SHADOW_Q[idx];
                default: rtl_val = 16'hXXXX;
            endcase
            gold_val = GOLD_STAGE[idx];
            if (gold_val !== rtl_val) begin
                mism = mism + 32'd1;
                if (first_bad == 32'hFFFF_FFFF) begin
                    first_bad = idx;
                    rtl_bad   = rtl_val;
                    gold_bad  = gold_val;
                end
            end
        end
        $display("[STAGE] BLOCK%0d %0s @ cycle %0d  (shadow %0d words)",
                blk, stg_name, cycle_cnt, expect_cnt);
        if (mism == 0)
            $display("  [PASS] BLOCK%0d %0s", blk, stg_name);
        else
            $display("  [FAIL] BLOCK%0d %0s  mismatches = %0d / %0d  first_bad_idx = %0d  RTL = %04h  GOLD = %04h",
                    blk, stg_name, mism, expect_cnt, first_bad, rtl_bad, gold_bad);
    end
endtask

task compare_shadow_merged;
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
            if (GOLD_MERGED[idx] !== SHADOW_TOK1[idx]) begin
                mism = mism + 32'd1;
                if (first_bad == 32'hFFFF_FFFF) begin
                    first_bad = idx;
                    rtl_bad   = SHADOW_TOK1[idx];
                    gold_bad  = GOLD_MERGED[idx];
                end
            end
        end
        $display("[STAGE] %0s @ cycle %0d  (shadow tok1 %0d words)", tag, cycle_cnt, expect_cnt);
        if (mism == 0)
            $display("  [PASS] %0s", tag);
        else
            $display("  [FAIL] %0s  mismatches = %0d / %0d  first_bad_idx = %0d  RTL = %04h  GOLD = %04h",
                    tag, mism, expect_cnt, first_bad, rtl_bad, gold_bad);
    end
endtask

task compare_shadow_bb_norm;
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
            if (GOLD_BB[idx] !== SHADOW_TOK1[idx]) begin
                mism = mism + 32'd1;
                if (first_bad == 32'hFFFF_FFFF) begin
                    first_bad = idx;
                    rtl_bad   = SHADOW_TOK1[idx];
                    gold_bad  = GOLD_BB[idx];
                end
            end
        end
        $display("[STAGE] %0s @ cycle %0d  (shadow tok1 %0d words)", tag, cycle_cnt, expect_cnt);
        if (mism == 0)
            $display("  [PASS] %0s", tag);
        else
            $display("  [FAIL] %0s  mismatches = %0d / %0d  first_bad_idx = %0d  RTL = %04h  GOLD = %04h",
                    tag, mism, expect_cnt, first_bad, rtl_bad, gold_bad);
    end
endtask

// ---------------------------------------------------------------------------
// Edge detectors
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    bb_state_d <= bb_state;
    tb_state_d <= tb_state;
    tb_done_d  <= tb_done;
end

// LOAD_IN complete
always @(posedge clk) begin
    if (!reset &&
        (bb_state_d == BB_S_LOAD_IN) && (bb_state == BB_S_RUN_FIXED))
        compare_shadow_merged("LOAD_IN merged_tokens", TOK_FLAT);
end

// transformer_block internal stages (only while backbone runs a block)
always @(posedge clk) begin
    if (reset)
        ;
    else if ((bb_state == BB_S_RUN_FIXED) || (bb_state == BB_S_RUN_SELECTED)) begin
        if ((tb_state_d == TB_S_NORM1) && (tb_state == TB_S_ATTN_FEED))
            compare_shadow(2'd0, bb_block_idx, 3'd0, "NORM1", TOK_FLAT);
        else if ((tb_state_d == TB_S_ATTN_WAIT) && (tb_state == TB_S_RES1))
            compare_shadow(2'd0, bb_block_idx, 3'd1, "ATTN", TOK_FLAT);
        else if ((tb_state_d == TB_S_RES1) && (tb_state == TB_S_NORM2))
            compare_shadow(2'd1, bb_block_idx, 3'd2, "RES1", TOK_FLAT);
        else if ((tb_state_d == TB_S_NORM2) && (tb_state == TB_S_MLP_FEED))
            compare_shadow(2'd2, bb_block_idx, 3'd3, "NORM2", TOK_FLAT);
        else if ((tb_state_d == TB_S_MLP_WAIT) && (tb_state == TB_S_RES2))
            compare_shadow(2'd2, bb_block_idx, 3'd4, "MLP", TOK_FLAT);
        else if (tb_done && !tb_done_d)
            compare_shadow(2'd0, bb_block_idx, 3'd5, "BLOCK_OUT", TOK_FLAT);
    end
end

always @(posedge clk) begin
    if (!reset &&
        (bb_state_d == BB_S_RUN_SELECTED) && (bb_state == BB_S_BACKBONE_NORM)) begin
        $display("[STAGE] enter BACKBONE_NORM @ cycle %0d", cycle_cnt);
        compare_shadow(2'd0, 4'd6, 3'd5,
                    "PRE_NORM block6_after_block_out", TOK_FLAT);
    end
end

// Latch backbone done for final compare
always @(posedge clk) begin
    if (reset)
        done_seen <= 1'b0;
    else if (done)
        done_seen <= 1'b1;
end

initial begin
    $fsdbDumpfile("sglatrack_top.fsdb");
    $fsdbDumpvars;

    $set_toggle_region("u_DUT");
    $toggle_start();

    $readmemb({`GOLDEN_ACT, "/merged_tokens_bi.txt"},                    GOLD_MERGED);
    $readmemb({`GOLDEN_ACT, "/backbone_after_norm_backbone_out_bi.txt"}, GOLD_BB);
    $readmemb({`GOLDEN_ACT, "/template_post_embed_input_bi.txt"},        TEMPL_MEM);
    $readmemb({`GOLDEN_ACT, "/search_post_embed_input_bi.txt"},         SRCH_MEM);

    $display("[TB] backbone E2E test  N_TOKENS=%0d  TOK_FLAT=%0d", N_TOKENS, TOK_FLAT);
    $display("[TB] Golden dir: %s", `GOLDEN_ACT);
    $display("[TB] Per-block stages: NORM1 ATTN RES1 NORM2 MLP BLOCK_OUT");

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    done_seen  = 1'b0;
    cycle_cnt  = 0;
    tok_cnt    = 0;
    bb_state_d = BB_S_IDLE;
    tb_state_d = 4'd0;
    tb_done_d  = 1'b0;

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
    compare_shadow_bb_norm("BACKBONE_NORM backbone_after_norm_backbone_out", TOK_FLAT);

    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

initial begin
    #(CYCLE * 340_000_000);
    $display("[TB] TIMEOUT: backbone_top did not finish (cycle %0d)", cycle_cnt);
    $toggle_stop();
    $toggle_report("sglatrack_top_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
