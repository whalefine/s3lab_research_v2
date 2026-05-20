`timescale 1ns/10ps

// Accept common VCS typo: +define+NOM1_DEBUG (maps to DUMP_NORM1_DEBUG)
`ifdef NOM1_DEBUG
  `ifndef DUMP_NORM1_DEBUG
    `define DUMP_NORM1_DEBUG
  `endif
`endif

// External +define+block=... can break hierarchical paths like u_*.block_idx
`ifdef block
`undef block
`endif

// =============================================================================
// TEST_backbone.v -- backbone-only testbench (backbone_top vs numpy golden)
//
// DUT: backbone_top (same ROM macro ports as python/lib/models/verilog/backbone_top.v)
//
// Flow:
//   1. $readmemb template + search post-embedding (same layout as verilog/TEST.v)
//   2. reset; sel_block_i = 6; start
//   3. stream TOK_TOTAL tokens (template then search), gated like verilog/TEST.v
//   4. capture y_o on y_valid; optional compare to golden activation file
//
// Golden (Q8.8, one line per element, same as write_bi):
//   ./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt
//   Maps: vit_care_relu6_numpy_trunk_dim32_out/Activation/backbone_after_norm_backbone_out_bi.txt
//
// Compile: list all *.v in this directory + memory/ ROM/SRAM macros (see verilog rule).
// Run simv from a CWD where ./TXT_File/Activation/ exists (relative $readmemb paths).
//
// Optional: +define+BB_COMPARE_GOLDEN -- load golden file and check every y_o vs GOLD_BB[i]
//
// Norm1 debug (block0 norm1 only):
//   +define+DUMP_NORM1_DEBUG   (alias: +define+NOM1_DEBUG)
//   Stops when block0 norm1 finishes (enters S_ATTN_FEED on all tokens).
//
// Attention debug (block0 only):
//   +define+DUMP_ATTN_DEBUG
//   Loads 4 golden files (q, k, v, attn_out) and compares vs
//   u_DUT.u_tb.u_attn.{q,k,v}_buf at QKV->SPLIT transition (10240 elements each)
//   plus per-beat y_o vs gold_ao during u_attn.S_PROJ. Stops once block0
//   attention finishes. Cannot be combined with DUMP_NORM1_DEBUG.
//   Optional wgt prefetch debug on u_lin_qkv only:
//     +define+DUMP_WGT_PREFETCH   -> grep "WGT_" in simv.log (u_lin_qkv only; DUMP_WGT=1)
//
// Full backbone end compare:
//   +define+BB_COMPARE_GOLDEN
//
// Simulation log strings: English only (tooling may not render CJK).
// =============================================================================

module TEST_backbone;

parameter CYCLE = 2.0;
parameter [31:0] FSDB_START_MULT = 32'd00_000_000;

parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;
parameter FEAT_SZ     = FEAT_H * FEAT_W;
parameter TEMPL_TOT   = LENS_Z  * EMBED_DIM;
parameter SRCH_TOT    = FEAT_SZ * EMBED_DIM;
parameter TOK_TOTAL   = TEMPL_TOT + SRCH_TOT;
parameter N_TOKENS    = TOK_TOTAL / EMBED_DIM;

// ---------------------------------------------------------------------------
// DUT I/O
// ---------------------------------------------------------------------------
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

// Gate TB pushes: only when block 0 loads external stream (matches verilog/TEST.v)
// transformer_block S_LOAD_X is now state 4'd1 (unchanged in new revision).
wire tb_stream_gate =
    !u_DUT.tok_replay &&
    (u_DUT.u_tb.state == 4'd1);

function integer abs_diff16;
    input signed [15:0] a, b;
    reg signed [16:0] d;
    begin
        d = a - b;
        abs_diff16 = (d < 0) ? -d : d;
    end
endfunction

`ifdef DUMP_NORM1_DEBUG
// transformer_block state names (for log only)
localparam TB_S_LOAD_X    = 4'd1;
localparam TB_S_NORM1     = 4'd2;
localparam TB_S_ATTN_FEED = 4'd3;   // transformer_block.v: after norm1 on all tokens

wire [3:0]  dbg_blk   = u_DUT.block_idx;
wire [3:0]  dbg_state = u_DUT.u_tb.state;
wire [13:0] dbg_baddr = u_DUT.u_tb.buf_addr;
wire [8:0]  dbg_tok   = u_DUT.u_tb.tok_cnt;
wire [4:0]  dbg_feat  = u_DUT.u_tb.u_norm1.feat_addr_o[4:0];
wire        dbg_ln1yv = u_DUT.u_tb.ln1_yv;
wire signed [15:0] dbg_ln1y  = u_DUT.u_tb.ln1_y_sat;
wire signed [15:0] dbg_xin   = u_DUT.u_tb.xbuf_rp_data;
wire signed [15:0] dbg_wgt    = u_DUT.u_tb.wgt_i;
wire signed [15:0] dbg_bias   = u_DUT.u_tb.bias_i;
wire signed [15:0] dbg_mean   = u_DUT.u_tb.u_norm1.mean_q88;
wire signed [15:0] dbg_var    = u_DUT.u_tb.u_norm1.var_q88;
wire signed [15:0] dbg_invstd = u_DUT.u_tb.u_norm1.u_inv_sqrt.y_o;

reg [15:0] GOLD_MERGED [0:TOK_TOTAL-1];
reg [15:0] GOLD_NORM1  [0:TOK_TOTAL-1];
reg [15:0] GOLD_N1W     [0:EMBED_DIM-1];
reg [15:0] GOLD_N1B     [0:EMBED_DIM-1];

reg [31:0] merged_load_mism;
reg [31:0] norm1_mism;
reg        norm1_bad_found;
reg [13:0] norm1_first_bad;
reg [8:0]  norm1_bad_tok;
reg [4:0]  norm1_bad_feat;
reg [31:0] load_bad_found;
reg [13:0] load_first_bad;
reg [31:0] norm1_mism_le2;
reg [31:0] norm1_max_diff;
reg [4:0]  norm1_max_feat;
reg [8:0]  norm1_max_tok;
reg        gold_wgt_ok;
// Latch ln1_y_sat + feat index at posedge; compare on negedge (avoid addr/y_valid race).
reg [8:0]  cmp_tok;
reg [4:0]  cmp_feat;
reg signed [15:0] cmp_y;
reg        cmp_pending;
// Per-token max |rtl-gold| (scan which token first blows up)
reg [31:0] tok_max_diff [0:N_TOKENS-1];
reg [4:0]  tok_max_feat  [0:N_TOKENS-1];
`endif

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
always #(CYCLE/2.0) clk = ~clk;

reg [31:0] cycle_cnt;
always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

// ---------------------------------------------------------------------------
// Input memories + token stream
// ---------------------------------------------------------------------------
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

`ifdef DUMP_NORM1_DEBUG
// Compare each beat written to block0 x_buf vs merged golden (after_pos_drop)
always @(posedge clk) begin
    if (reset) begin
        merged_load_mism <= 32'd0;
        load_bad_found   <= 1'b0;
        load_first_bad   <= 14'd0;
    end else if (dbg_blk == 4'd0 && dbg_state == TB_S_LOAD_X && data_valid && tb_stream_gate) begin
        if (GOLD_MERGED[dbg_baddr] !== data_in[15:0]) begin
            merged_load_mism <= merged_load_mism + 32'd1;
            if (!load_bad_found) begin
                load_bad_found <= 1'b1;
                load_first_bad   <= dbg_baddr;
                $display("[NORM1_DBG] LOAD mismatch first at flat=%0d rtl=%h gold=%h diff=%0d",
                         dbg_baddr, data_in[15:0], GOLD_MERGED[dbg_baddr],
                         abs_diff16($signed(data_in), $signed(GOLD_MERGED[dbg_baddr])));
            end
        end
    end
end

// Latch norm1 beat at posedge y_valid (combo y_sat_o + feat_addr_o).
always @(posedge clk) begin
    if (reset)
        cmp_pending <= 1'b0;
    else if (dbg_blk == 4'd0 && dbg_state == TB_S_NORM1 && dbg_ln1yv) begin
        cmp_tok   <= dbg_tok;
        cmp_feat  <= dbg_feat;
        cmp_y     <= dbg_ln1y;
        cmp_pending <= 1'b1;
    end else
        cmp_pending <= 1'b0;
end

// Compare on negedge -- posedge compare mis-indexes y[k] vs gold[k+1] (rev5: ~10222 mism).
always @(negedge clk) begin
    if (reset) begin
        norm1_mism       <= 32'd0;
        norm1_mism_le2   <= 32'd0;
        norm1_max_diff   <= 32'd0;
        norm1_bad_found  <= 1'b0;
        norm1_first_bad  <= 14'd0;
        norm1_bad_tok    <= 9'd0;
        norm1_bad_feat   <= 5'd0;
    end else if (cmp_pending) begin
        begin : norm1_cmp
            reg [13:0] flat_i;
            reg [4:0]  feat_i;
            reg [15:0] gw;
            reg [15:0] gb;
            reg [31:0] diff_i;

            feat_i  = cmp_feat;
            flat_i  = cmp_tok * EMBED_DIM + {9'b0, feat_i};
            diff_i  = abs_diff16(cmp_y, $signed(GOLD_NORM1[flat_i]));

            if (diff_i > tok_max_diff[cmp_tok]) begin
                tok_max_diff[cmp_tok] <= diff_i;
                tok_max_feat[cmp_tok]  <= feat_i;
            end
            if (diff_i > norm1_max_diff) begin
                norm1_max_diff <= diff_i;
                norm1_max_tok  <= cmp_tok;
                norm1_max_feat <= feat_i;
            end

            if (GOLD_NORM1[flat_i] !== cmp_y[15:0]) begin
                norm1_mism <= norm1_mism + 32'd1;
                if (diff_i <= 32'd2)
                    norm1_mism_le2 <= norm1_mism_le2 + 32'd1;
                if (!norm1_bad_found) begin
                    norm1_bad_found <= 1'b1;
                    norm1_first_bad <= flat_i;
                    norm1_bad_tok   <= cmp_tok;
                    norm1_bad_feat  <= feat_i;
                    gw = GOLD_N1W[feat_i];
                    gb = GOLD_N1B[feat_i];
                    $display("[NORM1_DBG] OUT mismatch first at flat=%0d tok=%0d feat=%0d",
                             flat_i, cmp_tok, feat_i);
                    $display("  rtl_ln1=%h gold=%h diff=%0d",
                             cmp_y[15:0], GOLD_NORM1[flat_i], diff_i);
                    $display("  gold_merged=%h wgt=%h gold_w=%h bias=%h gold_b=%h",
                             GOLD_MERGED[flat_i], dbg_wgt, gw, dbg_bias, gb);
                    $display("  mean=%h var=%h inv_std=%h",
                             u_DUT.u_tb.u_norm1.mean_q88,
                             u_DUT.u_tb.u_norm1.var_q88,
                             u_DUT.u_tb.u_norm1.u_inv_sqrt.y_o);
                end
            end
        end
    end
end

`ifndef DUMP_NORM1_STATS
`define DUMP_NORM1_STATS
`endif
`ifdef DUMP_NORM1_STATS
// Per-token summary after each token norm1 completes
always @(posedge clk) begin
    if (!reset && dbg_blk == 4'd0 && dbg_state == TB_S_NORM1 &&
        u_DUT.u_tb.ln1_done && (u_DUT.u_tb.tok_cnt < 9'd3)) begin
        $display("[NORM1_TOK] tok=%0d done mean=%h var=%h inv_std=%h",
                 u_DUT.u_tb.tok_cnt,
                 u_DUT.u_tb.u_norm1.mean_q88,
                 u_DUT.u_tb.u_norm1.var_q88,
                 u_DUT.u_tb.u_norm1.u_inv_sqrt.y_o);
    end
end
`endif

// End of block0 load: summary before norm1 starts
always @(posedge clk) begin
    if (!reset && dbg_blk == 4'd0 && (dbg_state == TB_S_NORM1) &&
        (u_DUT.u_tb.tok_cnt == 9'd0) && u_DUT.u_tb.ln1_start_r) begin
        $display("[NORM1_DBG] block0 LOAD done: load_mism=%0d first_bad_flat=%0d (expect 0)",
                 merged_load_mism, load_first_bad);
        $display("  x_buf[0:3]=%h %h %h %h  gold_merged[0:3]=%h %h %h %h",
                 u_DUT.u_tb.x_buf[0], u_DUT.u_tb.x_buf[1],
                 u_DUT.u_tb.x_buf[2], u_DUT.u_tb.x_buf[3],
                 GOLD_MERGED[0], GOLD_MERGED[1], GOLD_MERGED[2], GOLD_MERGED[3]);
        if (gold_wgt_ok) begin
            $display("  gold norm1_w[0]=%h norm1_b[0]=%h (from TXT_File/Weight/)",
                     GOLD_N1W[0], GOLD_N1B[0]);
        end else begin
            $display("  [WARN] norm1 weight golden not loaded -- copy Weight/*.txt to TXT_File/Weight/");
        end
    end
end
`endif

// ---------------------------------------------------------------------------
// DUMP_ATTN_DEBUG -- Bit-accurate compare of block0 CARE attention vs golden.
//
//   Golden files (10240 lines each, Q8.8 16-bit binary one-per-line):
//     ./TXT_File/Activation/backbone_blocks_0_attn_after_qkv_q_bi.txt
//     ./TXT_File/Activation/backbone_blocks_0_attn_after_qkv_k_bi.txt
//     ./TXT_File/Activation/backbone_blocks_0_attn_after_qkv_v_bi.txt
//     ./TXT_File/Activation/backbone_blocks_0_after_attn_attn_out_bi.txt
//
// q/k/v flatten order: (H=4, N=320, d=8) C-order -- index = h*2560 + n*8 + di.
// attn_out flatten order: (N=320, C=32) -- index = n*32 + c.
//
// Compare points:
//   1. QKV->SPLIT transition inside u_attn (q/k/v_buf hold post-QKV-linear
//      values BEFORE in-place scale+ReLU6). Compare 10240 elements each.
//   2. u_attn.y_o stream during S_PROJ vs gold_ao (per-beat).
//
// All $display strings are ASCII English.
// Cannot be combined with DUMP_NORM1_DEBUG (that one finishes before attention).
// ---------------------------------------------------------------------------
`ifdef DUMP_ATTN_DEBUG
reg [15:0] GOLD_Q  [0:10239];
reg [15:0] GOLD_K  [0:10239];
reg [15:0] GOLD_V  [0:10239];
reg [15:0] GOLD_AO [0:10239];

reg [31:0] attn_mism_q, attn_mism_k, attn_mism_v, attn_mism_ao;
reg [31:0] attn_cnt_ao;
reg [31:0] attn_first_bad_q, attn_first_bad_k, attn_first_bad_v, attn_first_bad_ao;
reg        attn_q_done, attn_k_done, attn_v_done;
reg        attn_qkv_compared;
reg        attn_done_seen;

// u_attn FSM state encoding (must match care_attention.v parameter list)
localparam ATTN_S_QKV    = 4'd2;
localparam ATTN_S_SPLIT  = 4'd3;
localparam ATTN_S_PROJ   = 4'd9;

// Trigger pulse: u_attn just entered S_SPLIT from S_QKV (q/k/v_buf are
// fully populated by the linear, NOT yet modified by SPLIT scale+ReLU6).
wire attn_qkv_done_pulse = (u_DUT.u_tb.u_attn.state      == ATTN_S_SPLIT) &&
                           (u_DUT.u_tb.u_attn.prev_state == ATTN_S_QKV);

// Block0 gate (only compare block0's attention vs golden)
wire attn_dbg_block0 = (u_DUT.block_idx == 4'd0);

// q/k/v_buf bulk compare -- done as a single procedural scan triggered once.
// Use a separate always block since the loop is large (10240 elements).
always @(posedge clk) begin
    if (reset) begin
        attn_mism_q       <= 32'd0;
        attn_mism_k       <= 32'd0;
        attn_mism_v       <= 32'd0;
        attn_first_bad_q  <= 32'hFFFF_FFFF;
        attn_first_bad_k  <= 32'hFFFF_FFFF;
        attn_first_bad_v  <= 32'hFFFF_FFFF;
        attn_qkv_compared <= 1'b0;
    end else if (attn_dbg_block0 && attn_qkv_done_pulse && !attn_qkv_compared) begin : qkv_scan
        integer i;
        reg [31:0] mq, mk, mv;
        reg [31:0] fbq, fbk, fbv;
        mq = 32'd0; mk = 32'd0; mv = 32'd0;
        fbq = 32'hFFFF_FFFF; fbk = 32'hFFFF_FFFF; fbv = 32'hFFFF_FFFF;
        for (i = 0; i < 10240; i = i + 1) begin
            if (u_DUT.u_tb.u_attn.q_buf[i] !== GOLD_Q[i]) begin
                mq = mq + 32'd1;
                if (fbq == 32'hFFFF_FFFF) fbq = i[31:0];
            end
            if (u_DUT.u_tb.u_attn.k_buf[i] !== GOLD_K[i]) begin
                mk = mk + 32'd1;
                if (fbk == 32'hFFFF_FFFF) fbk = i[31:0];
            end
            if (u_DUT.u_tb.u_attn.v_buf[i] !== GOLD_V[i]) begin
                mv = mv + 32'd1;
                if (fbv == 32'hFFFF_FFFF) fbv = i[31:0];
            end
        end
        attn_mism_q      <= mq;
        attn_mism_k      <= mk;
        attn_mism_v      <= mv;
        attn_first_bad_q <= fbq;
        attn_first_bad_k <= fbk;
        attn_first_bad_v <= fbv;
        attn_qkv_compared <= 1'b1;

        $display("[ATTN_CMP_Q] match=%0d/10240 mism=%0d first_bad_idx=%0d rtl[0..3]=%h %h %h %h gold[0..3]=%h %h %h %h",
                 10240 - mq, mq, fbq,
                 u_DUT.u_tb.u_attn.q_buf[0], u_DUT.u_tb.u_attn.q_buf[1],
                 u_DUT.u_tb.u_attn.q_buf[2], u_DUT.u_tb.u_attn.q_buf[3],
                 GOLD_Q[0], GOLD_Q[1], GOLD_Q[2], GOLD_Q[3]);
        $display("[ATTN_CMP_K] match=%0d/10240 mism=%0d first_bad_idx=%0d rtl[0..3]=%h %h %h %h gold[0..3]=%h %h %h %h",
                 10240 - mk, mk, fbk,
                 u_DUT.u_tb.u_attn.k_buf[0], u_DUT.u_tb.u_attn.k_buf[1],
                 u_DUT.u_tb.u_attn.k_buf[2], u_DUT.u_tb.u_attn.k_buf[3],
                 GOLD_K[0], GOLD_K[1], GOLD_K[2], GOLD_K[3]);
        $display("[ATTN_CMP_V] match=%0d/10240 mism=%0d first_bad_idx=%0d rtl[0..3]=%h %h %h %h gold[0..3]=%h %h %h %h",
                 10240 - mv, mv, fbv,
                 u_DUT.u_tb.u_attn.v_buf[0], u_DUT.u_tb.u_attn.v_buf[1],
                 u_DUT.u_tb.u_attn.v_buf[2], u_DUT.u_tb.u_attn.v_buf[3],
                 GOLD_V[0], GOLD_V[1], GOLD_V[2], GOLD_V[3]);
    end
end

// attn_out per-beat compare -- u_attn.y_valid pulses during S_PROJ; each
// pulse emits one Q8.8 element of the final attn output for block0.
always @(posedge clk) begin
    if (reset) begin
        attn_mism_ao      <= 32'd0;
        attn_cnt_ao       <= 32'd0;
        attn_first_bad_ao <= 32'hFFFF_FFFF;
    end else if (attn_dbg_block0 && u_DUT.u_tb.u_attn.y_valid) begin
        if (u_DUT.u_tb.u_attn.y_o !== $signed(GOLD_AO[attn_cnt_ao])) begin
            attn_mism_ao <= attn_mism_ao + 32'd1;
            if (attn_first_bad_ao == 32'hFFFF_FFFF)
                attn_first_bad_ao <= attn_cnt_ao;
        end
        attn_cnt_ao <= attn_cnt_ao + 32'd1;
    end
end

// Watch for u_attn.done (block0 only) -- used by stimulus to early-finish
always @(posedge clk) begin
    if (reset)
        attn_done_seen <= 1'b0;
    else if (attn_dbg_block0 && u_DUT.u_tb.u_attn.done)
        attn_done_seen <= 1'b1;
end
`endif

// ---------------------------------------------------------------------------
// Capture backbone norm stream + optional golden compare
// ---------------------------------------------------------------------------
`ifdef BB_COMPARE_GOLDEN
reg [15:0] GOLD_BB [0:TOK_TOTAL-1];
`endif

reg [13:0] rtl_bb_cnt;
reg [31:0] bb_mism;

always @(posedge clk) begin
    if (reset) begin
        rtl_bb_cnt <= 14'd0;
        bb_mism    <= 32'd0;
    end else if (y_valid) begin
`ifdef BB_COMPARE_GOLDEN
        if (GOLD_BB[rtl_bb_cnt] !== y_o[15:0])
            bb_mism <= bb_mism + 32'd1;
`endif
        rtl_bb_cnt <= rtl_bb_cnt + 14'd1;
    end
end

// ---------------------------------------------------------------------------
// Done: summary
// ---------------------------------------------------------------------------
always @(negedge clk) begin
    if (done) begin
        $display("\n---- backbone_top done @ cycle %0d ----", cycle_cnt);
        $display("  tokens_in: template=%0d search=%0d total=%0d", TEMPL_TOT, SRCH_TOT, TOK_TOTAL);
        $display("  sel_block_i = %0d", sel_block_i);
        $display("  y_valid samples captured = %0d (expect %0d)", rtl_bb_cnt, TOK_TOTAL);
`ifdef BB_COMPARE_GOLDEN
        if (bb_mism == 0 && rtl_bb_cnt == TOK_TOTAL)
            $display("  [PASS] backbone_after_norm_backbone_out (golden) all %0d elems match", TOK_TOTAL);
        else if (rtl_bb_cnt != TOK_TOTAL)
            $display("  [FAIL] sample count mismatch (PIPELINE?)");
        else
            $display("  [FAIL] backbone golden mismatches = %0d (see BB_COMPARE_GOLDEN)", bb_mism);
`else
        $display("  [INFO] re-run with +define+BB_COMPARE_GOLDEN to diff vs backbone_after_norm_backbone_out_bi.txt");
`endif
        $toggle_stop();
        $toggle_report("backbone_only_rtl.saif", 1.0e-9, "u_DUT");
        $finish;
    end
end

// ---------------------------------------------------------------------------
// FSDB (Synopsys VCS): edit out if your simulator does not support these tasks
// ---------------------------------------------------------------------------
initial begin
    $fsdbDumpfile("backbone_tb.fsdb");
    $fsdbDumpvars(0, TEST_backbone.u_DUT);
    $fsdbDumpoff;
end

initial begin
    #(CYCLE * FSDB_START_MULT);
    $fsdbDumpon;
end

initial begin
    $set_toggle_region("u_DUT");
    $toggle_start();
end

// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------
initial begin
`ifdef DUMP_NORM1_DEBUG
    $readmemb("./TXT_File/Activation/after_pos_drop_out_bi.txt", GOLD_MERGED);
    $readmemb("./TXT_File/Activation/backbone_blocks_0_after_norm1_out_bi.txt", GOLD_NORM1);
    $readmemb("./TXT_File/Weight/backbone_blocks_0_norm1_weight_bi.txt", GOLD_N1W);
    $readmemb("./TXT_File/Weight/backbone_blocks_0_norm1_bias_bi.txt", GOLD_N1B);
    gold_wgt_ok = (GOLD_N1W[0] !== 16'hxxxx) && (GOLD_N1B[0] !== 16'hxxxx);
    $display("[TB] TEST_backbone rev=7 (norm1 compare on negedge; expect ~8836 mism vs float golden)");
    $display("[TB] DUMP_NORM1_DEBUG: merged+norm1 act, norm1 w/b from TXT_File/Weight/");
    if (!gold_wgt_ok)
        $display("[TB] WARN: norm1 weight/bias golden unreadable -- symlink Weight/ from numpy trunk output");
    merged_load_mism = 0;
    norm1_mism       = 0;
    norm1_mism_le2   = 0;
    norm1_max_diff   = 0;
    norm1_bad_found  = 0;
    load_bad_found   = 0;
    begin : init_tok_scan
        integer ti;
        for (ti = 0; ti < N_TOKENS; ti = ti + 1) begin
            tok_max_diff[ti] = 0;
            tok_max_feat[ti] = 0;
        end
    end
`endif
`ifdef BB_COMPARE_GOLDEN
    $readmemb("./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt", GOLD_BB);
    $display("[TB] Loaded backbone golden: %0d values", TOK_TOTAL);
`endif
`ifdef DUMP_ATTN_DEBUG
    $readmemb("./TXT_File/Activation/backbone_blocks_0_attn_after_qkv_q_bi.txt",   GOLD_Q);
    $readmemb("./TXT_File/Activation/backbone_blocks_0_attn_after_qkv_k_bi.txt",   GOLD_K);
    $readmemb("./TXT_File/Activation/backbone_blocks_0_attn_after_qkv_v_bi.txt",   GOLD_V);
    $readmemb("./TXT_File/Activation/backbone_blocks_0_after_attn_attn_out_bi.txt", GOLD_AO);
    $display("[ATTN_DBG] golden loaded: q/k/v=10240 ea, ao=10240 (block0 CARE attention)");
    attn_mism_q       = 0;
    attn_mism_k       = 0;
    attn_mism_v       = 0;
    attn_mism_ao      = 0;
    attn_cnt_ao       = 0;
    attn_first_bad_q  = 32'hFFFF_FFFF;
    attn_first_bad_k  = 32'hFFFF_FFFF;
    attn_first_bad_v  = 32'hFFFF_FFFF;
    attn_first_bad_ao = 32'hFFFF_FFFF;
    attn_qkv_compared = 1'b0;
    attn_done_seen    = 1'b0;
`endif

    $readmemb("./TXT_File/Activation/template_post_embed_input_bi.txt", TEMPL_MEM);
    $display("[TB] Loaded template_post_embed_input_bi.txt: %0d values", TEMPL_TOT);
    $readmemb("./TXT_File/Activation/search_post_embed_input_bi.txt", SRCH_MEM);
    $display("[TB] Loaded search_post_embed_input_bi.txt: %0d values", SRCH_TOT);

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 0;
    tok_cnt    = 0;

    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

`ifdef DUMP_NORM1_DEBUG
    // Stop early once block0 norm1 finished (all tokens) -- save sim time while debugging
    wait (u_DUT.block_idx == 4'd0 &&
          u_DUT.u_tb.state == TB_S_ATTN_FEED &&
          u_DUT.u_tb.tok_cnt == N_TOKENS[8:0]);
    #(CYCLE * 4);
    $display("\n[NORM1_DBG] === block0 norm1 phase ended (entered S_ATTN_FEED) ===");
    $display("  load_mism=%0d norm1_out_mism=%0d mism_with_diff_le2=%0d",
             merged_load_mism, norm1_mism, norm1_mism_le2);
    $display("  first_bad_flat=%0d max_abs_diff=%0d at tok=%0d feat=%0d",
             norm1_first_bad, norm1_max_diff, norm1_max_tok, norm1_max_feat);
    begin : tok_scan_print
        integer ti;
        for (ti = 0; ti < 8; ti = ti + 1)
            $display("  [TOK_SCAN] tok=%0d max_diff=%0d at_feat=%0d",
                     ti, tok_max_diff[ti], tok_max_feat[ti]);
        $display("  [TOK_SCAN] tok=227 max_diff=%0d at_feat=%0d (worst from prior run)",
                 tok_max_diff[227], tok_max_feat[227]);
    end
    if (merged_load_mism == 0 && norm1_mism == 0)
        $display("  [PASS] block0 load + norm1 bit-exact vs golden");
    else if (merged_load_mism != 0)
        $display("  [HINT] fix LOAD first (merged/concat/TB gate) before norm1 math");
    else if (norm1_mism == norm1_mism_le2)
        $display("  [HINT] all mismatches are <=2 LSB -- check ROM vs Weight/ or inv_sqrt");
    else if (norm1_mism_le2 > (norm1_mism >> 1))
        $display("  [HINT] mostly <=2 LSB -- float golden vs fixed inv_sqrt_nr; or tune round/eps");
    else if (norm1_mism > 32'd9500)
        $display("  [HINT] ~10222 mism -- TB index race; recompile rev=7 (negedge compare)");
    else
        $display("  [HINT] large diffs -- check RTL layer_norm vs bit-accurate python model");
    $finish;
`endif

`ifdef DUMP_ATTN_DEBUG
    // Wait until block0 CARE attention finishes (u_attn.done pulse captured)
    wait (attn_done_seen);
    #(CYCLE * 4);
    $display("\n[ATTN_DBG] === block0 CARE attention finished ===");
    $display("  [ATTN_CMP_Q]  total_mism=%0d  first_bad_idx=%0d", attn_mism_q, attn_first_bad_q);
    $display("  [ATTN_CMP_K]  total_mism=%0d  first_bad_idx=%0d", attn_mism_k, attn_first_bad_k);
    $display("  [ATTN_CMP_V]  total_mism=%0d  first_bad_idx=%0d", attn_mism_v, attn_first_bad_v);
    $display("  [ATTN_CMP_AO] total_mism=%0d/%0d  first_bad_idx=%0d",
             attn_mism_ao, attn_cnt_ao, attn_first_bad_ao);
    if (attn_mism_q == 0 && attn_mism_k == 0 && attn_mism_v == 0 && attn_mism_ao == 0)
        $display("  [PASS] block0 q/k/v + attn_out bit-exact vs golden (10240 elems each)");
    else if (attn_mism_q != 0 || attn_mism_k != 0 || attn_mism_v != 0)
        $display("  [FAIL] QKV linear mismatch -- check linear.v MAC / ROM addr / bias decode");
    else
        $display("  [FAIL] QKV ok but attn_out diverged -- check SPLIT/K_MEAN/QK_MEAN/Z_RECIP/KV/ATTN stages");
    $finish;
`endif

    // Backbone-only wall (~53M+ MAC path per project notes; margin for safety)
    #(CYCLE * 33_000_000);
    $display("[TB] TIMEOUT: backbone_top did not finish within 33M cycles");
    $toggle_stop();
    $toggle_report("backbone_only_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
