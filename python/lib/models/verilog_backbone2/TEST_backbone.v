`timescale 1ns/10ps

// External +define+block=... can break hierarchical paths like u_*.block_idx
`ifdef block
`undef block
`endif

// ---------------------------------------------------------------------------
// Activation buffer backend (mutually exclusive; pass at most one +define to VCS)
//   USE_REG_BUF  -- legacy reg arrays inside DUT (default when neither is set)
//   USE_SRAM_BUF -- SP SRAM macros; include your prepared sram*.v in the file list
// Do NOT pass both +define+USE_REG_BUF and +define+USE_SRAM_BUF.
// ---------------------------------------------------------------------------
`ifdef USE_REG_BUF
`ifdef USE_SRAM_BUF
`define TB_BUF_MODE_ERROR
`endif
`endif

`ifndef USE_REG_BUF
`ifndef USE_SRAM_BUF
`define USE_REG_BUF
`endif
`endif

// =============================================================================
// TEST_backbone.v -- backbone_top end-to-end test (blocks 0..6 + backbone_norm)
//
// Flow:
//   1. Load template + search post-embedding from TXT_File/Activation/
//   2. reset; sel_block_i = 6; pulse start
//   3. backbone_top runs blocks 0..START_LAYER, adaptive block 6, backbone_norm
//   4. Compare y_o stream vs golden; print [PASS] or [FAIL] at done
//
// Golden: ./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt
// Run simv from a directory where ./TXT_File/Activation/ resolves.
//
// VCS examples (add memory/ ROM list per your flow):
//
//   Reg arrays (default; same as +define+USE_REG_BUF):
//     vcs verilog_backbone2/*.v memory/*.v \
//       +define+TSMC_CM_NO_WARNING +define+USE_REG_BUF | tee runvcs.log
//
//   SP SRAM macros (compiler Sram_tok1.v / Sram_tok2.v only, no *_wrap.v):
//     vcs verilog_backbone2/*.v memory/*.v <path>/Sram_tok1.v <path>/Sram_tok2.v \
//       +define+TSMC_CM_NO_WARNING +define+USE_SRAM_BUF | tee runvcs.log
//     Note: Sram_tok1 is instanced 4x (backbone s1 + transformer_block u_sram_x/tmp x2).
//
//   Block 0 only (compare u_tb stream, finish after block 0; much shorter runtime):
//     vcs ... +define+CHECK_BLOCK0_ONLY +define+USE_SRAM_BUF | tee runvcs.log
//     Golden: ./TXT_File/Activation/backbone_blocks_0_after_block_out_bi.txt
//
//   SRAM full-chain step debug (optional; grep DBG_STEP in simv.log):
//     vcs ... +define+USE_SRAM_BUF +define+DUMP_SRAM_BUF_DEBUG | tee runvcs.log
//
//   Block6 boundary debug (grep DBG_B6 in simv.log):
//     vcs ... +define+USE_SRAM_BUF +define+DUMP_BLOCK6_DEBUG | tee runvcs.log
//
//   Norm + S1 narrow check (grep NS1_ in simv.log; needs block6 golden for input):
//     vcs ... +define+USE_SRAM_BUF +define+DUMP_NORM_S1_DEBUG | tee runvcs.log
//     Golden: backbone_blocks_6_after_block_out_bi.txt (norm in)
//             backbone_after_norm_backbone_out_bi.txt (norm out / y_o)
//     DUT: NS1_S1RD_POSEDGE / NS1_S1RD_NEGEDGE (grep NS1_S1RD_ in log).
//     TB:  NS1_GOLDCHK at USE posedge/negedge vs golden[tgt/out_addr].
//
//   ./simv | tee simv.log
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

wire tb_stream_gate =
    !u_DUT.tok_replay &&
    (u_DUT.u_tb.state == 4'd1);

`ifndef CHECK_BLOCK0_ONLY
reg [15:0] GOLD_BB [0:TOK_TOTAL-1];
reg [13:0] rtl_bb_cnt;
reg [31:0] bb_mism;
reg [31:0] bb_first_bad;
`else
reg [15:0] GOLD_B0 [0:TOK_TOTAL-1];
reg [13:0] rtl_b0_cnt;
reg [31:0] b0_mism;
reg [31:0] b0_first_bad;
wire       b0_sample_en =
    u_DUT.tb_y_valid &&
    (u_DUT.block_idx == 4'd0) &&
    (u_DUT.state == 3'd1);   // S_RUN_FIXED
`endif

// Stepwise SRAM debug: golden spot-checks at block1 replay / block1 out / norm / S_OUT.
// Requires +define+USE_SRAM_BUF (hierarchical s2_q, tb_x_mux, etc.). Default off.
`ifdef DUMP_SRAM_BUF_DEBUG
`ifndef CHECK_BLOCK0_ONLY
reg [15:0] DBG_GOLD_B0 [0:TOK_TOTAL-1];
reg [15:0] DBG_GOLD_B1 [0:TOK_TOTAL-1];
reg [3:0]  dbg_rp_chk;       // STEP2: first N block1 replay beats vs block0 golden
reg [3:0]  dbg_b1_y_chk;     // STEP3: first N block1 tb_y vs block1 golden
reg        dbg_s2;           // STEP1 done
reg        dbg_s3;           // STEP3 banner
reg        dbg_s4;           // STEP4 norm enter
reg        dbg_s5;           // STEP5 first bn write
reg        dbg_s6;           // STEP6 first y_o
reg [2:0]  dbg_state_d;
wire [13:0] dbg_replay_idx = u_DUT.tok_rd_ptr;
wire        dbg_replay_en =
    u_DUT.tok_rp_phase &&
    (u_DUT.block_idx == 4'd1) &&
    (u_DUT.tok_rd_ptr < TOK_TOTAL);
wire        dbg_b1_y_en =
    u_DUT.tb_y_valid &&
    (u_DUT.block_idx == 4'd1);
`endif
`endif

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

`ifndef CHECK_BLOCK0_ONLY
always @(posedge clk) begin
    if (reset) begin
        rtl_bb_cnt   <= 14'd0;
        bb_mism      <= 32'd0;
        bb_first_bad <= 32'hFFFF_FFFF;
    end else if (y_valid) begin
        if (GOLD_BB[rtl_bb_cnt] !== y_o[15:0]) begin
            bb_mism <= bb_mism + 32'd1;
            if (bb_first_bad == 32'hFFFF_FFFF)
                bb_first_bad <= {18'b0, rtl_bb_cnt};
        end
        rtl_bb_cnt <= rtl_bb_cnt + 14'd1;
    end
end
`else
always @(posedge clk) begin
    if (reset) begin
        rtl_b0_cnt   <= 14'd0;
        b0_mism      <= 32'd0;
        b0_first_bad <= 32'hFFFF_FFFF;
    end else if (b0_sample_en) begin
        if (GOLD_B0[rtl_b0_cnt] !== u_DUT.tb_y[15:0]) begin
            b0_mism <= b0_mism + 32'd1;
            if (b0_first_bad == 32'hFFFF_FFFF)
                b0_first_bad <= {18'b0, rtl_b0_cnt};
        end
        rtl_b0_cnt <= rtl_b0_cnt + 14'd1;
    end
end

always @(negedge clk) begin
    if (u_DUT.tb_done && (u_DUT.block_idx == 4'd0) && (u_DUT.state == 3'd1)) begin
        $display("\n---- block 0 done @ cycle %0d (CHECK_BLOCK0_ONLY) ----", cycle_cnt);
        $display("  tb_y_valid samples = %0d (expect %0d)", rtl_b0_cnt, TOK_TOTAL);
        if (rtl_b0_cnt != TOK_TOTAL)
            $display("  [FAIL] block0 sample count mismatch");
        else if (b0_mism == 0)
            $display("  [PASS] backbone_blocks_0_after_block_out matches golden (%0d elems)",
                     TOK_TOTAL);
        else
            $display("  [FAIL] backbone_blocks_0_after_block_out mismatches = %0d / %0d  first_bad_idx = %0d",
                     b0_mism, TOK_TOTAL, b0_first_bad);
        $finish;
    end
end
`endif

`ifndef CHECK_BLOCK0_ONLY
always @(negedge clk) begin
    if (done) begin
        $display("\n---- backbone_top done @ cycle %0d ----", cycle_cnt);
        $display("  sel_block_i = %0d", sel_block_i);
        $display("  y_valid samples = %0d (expect %0d)", rtl_bb_cnt, TOK_TOTAL);
        if (rtl_bb_cnt != TOK_TOTAL)
            $display("  [FAIL] sample count mismatch");
        else if (bb_mism == 0)
            $display("  [PASS] backbone_after_norm_backbone_out matches golden (%0d elems)",
                     TOK_TOTAL);
        else
            $display("  [FAIL] backbone_after_norm_backbone_out mismatches = %0d / %0d  first_bad_idx = %0d",
                     bb_mism, TOK_TOTAL, bb_first_bad);
        $toggle_stop();
        $toggle_report("backbone_only_rtl.saif", 1.0e-9, "u_DUT");
        $finish;
    end
end
`endif

`ifdef DUMP_SRAM_BUF_DEBUG
`ifndef CHECK_BLOCK0_ONLY
// Spot-check vs golden at replay / per-block write / backbone_norm / final out.
always @(posedge clk) begin
    if (reset) begin
        dbg_rp_chk   <= 4'd0;
        dbg_b1_y_chk <= 4'd0;
        dbg_s2       <= 1'b0;
        dbg_s3       <= 1'b0;
        dbg_s4       <= 1'b0;
        dbg_s5       <= 1'b0;
        dbg_s6       <= 1'b0;
        dbg_state_d  <= 3'd0;
    end else begin
        dbg_state_d <= u_DUT.state;

        // STEP1: block0 finished (s2 should hold block0 out until block1 overwrites)
        if (u_DUT.tb_done && (u_DUT.block_idx == 4'd0) && !dbg_s2) begin
            dbg_s2 <= 1'b1;
            $display("[DBG_STEP1] block0 done @ cycle %0d  tok_wr_ptr=%0d tok_replay=%0d",
                     cycle_cnt, u_DUT.tok_wr_ptr, u_DUT.tok_replay);
        end

        // STEP2: block1 s2 replay USE beat (tok_rp_phase=1, rd_ptr = golden index)
        if (dbg_replay_en && (dbg_rp_chk < 4'd8)) begin
            if (DBG_GOLD_B0[dbg_replay_idx] !== u_DUT.tb_x_mux[15:0])
                $display("[DBG_STEP2_FAIL] replay idx=%0d gold=%h rtl_tb_x=%h s2_q=%h rd_ptr=%0d s2_addr=%0d",
                         dbg_replay_idx, DBG_GOLD_B0[dbg_replay_idx],
                         u_DUT.tb_x_mux[15:0], u_DUT.s2_q, u_DUT.tok_rd_ptr, u_DUT.bt_s2_addr);
            else
                $display("[DBG_STEP2_OK]   replay idx=%0d data=%h rd_ptr=%0d",
                         dbg_replay_idx, u_DUT.tb_x_mux[15:0], u_DUT.tok_rd_ptr);
            dbg_rp_chk <= dbg_rp_chk + 4'd1;
        end

        // STEP3: block1 overwrites s2 (first beats vs block1 golden)
        if (dbg_b1_y_en && (dbg_b1_y_chk < 4'd4)) begin
            if (DBG_GOLD_B1[{11'b0, dbg_b1_y_chk}] !== u_DUT.tb_y[15:0])
                $display("[DBG_STEP3_FAIL] block1 tb_y n=%0d gold=%h rtl=%h wr_ptr=%0d",
                         dbg_b1_y_chk, DBG_GOLD_B1[{11'b0, dbg_b1_y_chk}],
                         u_DUT.tb_y[15:0], u_DUT.tok_wr_ptr);
            else
                $display("[DBG_STEP3_OK]   block1 tb_y n=%0d data=%h wr_ptr=%0d",
                         dbg_b1_y_chk, u_DUT.tb_y[15:0], u_DUT.tok_wr_ptr);
            dbg_b1_y_chk <= dbg_b1_y_chk + 4'd1;
        end
        if (u_DUT.tb_done && (u_DUT.block_idx == 4'd1) && !dbg_s3) begin
            dbg_s3 <= 1'b1;
            $display("[DBG_STEP3] block1 done @ cycle %0d  tok_wr_ptr=%0d",
                     cycle_cnt, u_DUT.tok_wr_ptr);
        end

        // STEP4: enter backbone norm (reads s2, writes s1)
        if ((dbg_state_d != 3'd3) && (u_DUT.state == 3'd3) && !dbg_s4) begin
            dbg_s4 <= 1'b1;
            $display("[DBG_STEP4] enter S_BACKBONE_NORM @ cycle %0d  bn_tok_cnt=%0d",
                     cycle_cnt, u_DUT.bn_tok_cnt);
        end

        // STEP5: first norm writes to s1
        if (u_DUT.bn_y_valid && !dbg_s5) begin
            dbg_s5 <= 1'b1;
            if (GOLD_BB[u_DUT.bn_cap_flat] !== u_DUT.bn_y_sat[15:0])
                $display("[DBG_STEP5_FAIL] first s1_wr flat=%0d gold=%h rtl=%h tok=%0d feat_addr=%0d",
                         u_DUT.bn_cap_flat, GOLD_BB[u_DUT.bn_cap_flat],
                         u_DUT.bn_y_sat[15:0], u_DUT.bn_tok_cnt, u_DUT.bn_feat_addr);
            else
                $display("[DBG_STEP5_OK]   first s1_wr flat=%0d data=%h feat_addr=%0d",
                         u_DUT.bn_cap_flat, u_DUT.bn_y_sat[15:0], u_DUT.bn_feat_addr);
        end

        // STEP6: first S_OUT beat (s1 read latency)
        if (y_valid && !dbg_s6) begin
            dbg_s6 <= 1'b1;
            if (GOLD_BB[14'd0] !== y_o[15:0])
                $display("[DBG_STEP6_FAIL] first y_o gold=%h rtl=%h out_rd_addr=%0d phase=%0d s1_q=%h",
                         GOLD_BB[14'd0], y_o[15:0], u_DUT.out_rd_addr,
                         u_DUT.out_rd_phase, u_DUT.s1_q);
            else
                $display("[DBG_STEP6_OK]   first y_o data=%h out_rd_addr=%0d phase=%0d",
                         y_o[15:0], u_DUT.out_rd_addr, u_DUT.out_rd_phase);
        end

        // Early mismatch on first 4 final outputs
        if (y_valid && (rtl_bb_cnt < 14'd4)) begin
            if (GOLD_BB[rtl_bb_cnt] !== y_o[15:0])
                $display("[DBG_STEP7_FAIL] y_o[%0d] gold=%h rtl=%h s1_q=%h out_phase=%0d",
                         rtl_bb_cnt, GOLD_BB[rtl_bb_cnt], y_o[15:0],
                         u_DUT.s1_q, u_DUT.out_rd_phase);
        end
    end
end

`endif
`endif

`ifdef DUMP_BLOCK6_DEBUG
`ifndef CHECK_BLOCK0_ONLY
reg [15:0] DBG_GOLD_B5 [0:TOK_TOTAL-1];
reg [15:0] DBG_GOLD_B6 [0:TOK_TOTAL-1];
reg [3:0]  dbg_b6_rp_n;
reg [3:0]  dbg_b6_wr_n;
reg        dbg_b5_done_f;
reg        dbg_b6_done_f;
reg        dbg_norm_enter_f;
reg [2:0]  dbg_b6_st_d;
wire [13:0] b6_rp_idx = u_DUT.tok_rd_ptr;

always @(posedge clk) begin
    if (reset) begin
        dbg_b6_rp_n      <= 4'd0;
        dbg_b6_wr_n      <= 4'd0;
        dbg_b5_done_f    <= 1'b0;
        dbg_b6_done_f    <= 1'b0;
        dbg_norm_enter_f <= 1'b0;
        dbg_b6_st_d      <= 3'd0;
    end else begin
        dbg_b6_st_d <= u_DUT.state;

        // After block5: s2 holds block5 out (block6 input source)
        if (u_DUT.tb_done && (u_DUT.block_idx == 4'd5) && !dbg_b5_done_f) begin
            dbg_b5_done_f <= 1'b1;
            $display("[DBG_B6_PRE] block5 done @ cycle %0d  next=block6 replay from s2",
                     cycle_cnt);
            $display("  gold_b5[0..3]=%h %h %h %h",
                     DBG_GOLD_B5[0], DBG_GOLD_B5[1], DBG_GOLD_B5[2], DBG_GOLD_B5[3]);
        end

        // Block6 replay USE: compare tb_x vs block5 golden
        if (u_DUT.tok_rp_phase && (u_DUT.block_idx == 4'd6) &&
            (dbg_b6_rp_n < 4'd8)) begin
            if (DBG_GOLD_B5[b6_rp_idx] !== u_DUT.tb_x_mux[15:0])
                $display("[DBG_B6_REPLAY_FAIL] idx=%0d gold_b5=%h rtl_x=%h s2_q=%h rd_ptr=%0d",
                         b6_rp_idx, DBG_GOLD_B5[b6_rp_idx], u_DUT.tb_x_mux[15:0],
                         u_DUT.s2_q, u_DUT.tok_rd_ptr);
            else
                $display("[DBG_B6_REPLAY_OK]   idx=%0d data=%h rd_ptr=%0d",
                         b6_rp_idx, u_DUT.tb_x_mux[15:0], u_DUT.tok_rd_ptr);
            dbg_b6_rp_n <= dbg_b6_rp_n + 4'd1;
        end

        // Block6 write to s2: compare tb_y vs block6 golden
        if (u_DUT.tb_y_valid && (u_DUT.block_idx == 4'd6) && (dbg_b6_wr_n < 4'd8)) begin
            if (DBG_GOLD_B6[{11'b0, dbg_b6_wr_n}] !== u_DUT.tb_y[15:0])
                $display("[DBG_B6_WR_FAIL] n=%0d gold_b6=%h rtl_y=%h wr_ptr=%0d",
                         dbg_b6_wr_n, DBG_GOLD_B6[{11'b0, dbg_b6_wr_n}],
                         u_DUT.tb_y[15:0], u_DUT.tok_wr_ptr);
            else
                $display("[DBG_B6_WR_OK]   n=%0d data=%h wr_ptr=%0d",
                         dbg_b6_wr_n, u_DUT.tb_y[15:0], u_DUT.tok_wr_ptr);
            dbg_b6_wr_n <= dbg_b6_wr_n + 4'd1;
        end

        if (u_DUT.tb_done && (u_DUT.block_idx == 4'd6) && !dbg_b6_done_f) begin
            dbg_b6_done_f <= 1'b1;
            $display("[DBG_B6_POST] block6 done @ cycle %0d  gold_b6[0..3]=%h %h %h %h",
                     cycle_cnt,
                     DBG_GOLD_B6[0], DBG_GOLD_B6[1], DBG_GOLD_B6[2], DBG_GOLD_B6[3]);
        end

        if ((dbg_b6_st_d != 3'd3) && (u_DUT.state == 3'd3) && !dbg_norm_enter_f) begin
            dbg_norm_enter_f <= 1'b1;
            $display("[DBG_B6_NORM_ENTER] cycle=%0d  expect s2==block6 out, gold_b6[0]=%h",
                     cycle_cnt, DBG_GOLD_B6[0]);
            $display("  final_bb_gold[0..3]=%h %h %h %h",
                     GOLD_BB[0], GOLD_BB[1], GOLD_BB[2], GOLD_BB[3]);
        end
    end
end
`endif
`endif

// Narrow norm/S1 check: requires +define+USE_SRAM_BUF (hierarchical s1_q, bt_s1_addr, etc.).
`ifdef DUMP_NORM_S1_DEBUG
`ifndef USE_SRAM_BUF
initial begin
    $display("[TB] FATAL: DUMP_NORM_S1_DEBUG requires +define+USE_SRAM_BUF");
    $finish;
end
`else
`ifndef CHECK_BLOCK0_ONLY
reg [15:0] NS1_GOLD_B6 [0:TOK_TOTAL-1];
reg [5:0]  ns1_in_n;
reg [5:0]  ns1_in_mis;
reg        ns1_in_done;
reg [5:0]  ns1_out_n;
reg [5:0]  ns1_out_mis;
reg        ns1_out_done;
reg [3:0]  ns1_s1_n;
reg [2:0]  ns1_st_d;
reg [13:0] ns1_rd_tgt;
reg [5:0]  ns1_gchk_n;

wire       ns1_sout     = (u_DUT.state == 3'd4);
wire       ns1_s1_addr  = (u_DUT.state == 3'd4) && (u_DUT.out_rd_phase == 1'b0);
wire       ns1_s1_use   = (u_DUT.state == 3'd4) && (u_DUT.out_rd_phase == 1'b1);

wire       ns1_in_en =
    (u_DUT.state == 3'd3) &&
    u_DUT.bn_rp_use &&
    (u_DUT.bn_tok_cnt == 9'd0);

always @(posedge clk) begin
    if (reset) begin
        ns1_in_n      <= 6'd0;
        ns1_in_mis    <= 6'd0;
        ns1_in_done   <= 1'b0;
        ns1_out_n     <= 6'd0;
        ns1_out_mis   <= 6'd0;
        ns1_out_done  <= 1'b0;
        ns1_s1_n      <= 4'd0;
        ns1_st_d      <= 3'd0;
        ns1_rd_tgt    <= 14'd0;
        ns1_gchk_n    <= 6'd0;
    end else begin
        ns1_st_d <= u_DUT.state;

        // Enter norm
        if ((ns1_st_d != 3'd3) && (u_DUT.state == 3'd3) && !ns1_in_done) begin
            $display("\n[NS1_ENTER] cycle=%0d enter S_BACKBONE_NORM", cycle_cnt);
            $display("  gold_b6[0..3]=%h %h %h %h  gold_bb[0..3]=%h %h %h %h",
                     NS1_GOLD_B6[0], NS1_GOLD_B6[1], NS1_GOLD_B6[2], NS1_GOLD_B6[3],
                     GOLD_BB[0], GOLD_BB[1], GOLD_BB[2], GOLD_BB[3]);
        end

        // NS1_IN: token0 norm inputs (USE beat) vs block6 after_block_out golden
        if (ns1_in_en && (ns1_in_n < EMBED_DIM)) begin
            if (NS1_GOLD_B6[ns1_in_n] !== u_DUT.bn_x_mux[15:0]) begin
                ns1_in_mis <= ns1_in_mis + 6'd1;
                $display("[NS1_IN_FAIL] n=%0d feat=%0d gold_b6=%h rtl_bn_x=%h s2_q=%h phase=%0d",
                         ns1_in_n, u_DUT.bn_rp_feat, NS1_GOLD_B6[ns1_in_n],
                         u_DUT.bn_x_mux[15:0], u_DUT.s2_q, u_DUT.bn_s2_phase);
            end else if (ns1_in_n < 6'd4)
                $display("[NS1_IN_OK]   n=%0d feat=%0d data=%h",
                         ns1_in_n, u_DUT.bn_rp_feat, u_DUT.bn_x_mux[15:0]);
            ns1_in_n <= ns1_in_n + 6'd1;
            if (ns1_in_n == EMBED_DIM - 6'd1) begin
                ns1_in_done <= 1'b1;
                if (ns1_in_mis == 6'd0)
                    $display("[NS1_IN_SUM] PASS token0 norm inputs (%0d feats)", EMBED_DIM);
                else
                    $display("[NS1_IN_SUM] FAIL token0 norm inputs mismatches=%0d/%0d",
                             ns1_in_mis, EMBED_DIM);
            end
        end

        // NS1_S1_WR: first 4 latched norm writes (token0) vs golden_bb[flat]
        if (u_DUT.s1_wr_do && (u_DUT.bn_tok_cnt == 9'd0) && (ns1_s1_n < 4'd4)) begin
            if (GOLD_BB[u_DUT.s1_wr_flat_lat] !== u_DUT.s1_wr_din_lat[15:0])
                $display("[NS1_S1_WR_FAIL] n=%0d flat=%0d gold_bb=%h rtl_sat=%h feat_addr=%0d",
                         ns1_s1_n, u_DUT.s1_wr_flat_lat, GOLD_BB[u_DUT.s1_wr_flat_lat],
                         u_DUT.s1_wr_din_lat[15:0], u_DUT.bn_feat_addr);
            else
                $display("[NS1_S1_WR_OK]   n=%0d flat=%0d data=%h feat_addr=%0d",
                         ns1_s1_n, u_DUT.s1_wr_flat_lat, u_DUT.s1_wr_din_lat[15:0],
                         u_DUT.bn_feat_addr);
            ns1_s1_n <= ns1_s1_n + 4'd1;
        end

        if (ns1_s1_addr)
            ns1_rd_tgt <= u_DUT.out_rd_addr;

        // NS1_GOLDCHK posedge: USE beat y_valid vs golden[tgt] and golden[out_addr]
        if (ns1_s1_use && (ns1_gchk_n < 6'd12)) begin
            if (GOLD_BB[ns1_rd_tgt] !== u_DUT.s1_q[15:0])
                $display("[NS1_GOLDCHK_USE_POS] n=%0d tgt=%0d gold_tgt=%h s1_q=%h y_o=%h vld=%0d out_addr=%0d",
                         ns1_gchk_n, ns1_rd_tgt, GOLD_BB[ns1_rd_tgt], u_DUT.s1_q,
                         y_o[15:0], y_valid, u_DUT.out_rd_addr);
            else
                $display("[NS1_GOLDCHK_USE_POS] n=%0d tgt=%0d OK s1_q=%h y_o=%h vld=%0d",
                         ns1_gchk_n, ns1_rd_tgt, u_DUT.s1_q, y_o[15:0], y_valid);
            ns1_gchk_n <= ns1_gchk_n + 6'd1;
        end

        // NS1_OUT: first 8 y_o vs golden_bb
        if (y_valid && (ns1_out_n < 6'd8)) begin
            if (GOLD_BB[rtl_bb_cnt] !== y_o[15:0]) begin
                ns1_out_mis <= ns1_out_mis + 6'd1;
                $display("[NS1_OUT_FAIL] strm_idx=%0d gold=%h rtl_y=%h s1_q=%h tgt=%0d gold_tgt=%h out_addr=%0d ph=%0d",
                         rtl_bb_cnt, GOLD_BB[rtl_bb_cnt], y_o[15:0], u_DUT.s1_q,
                         ns1_rd_tgt, GOLD_BB[ns1_rd_tgt], u_DUT.out_rd_addr,
                         u_DUT.out_rd_phase);
                if (GOLD_BB[ns1_rd_tgt] === y_o[15:0])
                    $display("  note: rtl_y matches golden[tgt] not golden[strm_idx]");
                else if (GOLD_BB[u_DUT.out_rd_addr] === y_o[15:0])
                    $display("  note: rtl_y matches golden[out_addr]");
                else if ((ns1_rd_tgt < TOK_TOTAL) &&
                         (GOLD_BB[ns1_rd_tgt] === u_DUT.s1_q[15:0]))
                    $display("  note: s1_q matches golden[tgt] at USE posedge");
            end else
                $display("[NS1_OUT_OK]   strm_idx=%0d tgt=%0d data=%h",
                         rtl_bb_cnt, ns1_rd_tgt, y_o[15:0]);
            ns1_out_n <= ns1_out_n + 6'd1;
            if (ns1_out_n == 6'd7) begin
                ns1_out_done <= 1'b1;
                if (ns1_out_mis == 6'd0)
                    $display("[NS1_OUT_SUM] PASS first 8 y_o vs golden_bb");
                else
                    $display("[NS1_OUT_SUM] FAIL first 8 y_o mismatches=%0d/8", ns1_out_mis);
                $display("[NS1_DONE] narrow check complete, stop sim (full chain not finished)");
                $finish;
            end
        end
    end
end

always @(negedge clk) begin
    if (!reset && ns1_sout && ns1_s1_use && (ns1_gchk_n <= 6'd12)) begin
        if (GOLD_BB[ns1_rd_tgt] !== u_DUT.s1_q[15:0])
            $display("[NS1_GOLDCHK_USE_NEG] tgt=%0d gold_tgt=%h s1_q=%h out_addr=%0d s1_a=%0d",
                     ns1_rd_tgt, GOLD_BB[ns1_rd_tgt], u_DUT.s1_q,
                     u_DUT.out_rd_addr, u_DUT.bt_s1_addr);
        else
            $display("[NS1_GOLDCHK_USE_NEG] tgt=%0d OK s1_q=%h", ns1_rd_tgt, u_DUT.s1_q);
    end
end
`endif
`endif
`endif

// initial begin
//     $fsdbDumpfile("backbone_tb.fsdb");
//     $fsdbDumpvars(0, TEST_backbone.u_DUT);
//     $fsdbDumpoff;
// end

// initial begin
//     #(CYCLE * FSDB_START_MULT);
//     $fsdbDumpon;
// end

// initial begin
//     $set_toggle_region("u_DUT");
//     $toggle_start();
// end

`ifdef TB_BUF_MODE_ERROR
initial begin
    $display("[TB] FATAL: +define+USE_REG_BUF and +define+USE_SRAM_BUF are mutually exclusive.");
    $finish;
end
`endif

initial begin
    `ifdef USE_SRAM_BUF
    $display("[TB] Buffer mode: USE_SRAM_BUF (SP SRAM activation buffers)");
    `ifdef DUMP_SRAM_BUF_DEBUG
    $display("[TB] DUMP_SRAM_BUF_DEBUG on (grep DBG_STEP in log)");
    `endif
    `ifdef DUMP_BLOCK6_DEBUG
    $display("[TB] DUMP_BLOCK6_DEBUG on (grep DBG_B6 in log)");
    `endif
    `ifdef DUMP_NORM_S1_DEBUG
    $display("[TB] DUMP_NORM_S1_DEBUG on (grep NS1_ in log; stops after first 8 y_o)");
    `endif
    `elsif USE_REG_BUF
    $display("[TB] Buffer mode: USE_REG_BUF (reg activation buffers)");
    `endif

`ifndef CHECK_BLOCK0_ONLY
    $readmemb("./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt", GOLD_BB);
    `ifdef DUMP_SRAM_BUF_DEBUG
    $readmemb("./TXT_File/Activation/backbone_blocks_0_after_block_out_bi.txt", DBG_GOLD_B0);
    $readmemb("./TXT_File/Activation/backbone_blocks_1_after_block_out_bi.txt", DBG_GOLD_B1);
    $display("[TB] DUMP_SRAM_BUF_DEBUG: loaded block0/block1 after_block_out golden");
    `endif
    `ifdef DUMP_BLOCK6_DEBUG
    $readmemb("./TXT_File/Activation/backbone_blocks_5_after_block_out_bi.txt", DBG_GOLD_B5);
    $readmemb("./TXT_File/Activation/backbone_blocks_6_after_block_out_bi.txt", DBG_GOLD_B6);
    $display("[TB] DUMP_BLOCK6_DEBUG: loaded block5/block6 after_block_out golden");
    `endif
    `ifdef DUMP_NORM_S1_DEBUG
    $readmemb("./TXT_File/Activation/backbone_blocks_6_after_block_out_bi.txt", NS1_GOLD_B6);
    $display("[TB] DUMP_NORM_S1_DEBUG: loaded block6 in + backbone_after_norm golden");
    `endif
`else
    $readmemb("./TXT_File/Activation/backbone_blocks_0_after_block_out_bi.txt", GOLD_B0);
    $display("[TB] CHECK_BLOCK0_ONLY: golden backbone_blocks_0_after_block_out (%0d elems)",
             TOK_TOTAL);
`endif
    $readmemb("./TXT_File/Activation/template_post_embed_input_bi.txt", TEMPL_MEM);
    $readmemb("./TXT_File/Activation/search_post_embed_input_bi.txt", SRCH_MEM);
    $display("[TB] Loaded inputs from ./TXT_File/Activation/  template(%0d) search(%0d)",
             TEMPL_TOT, SRCH_TOT);

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 0;
    tok_cnt    = 0;
`ifndef CHECK_BLOCK0_ONLY
    rtl_bb_cnt = 0;
    bb_mism    = 0;
    bb_first_bad = 32'hFFFF_FFFF;
`else
    rtl_b0_cnt = 0;
    b0_mism    = 0;
    b0_first_bad = 32'hFFFF_FFFF;
`endif

    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    `ifdef CHECK_BLOCK0_ONLY
    #(CYCLE * 25_000_000);
    `elsif USE_SRAM_BUF
    #(CYCLE * 500_000_000);
    `else
    #(CYCLE * 80_000_000);
    `endif
    $display("[TB] TIMEOUT: backbone_top did not finish (see USE_SRAM_BUF for longer limit)");
    $toggle_stop();
    $toggle_report("backbone_only_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
