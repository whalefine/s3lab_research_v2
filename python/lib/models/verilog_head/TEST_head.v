`timescale 1ns/10ps

// =============================================================================
// TEST_head.v — Head-only testbench（假設 backbone 輸出正確）
//
// DUT：head_top（shared conv → tail → cal_bbox）
//
// 測試流程：
//   1. $readmemb 載入 backbone norm 後的 token 串流（與 backbone_top y_o 相同順序）
//   2. 重設 DUT；assert start
//   3. 在 S_FILL 期間串流 10240 筆 Q8.8（64 template + 256 search tokens × 32 dim）
//      → Golden: backbone_after_norm_backbone_out_bi.txt
//      → head 僅將 search 段（index 2048..10239）寫入 opt_buf
//   4. DUT done → 與 box_head_after_cal_bbox_bbox_bi.txt 比對
//
// Golden 目錄（相對 simv CWD 的 ./TXT_File/Activation/）：
//   python/output/golden/vit_care_relu6_numpy_trunk_dim32_out/Activation/
//
// 執行前請在 verilog_head 目錄執行：  ./setup_txt_file.sh
// 或手動建立 TXT_File/Activation → golden Activation 的 symlink。
//
// VCS 編譯：見 compile_head.sh（需外部 memory/ 內 SRAM + box_head ROM macro）
//
// Golden bbox（Q8.8，box_head_after_cal_bbox_bbox_bi.txt）：
//   cx = 0x007F  (127/256 ≈ 0.496)
//   cy = 0x007E  (126/256 ≈ 0.492)
//   w  = 0x0023  ( 35/256 ≈ 0.137)
//   h  = 0x0054  ( 84/256 ≈ 0.328)
//
// Simulation 注意：
//   - $readmemb 路徑相對於執行 simv 時的 CWD。
//   - 僅在 u_DUT.state==S_FILL 時推進串流計數，避免 fill_cnt 超前。
//   +define+DUMP_TB_ARGMAX
//       比對 cal_bbox argmax_idx_snap 與 golden score_map；並印錨點 idx 3/135/136 的 golden 值。
//   +define+DUMP_TB_SCORE_DEEP
//       載入 conv2 / tail_ctr raw golden；結束時印錨點表；initial 抽樣 backbone 輸入。
//   +define+DUMP_HEAD_SCORE_CHECK（head_top.v）
//       S_CONV2 僅 oc=0/47；S_CTR/S_BBOX 錨點 idx 3/135/136。
//   +define+DUMP_HEAD_SCORE_DEEP（head_top.v，建議與上併用）
//       錨點即時 [HEAD_GOLDEN_CMP] PASS/FAIL（±2 LSB）。
//
// 建議編譯 define 組合：
//   +define+DUMP_CAL_BBOX +define+DUMP_CAL_BBOX_ARGMAX +define+DUMP_TB_ARGMAX +define+DUMP_TB_SCORE_DEEP
//   +define+DUMP_HEAD_SCORE_CHECK +define+DUMP_HEAD_SCORE_DEEP +define+DUMP_HEAD_FEAT_TAP
//   cx/cy=0xf800（-8）時：先確認 S_CONV1/2 [HEAD_GOLDEN_CMP] PASS，再看 [CAL_BBOX] 的 sum_cx/sum_cy
// =============================================================================

module TEST_head;

parameter CYCLE = 2.0;   // 2 ns → 500 MHz

parameter [31:0] FSDB_START_MULT = 32'd5_000_000;

parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;
parameter N_TOKENS    = 320;
parameter FEAT_SZ     = FEAT_H * FEAT_W;
parameter TOT_VALS    = N_TOKENS * EMBED_DIM;   // 10240
parameter SCORE_LEN   = FEAT_SZ;
parameter C_SH2       = 48;
parameter CONV2_LEN   = C_SH2 * FEAT_SZ;   // 12288
parameter SKIP_VALS   = LENS_Z * EMBED_DIM; // 2048

`ifdef DUMP_TB_ARGMAX
reg [15:0] SCORE_GOLDEN [0:SCORE_LEN-1];
`endif

`ifdef DUMP_TB_SCORE_DEEP
parameter OPT_LEN   = EMBED_DIM * FEAT_SZ;
parameter CONV1_LEN = 96 * FEAT_SZ;
reg [15:0] OPT_GOLDEN [0:OPT_LEN-1];
reg [15:0] CONV1_GOLDEN [0:CONV1_LEN-1];
reg [15:0] CONV2_GOLDEN [0:CONV2_LEN-1];
reg [15:0] RAW_CTR_GOLDEN [0:SCORE_LEN-1];
`endif

reg         clk, reset, start;
reg  signed [15:0] data_in;
reg                data_valid;

wire [15:0] cx_out, cy_out, w_out, h_out;
wire        busy;
wire        done;

// 預設用 SRAM 版 head_top；+define+USE_BEH_HEAD 改用 head_top_beh（無 SRAM/ROM 行為層）
`ifdef USE_BEH_HEAD
head_top_beh #(
    .IN_CH    (EMBED_DIM),
    .FEAT_H   (FEAT_H),
    .FEAT_W   (FEAT_W),
    .N_TOKENS (N_TOKENS),
    .LENS_Z   (LENS_Z)
) u_DUT (
    .clk      (clk),
    .reset    (reset),
    .start    (start),
    .a_i      (data_in),
    .a_valid  (data_valid),
    .busy     (busy),
    .done     (done),
    .cx_o     (cx_out),
    .cy_o     (cy_out),
    .w_o      (w_out),
    .h_o      (h_out)
);
`else
head_top #(
    .IN_CH    (EMBED_DIM),
    .FEAT_H   (FEAT_H),
    .FEAT_W   (FEAT_W),
    .N_TOKENS (N_TOKENS),
    .LENS_Z   (LENS_Z)
) u_DUT (
    .clk      (clk),
    .reset    (reset),
    .start    (start),
    .a_i      (data_in),
    .a_valid  (data_valid),
    .busy     (busy),
    .done     (done),
    .cx_o     (cx_out),
    .cy_o     (cy_out),
    .w_o      (w_out),
    .h_o      (h_out)
);
`endif

// head_top S_FILL = 4'd1
wire tb_stream_gate = (u_DUT.state == 4'd1);

always #(CYCLE/2.0) clk = ~clk;

reg [31:0] cycle_cnt;
always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

// backbone norm 輸出串流（320 token × 32 dim，C-order flatten）
reg [15:0] BACKBONE_MEM [0:TOT_VALS-1];

reg [13:0] tok_cnt;

always @(posedge clk) begin
    if (reset) begin
        tok_cnt    <= 14'd0;
        data_in    <= 16'sd0;
        data_valid <= 1'b0;
    end else if ((start || busy) && tb_stream_gate) begin
        if (tok_cnt < TOT_VALS) begin
            data_valid <= 1'b1;
            data_in    <= BACKBONE_MEM[tok_cnt];
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
        $display("\n---- Head-only done @ cycle %0d ----", cycle_cnt);
        $display("  Input: backbone_after_norm stream, %0d values", TOT_VALS);
        $display("\n  Predicted bbox (Q8.8 hex | signed/256):");
        $display("    cx = 0x%04h  (%f)", cx_out, $itor($signed(cx_out)) / 256.0);
        $display("    cy = 0x%04h  (%f)", cy_out, $itor($signed(cy_out)) / 256.0);
        $display("    w  = 0x%04h  (%f)", w_out,  $itor($signed(w_out))  / 256.0);
        $display("    h  = 0x%04h  (%f)", h_out,  $itor($signed(h_out))  / 256.0);

        $display("\n  Golden bbox  (box_head_after_cal_bbox_bbox_bi.txt):");
        $display("    cx = 0x007F  (0.496094)");
        $display("    cy = 0x007E  (0.496094)");
        $display("    w  = 0x0023  (0.140625)");
        $display("    h  = 0x0054  (0.335938)");

`ifdef DUMP_TB_ARGMAX
        begin : argmax_golden_check
            integer           gi;
            reg signed [15:0] g_max;
            reg [7:0]         g_idx;
            reg [7:0]         rtl_idx;
            reg [3:0]         g_ix, g_iy, r_ix, r_iy;

            g_max  = 16'sh8000;
            g_idx  = 8'd0;
            for (gi = 0; gi < SCORE_LEN; gi = gi + 1) begin
                if ($signed(SCORE_GOLDEN[gi]) > g_max) begin
                    g_max = SCORE_GOLDEN[gi];
                    g_idx = gi[7:0];
                end
            end
            rtl_idx = u_DUT.u_bbox.argmax_idx_snap;
            g_ix    = g_idx[3:0];
            g_iy    = g_idx[7:4];
            r_ix    = rtl_idx[3:0];
            r_iy    = rtl_idx[7:4];

            $display("\n  ---- Argmax (score map) ----");
            $display("  numpy golden flat idx = %0d  (idx_x=%0d idx_y=%0d) max_sc=%h",
                     g_idx, g_ix, g_iy, g_max);
            $display("  RTL cal_bbox snap idx   = %0d  (idx_x=%0d idx_y=%0d) max_sc=%h",
                     rtl_idx, r_ix, r_iy, u_DUT.u_bbox.max_score_snap);
            if (rtl_idx == g_idx)
                $display("  [PASS] argmax index matches golden");
            else
                $display("  [FAIL] argmax index mismatch (RTL %0d vs golden %0d)",
                         rtl_idx, g_idx);

            $display("\n  ---- Score map anchor cells (golden *_bi.txt) ----");
            $display("  flat idx=3   (oh=0,ow=3):   golden_sc=%h", SCORE_GOLDEN[3]);
            $display("  flat idx=135 (oh=7,ow=8):   golden_sc=%h  (RTL argmax common)",
                     SCORE_GOLDEN[135]);
            $display("  flat idx=136 (oh=8,ow=8):   golden_sc=%h  (numpy argmax)",
                     SCORE_GOLDEN[136]);
            $display("  對照 log 中 [HEAD_SCORE_CHK] S_CTR wr / S_BBOX sc_q 同 idx 的 RTL 值");
        end
`endif

`ifdef DUMP_TB_SCORE_DEEP
        begin : score_deep_golden_summary
            integer ai;
            reg [7:0] anchor_idx [0:2];
            reg [5:0] anchor_oc [0:1];

            anchor_idx[0] = 8'd3;
            anchor_idx[1] = 8'd135;
            anchor_idx[2] = 8'd136;
            anchor_oc[0]  = 6'd0;
            anchor_oc[1]  = 6'd47;

            $display("\n  ---- Pre-sigmoid / conv golden anchors (TB reference) ----");
            $display("  head_input (box_head_head_input_bi.txt) ch0 @ idx:");
            for (ai = 0; ai < 3; ai = ai + 1)
                $display("    idx=%0d golden_opt_ch0=%h",
                         anchor_idx[ai], OPT_GOLDEN[anchor_idx[ai]]);
            $display("  conv1 oc=0:");
            for (ai = 0; ai < 3; ai = ai + 1)
                $display("    idx=%0d golden=%h",
                         anchor_idx[ai], CONV1_GOLDEN[anchor_idx[ai]]);
            $display("  tail_ctr raw (box_head_tail_ctr_after_conv_out_bi.txt):");
            for (ai = 0; ai < 3; ai = ai + 1)
                $display("    idx=%0d (oh=%0d,ow=%0d) golden_raw=%h",
                         anchor_idx[ai], anchor_idx[ai][7:4], anchor_idx[ai][3:0],
                         RAW_CTR_GOLDEN[anchor_idx[ai]]);

            $display("  conv2 ReLU (box_head_shared_after_conv2_out_bi.txt), oc=0 and oc=47:");
            for (ai = 0; ai < 3; ai = ai + 1) begin
                $display("    idx=%0d oc=0  golden=%h",
                         anchor_idx[ai],
                         CONV2_GOLDEN[{anchor_oc[0], anchor_idx[ai]}]);
                $display("    idx=%0d oc=47 golden=%h",
                         anchor_idx[ai],
                         CONV2_GOLDEN[{anchor_oc[1], anchor_idx[ai]}]);
            end
            $display("  對照 log: [HEAD_GOLDEN_CMP] S_CONV2 / S_CTR raw（即時 PASS/FAIL）");
        end
`endif

        if (($signed(cx_out) - $signed(16'sh007F)) <= 2 &&
            ($signed(16'sh007F) - $signed(cx_out)) <= 2 &&
            ($signed(cy_out) - $signed(16'sh007E)) <= 2 &&
            ($signed(16'sh007E) - $signed(cy_out)) <= 2 &&
            ($signed(w_out)  - $signed(16'sh0023)) <= 2 &&
            ($signed(16'sh0023) - $signed(w_out))  <= 2 &&
            ($signed(h_out)  - $signed(16'sh0054)) <= 2 &&
            ($signed(16'sh0054) - $signed(h_out))  <= 2) begin
            $display("\n  [PASS] bbox matches golden within +-2 LSB");
        end else begin
            $display("\n  [FAIL] bbox differs from golden (+-2 LSB)");
        end

        $finish;
    end
end

initial begin
    $fsdbDumpfile("head_tb.fsdb");
    $fsdbDumpvars(1, TEST_head.u_DUT);
    $fsdbDumpvars(0, TEST_head.u_DUT.u_bbox);
    $fsdbDumpoff;
end

initial begin
    #(CYCLE * FSDB_START_MULT);
    $fsdbDumpon;
end

initial begin
    $readmemb("./TXT_File/Activation/backbone_after_norm_backbone_out_bi.txt",
              BACKBONE_MEM);
    $display("[TB] Loaded backbone norm output: %0d values", TOT_VALS);
    $display("[TB]   Golden: backbone_after_norm_backbone_out_bi.txt");

`ifdef DUMP_TB_ARGMAX
    $readmemb("./TXT_File/Activation/box_head_after_forward_head_score_map_bi.txt",
              SCORE_GOLDEN);
    $display("[TB] Loaded golden score_map: %0d values", SCORE_LEN);
`endif

`ifdef DUMP_TB_SCORE_DEEP
    $readmemb("./TXT_File/Activation/box_head_head_input_bi.txt",
              OPT_GOLDEN);
    $readmemb("./TXT_File/Activation/box_head_shared_after_conv1_out_bi.txt",
              CONV1_GOLDEN);
    $readmemb("./TXT_File/Activation/box_head_shared_after_conv2_out_bi.txt",
              CONV2_GOLDEN);
    $readmemb("./TXT_File/Activation/box_head_tail_ctr_after_conv_out_bi.txt",
              RAW_CTR_GOLDEN);
    $display("[TB] Loaded golden opt(%0d) conv1(%0d) conv2(%0d) raw_ctr(%0d)",
             OPT_LEN, CONV1_LEN, CONV2_LEN, SCORE_LEN);
    $display("[TB] Backbone spot-check (search token0, flat base=%0d):", SKIP_VALS);
    $display("  BACKBONE_MEM[%0d..%3d] = %h %h %h %h",
             SKIP_VALS, SKIP_VALS + 3,
             BACKBONE_MEM[SKIP_VALS], BACKBONE_MEM[SKIP_VALS + 1],
             BACKBONE_MEM[SKIP_VALS + 2], BACKBONE_MEM[SKIP_VALS + 3]);
`endif

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 32'd0;
    tok_cnt    = 14'd0;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    $display("\n[TB] Reset complete. Starting head-only inference...");

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    // head_top_beh 全速 MAC 約 18M cycles；SRAM head_top 約 35M+。請勿低於 25M。
`ifdef USE_BEH_HEAD
    #(CYCLE * 25_000_000);
    $display("[TB] TIMEOUT: head_top_beh did not complete within 25M cycles (est ~18M)");
`else
    #(CYCLE * 25_000_000);
    $display("[TB] TIMEOUT: head_top did not complete within 25M cycles");
`endif
    $finish;
end

endmodule
