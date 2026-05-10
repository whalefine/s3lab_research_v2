`timescale 1ns/10ps

// =============================================================================
// TEST.v — SGLATrack shared-trunk testbench（SRAM/ROM macro 版本）
//
// 參照 reference/rongxuan_verilog/TEST.v 撰寫。
//
// DUT：sglatrack_top（SRAM/ROM macro 版本，無外部 weight ROM 介面）
//
// 測試流程：
//   1. $readmemb 分別載入 template / search post-embedding tokens
//   2. 重設 DUT；設定 sel_block_i = 6
//   3. 先串流 TEMPL_MEM（64×32=2048 筆），再串流 SRCH_MEM（256×32=8192 筆）
//      → 對應 run_backbone_numpy_shared_trunk.py：np.concatenate([z, x])
//   4. DUT done → 顯示 bbox，與 golden 比對
//
// NOTE：weight ROM 已由 backbone_top / head_top 內部 ROM macro 提供，
//       testbench 不需載入或傳遞任何 weight 資料。
//
// Golden bbox（Q8.8）：
//   cx = 0x007F  (127/256 ≈ 0.496)
//   cy = 0x007E  (126/256 ≈ 0.492)
//   w  = 0x0023  ( 35/256 ≈ 0.137)
//   h  = 0x0054  ( 84/256 ≈ 0.328)
// =============================================================================

module TEST;

// ---------------------------------------------------------------------------
// Clock parameter
// ---------------------------------------------------------------------------
parameter CYCLE = 2.0;   // 2 ns → 500 MHz

// ---------------------------------------------------------------------------
// Model constants (must match sglatrack_top parameters)
// ---------------------------------------------------------------------------
parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;                      // template tokens
parameter FEAT_SZ     = FEAT_H * FEAT_W;         // 256 search tokens
parameter TEMPL_TOT   = LENS_Z  * EMBED_DIM;     // 2048
parameter SRCH_TOT    = FEAT_SZ * EMBED_DIM;     // 8192
parameter TOK_TOTAL   = TEMPL_TOT + SRCH_TOT;    // 10240 (= N_TOKENS * EMBED_DIM)
parameter N_TOKENS    = TOK_TOTAL / EMBED_DIM;   // 320

// NOTE: BANK*_SZ and IMG_PIXELS parameters removed — weights/image no longer needed.

// ---------------------------------------------------------------------------
// DUT signals
// ---------------------------------------------------------------------------
reg         clk, reset, start;
reg  [3:0]  sel_block_i;

reg  signed [15:0] data_in;
reg                data_valid;

// NOTE: weight ROM ports removed — backbone_top and head_top now have
// internal ROM macros. No external weight wiring needed.

wire [15:0] cx_out, cy_out, w_out, h_out;
wire busy;
wire done;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Clock generation
// ---------------------------------------------------------------------------
always #(CYCLE/2.0) clk = ~clk;

// ---------------------------------------------------------------------------
// Cycle counter (for performance measurement)
// ---------------------------------------------------------------------------
reg [31:0] cycle_cnt;
always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

// ---------------------------------------------------------------------------
// Input data memories
// ---------------------------------------------------------------------------
// Template post-embedding tokens: LENS_Z×EMBED_DIM = 2048 × 16-bit Q8.8
reg [15:0] TEMPL_MEM [0:TEMPL_TOT-1];

// Search post-embedding tokens: FEAT_SZ×EMBED_DIM = 8192 × 16-bit Q8.8
reg [15:0] SRCH_MEM  [0:SRCH_TOT-1];

// Token streaming counter
reg [13:0] tok_cnt;

// NOTE: IMG_MEM, TOK_MEM, and BANK0-4 weight arrays removed.

// ---------------------------------------------------------------------------
// Token streaming (negedge driven, aligned to TEST.v style)
// Template tokens first (tok_cnt < TEMPL_TOT), then search tokens.
// Mirrors run_backbone_numpy_shared_trunk.py: np.concatenate([z, x])
// ---------------------------------------------------------------------------
always @(negedge clk) begin
    if (reset) begin
        tok_cnt    <= 14'd0;
        data_in    <= 16'sd0;
        data_valid <= 1'b0;
    end else if (start || busy) begin
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
    end
end

// ---------------------------------------------------------------------------
// Result check on done
// ---------------------------------------------------------------------------
always @(negedge clk) begin
    if (done) begin
        $display("\n---- SGLATrack done @ cycle %0d ----", cycle_cnt);
        $display("  Tokens: template=%0d, search=%0d, total=%0d", TEMPL_TOT, SRCH_TOT, TOK_TOTAL);
        $display("  sel_block_i = %0d", sel_block_i);
        $display("\n  Predicted bbox (Q8.8 hex | float):");
        $display("    cx = 0x%04h  (%f)", cx_out, cx_out / 256.0);
        $display("    cy = 0x%04h  (%f)", cy_out, cy_out / 256.0);
        $display("    w  = 0x%04h  (%f)", w_out,  w_out  / 256.0);
        $display("    h  = 0x%04h  (%f)", h_out,  h_out  / 256.0);

        $display("\n  Golden bbox  (Q8.8 hex | float):");
        $display("    cx = 0x007F  (0.496094)");
        $display("    cy = 0x007E  (0.492188)");
        $display("    w  = 0x0023  (0.136719)");
        $display("    h  = 0x0054  (0.328125)");

        // Tolerance check (±2 LSB in Q8.8)
        if ($signed({1'b0, cx_out}) - $signed(16'h007F) <= 2 &&
            $signed(16'h007F) - $signed({1'b0, cx_out}) <= 2 &&
            $signed({1'b0, cy_out}) - $signed(16'h007E) <= 2 &&
            $signed(16'h007E) - $signed({1'b0, cy_out}) <= 2) begin
            $display("\n  [PASS] bbox matches golden within +-2 LSB");
        end else begin
            $display("\n  [FAIL] bbox differs from golden");
        end

        $toggle_stop();
        $toggle_report("sglatrack_rtl.saif", 1.0e-9, "u_DUT");
        $finish;
    end
end

// ---------------------------------------------------------------------------
// Main initial block
// ---------------------------------------------------------------------------
initial begin
    $fsdbDumpfile("sglatrack_tb.fsdb");
    $fsdbDumpvars(0, TEST);
    $fsdbDumpMDA;

    $set_toggle_region("u_DUT");
    $toggle_start();

    // Load template post-embedding tokens (64 tokens × 32 dims = 2048 values)
    $readmemb("python/output/golden/vit_care_relu6_numpy_trunk_dim32_out/Activation/template_post_embed_input_bi.txt",
              TEMPL_MEM);
    $display("[TB] Loaded template tokens: %0d values", TEMPL_TOT);

    // Load search post-embedding tokens (256 tokens × 32 dims = 8192 values)
    $readmemb("python/output/golden/vit_care_relu6_numpy_trunk_dim32_out/Activation/search_post_embed_input_bi.txt",
              SRCH_MEM);
    $display("[TB] Loaded search tokens: %0d values", SRCH_TOT);

    // NOTE: weight banks no longer loaded here — ROM macros are internal to DUT.
    $display("[TB] Weight ROMs internal to DUT backbone_top/head_top.");

    // ---------------------------------------------------------------------------
    // DUT reset sequence (matching TEST.v style)
    // ---------------------------------------------------------------------------
    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 32'd0;
    tok_cnt    = 14'd0;

    // sel_block_i = 6 (from golden_manifest: adaptive_selected_layer_index = 6)
    sel_block_i = 4'd6;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    $display("\n[TB] Reset complete. sel_block_i=%0d, starting inference...", sel_block_i);

    // Assert start for one cycle
    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    // Timeout watchdog (10M cycles)
    #(CYCLE * 10_000_000);
    $display("[TB] TIMEOUT: DUT did not complete within 10M cycles");
    $toggle_stop();
    $toggle_report("sglatrack_rtl.saif", 1.0e-9, "u_DUT");
    $finish;
end

endmodule
