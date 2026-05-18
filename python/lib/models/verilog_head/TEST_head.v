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
//      → head 僅將 search 段（index 2048..10239）寫入 opt_buf
//   4. DUT done → 輸出 bbox，與 golden 比對後結束仿真
//
// Golden（相對 simv CWD 的 ./TXT_File/Activation/）：
//   輸入：backbone_after_norm_backbone_out_bi.txt
//   bbox：box_head_after_cal_bbox_bbox_bi.txt（cx,cy,w,h 各一筆 Q8.8）
//
// VCS 編譯：見 compile_head.sh（需外部 memory/ 內 SRAM + box_head ROM macro）
// =============================================================================

module TEST_head;

parameter CYCLE = 2.0;   // 2 ns → 500 MHz

parameter [31:0] FSDB_START_MULT = 32'd5_000_000;

parameter EMBED_DIM   = 32;
parameter FEAT_H      = 16;
parameter FEAT_W      = 16;
parameter LENS_Z      = 64;
parameter N_TOKENS    = 320;
parameter TOT_VALS    = N_TOKENS * EMBED_DIM;   // 10240
parameter SKIP_VALS   = LENS_Z * EMBED_DIM;     // 2048

// Golden bbox（box_head_after_cal_bbox_bbox_bi.txt，Q8.8）
localparam [15:0] GOLDEN_CX = 16'sh007F;
localparam [15:0] GOLDEN_CY = 16'sh007E;
localparam [15:0] GOLDEN_W  = 16'sh0023;
localparam [15:0] GOLDEN_H  = 16'sh0054;
localparam BBOX_TOL_LSB = 2;

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
        $display("\n  Predicted bbox (Q8.8 hex | float/256):");
        $display("    cx = 0x%04h  (%f)", cx_out, $itor($signed(cx_out)) / 256.0);
        $display("    cy = 0x%04h  (%f)", cy_out, $itor($signed(cy_out)) / 256.0);
        $display("    w  = 0x%04h  (%f)", w_out,  $itor($signed(w_out))  / 256.0);
        $display("    h  = 0x%04h  (%f)", h_out,  $itor($signed(h_out))  / 256.0);
        $display("\n  Golden bbox  (box_head_after_cal_bbox_bbox_bi.txt):");
        $display("    cx = 0x%04h  (%f)", GOLDEN_CX, $itor($signed(GOLDEN_CX)) / 256.0);
        $display("    cy = 0x%04h  (%f)", GOLDEN_CY, $itor($signed(GOLDEN_CY)) / 256.0);
        $display("    w  = 0x%04h  (%f)", GOLDEN_W,  $itor($signed(GOLDEN_W))  / 256.0);
        $display("    h  = 0x%04h  (%f)", GOLDEN_H,  $itor($signed(GOLDEN_H))  / 256.0);
        if (($signed(cx_out) - $signed(GOLDEN_CX)) <= BBOX_TOL_LSB &&
            ($signed(GOLDEN_CX) - $signed(cx_out)) <= BBOX_TOL_LSB &&
            ($signed(cy_out) - $signed(GOLDEN_CY)) <= BBOX_TOL_LSB &&
            ($signed(GOLDEN_CY) - $signed(cy_out)) <= BBOX_TOL_LSB &&
            ($signed(w_out)  - $signed(GOLDEN_W))  <= BBOX_TOL_LSB &&
            ($signed(GOLDEN_W)  - $signed(w_out))  <= BBOX_TOL_LSB &&
            ($signed(h_out)  - $signed(GOLDEN_H))  <= BBOX_TOL_LSB &&
            ($signed(GOLDEN_H)  - $signed(h_out))  <= BBOX_TOL_LSB)
            $display("\n  [PASS] bbox matches golden within +-%0d LSB", BBOX_TOL_LSB);
        else
            $display("\n  [FAIL] bbox differs from golden (+- %0d LSB)", BBOX_TOL_LSB);
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

    clk        = 0;
    reset      = 1;
    start      = 0;
    data_in    = 16'sd0;
    data_valid = 1'b0;
    cycle_cnt  = 32'd0;
    tok_cnt    = 14'd0;

    #(CYCLE) reset = 1;
    #(CYCLE) reset = 0;

    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    // head_top_beh 全速 MAC 約 18M cycles；SRAM head_top 約 35M+。請勿低於 25M。
`ifdef USE_BEH_HEAD
    #(CYCLE * 25_000_000);
    $display("[TB] TIMEOUT: head_top_beh did not complete within 25M cycles");
`else
    #(CYCLE * 25_000_000);
    $display("[TB] TIMEOUT: head_top did not complete within 25M cycles");
`endif
    $finish;
end

endmodule
