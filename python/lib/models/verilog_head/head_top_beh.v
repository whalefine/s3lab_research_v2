// =============================================================================
// head_top_beh.v — Behavioral / sim-only head（無 SRAM／無 ROM）
//
// 用途：純模擬除錯版本，把 head_top.v 內所有 SRAM macro / ROM macro 改成
//      `reg [15:0] xxx_mem [0:N-1]`，用 $readmemb 在 initial 階段載入權重，
//      去掉所有 CLK=~clk 的 negedge 寫入時序、mac_dv pipeline、bias_q_lat 等。
//      適合用來確認 FSM、MAC、ReLU、sigmoid、cal_bbox 計算路徑，不可上板。
//
// 介面：與 head_top.v 完全相同。TB 將原 `head_top` 換成 `head_top_beh` 即可。
//
// 對齊：python/tracking/run_backbone_numpy_shared_trunk.py head_shared_trunk()
//   - opt_feat[32,16,16] → shared_conv1(3×3,32→96,ReLU)
//                        → shared_conv2(3×3,96→48,ReLU)
//                        → tail_ctr(1×1,48→1, sigmoid)
//                        → tail_size(1×1,48→2, sigmoid)
//                        → tail_offset(1×1,48→2, raw Q8.8)
//                        → cal_bbox
//   - conv2d 採 acc>>>8 + bias，再 saturate Q8.8、ReLU/Sigmoid
//
// 權重來源（相對 simv CWD）：
//   ./TXT_File/Weight/box_head_shared_conv1_folded_weight_bi.txt   (96×32×3×3 = 27648)
//   ./TXT_File/Weight/box_head_shared_conv1_folded_bias_bi.txt     (96)
//   ./TXT_File/Weight/box_head_shared_conv2_folded_weight_bi.txt   (48×96×3×3 = 41472)
//   ./TXT_File/Weight/box_head_shared_conv2_folded_bias_bi.txt     (48)
//   ./TXT_File/Weight/box_head_tail_ctr_weight_bi.txt              (48)
//   ./TXT_File/Weight/box_head_tail_ctr_bias_bi.txt                (1)
//   ./TXT_File/Weight/box_head_tail_size_weight_bi.txt             (2×48 = 96)
//   ./TXT_File/Weight/box_head_tail_size_bias_bi.txt               (2)
//   ./TXT_File/Weight/box_head_tail_offset_weight_bi.txt           (2×48 = 96)
//   ./TXT_File/Weight/box_head_tail_offset_bias_bi.txt             (2)
//   flatten 順序為 numpy C-order（與 *_bi.txt 一致）：
//     conv1_w[oc*32*9 + ic*9 + kh*3 + kw]
//     conv2_w[oc*96*9 + ic*9 + kh*3 + kw]
//     tail_*_w[oc*48 + ic]
//
// Simulation debug：
//   +define+DUMP_BEH_GOLDEN       對拍 head_input / conv1 / conv2 / raw_ctr 錨點
//   +define+DUMP_BEH_TAP          第一個 MAC tap 印 feat/rom/prod
//   +define+DUMP_BEH_FSM          每 1M cycles 印 state（查卡住位置）
//   +define+DUMP_BEH_BBOX_DEBUG   S_OFF→S_BBOX 印 score/size/off 錨點與 RTL argmax
//   +define+DUMP_BEH_TAIL_WR      S_SIZE/S_OFF 寫入時印 sz_wa/of_wa/rom/mac（查 ch1=X）
//   +define+DUMP_BEH_STAGE_CMP    conv1/conv2/sh2 與 golden 錨點比對；S_CTR 累加 partial acc
//   +define+DUMP_BEH_CONV1_PATCH  S_CONV1 idx120/136 印 3x3 opt 與 mac_acc（查 conv1 偏差）
//   +define+DUMP_CAL_BBOX         cal_bbox S_CALC 印 s0/s1/o0/o1/sum_cx/cy
//   TB：+define+USE_BEH_HEAD，timeout 建議 >= 25M cycles
// =============================================================================

`timescale 1ns/10ps

`ifndef HEAD_GOLDEN_TOL_LSB
`define HEAD_GOLDEN_TOL_LSB 2
`endif

`ifndef BEH_WEIGHT_DIR
`define BEH_WEIGHT_DIR "./TXT_File/Weight"
`endif

`ifndef BEH_ACT_DIR
`define BEH_ACT_DIR "./TXT_File/Activation"
`endif

module head_top_beh #(
    parameter IN_CH    = 32,
    parameter C_SH1    = 96,
    parameter C_SH2    = 48,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter N_TOKENS = 320,
    parameter LENS_Z   = 64
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    input  wire signed [15:0] a_i,
    input  wire               a_valid,

    output wire        busy,
    output reg         done,

    output wire [15:0] cx_o,
    output wire [15:0] cy_o,
    output wire [15:0] w_o,
    output wire [15:0] h_o
);

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------
localparam FEAT_SZ    = FEAT_H * FEAT_W;                 // 256
localparam SKIP_VALS  = LENS_Z * IN_CH;                  // 2048（template 段，不寫 opt）
localparam TOT_VALS     = N_TOKENS * IN_CH;              // 10240
localparam TOT_VALS_M1  = TOT_VALS - 1;                  // 10239
localparam IN_CH_M1   = IN_CH  - 1;                      // 31
localparam C_SH1_M1   = C_SH1  - 1;                      // 95
localparam C_SH2_M1   = C_SH2  - 1;                      // 47
localparam FEAT_H_M1  = FEAT_H - 1;                      // 15
localparam FEAT_W_M1  = FEAT_W - 1;                      // 15

localparam OPT_LEN    = IN_CH * FEAT_SZ;                 //  8192
localparam SH1_LEN    = C_SH1 * FEAT_SZ;                 // 24576
localparam SH2_LEN    = C_SH2 * FEAT_SZ;                 // 12288
localparam CONV1_W_N  = C_SH1 * IN_CH * 9;               // 27648
localparam CONV2_W_N  = C_SH2 * C_SH1 * 9;               // 41472
localparam TAIL_W_CH  = C_SH2;                           // 48

// cal_bbox stream lengths
localparam BBOX_N_SCORE      = FEAT_SZ;                  // 256
localparam BBOX_N_SIZE       = 2 * FEAT_SZ;              // 512
localparam BBOX_N_OFF        = 2 * FEAT_SZ;              // 512
localparam BBOX_STREAM_LAST  = BBOX_N_SCORE + BBOX_N_SIZE + BBOX_N_OFF - 1; // 1279

// 粗估 cycle（1 MAC/tap，無 mac_dv）：FILL 10k + CONV1 ~7.1M + CONV2 ~10.6M + tail ~65k + BBOX ~1.3k ≈ 18M
// TB 請設 >= 25M cycles（+define+USE_BEH_HEAD）

// ---------------------------------------------------------------------------
// FSM states (對齊 head_top.v 的 4-bit 編碼；TEST_head.v 以 state==4'd1 推 stream)
// ---------------------------------------------------------------------------
parameter S_IDLE  = 4'd0;
parameter S_FILL  = 4'd1;
parameter S_CONV1 = 4'd2;
parameter S_CONV2 = 4'd3;
parameter S_CTR   = 4'd4;
parameter S_SIZE  = 4'd5;
parameter S_OFF   = 4'd6;
parameter S_BBOX  = 4'd7;
parameter S_DONE  = 4'd8;

reg [3:0] state;

// ---------------------------------------------------------------------------
// Storage（行為層 reg array；模擬即可，不可合成成 RAM 巨集）
// ---------------------------------------------------------------------------
reg signed [15:0] opt_mem   [0:OPT_LEN-1];
reg signed [15:0] sh1_mem   [0:SH1_LEN-1];
reg signed [15:0] sh2_mem   [0:SH2_LEN-1];
// Golden: box_head_after_forward_head_score_map_bi.txt — flatten oh*16+ow
reg signed [15:0] score_mem [0:FEAT_SZ-1];
// Golden: box_head_after_forward_head_size_map_bi.txt — ch*256 + oh*16+ow
reg signed [15:0] size_mem  [0:2*FEAT_SZ-1];
// Golden: box_head_after_forward_head_offset_map_bi.txt — ch*256 + oh*16+ow
reg signed [15:0] off_mem   [0:2*FEAT_SZ-1];

`ifdef DUMP_BEH_BBOX_DEBUG
// golden 對照（僅模擬；須在 module 內，Verilog-2001）
reg [15:0] dbg_score_golden [0:FEAT_SZ-1];
reg [15:0] dbg_size_golden  [0:2*FEAT_SZ-1];
reg [15:0] dbg_off_golden   [0:2*FEAT_SZ-1];
reg        dbg_tail_dumped;
integer    dbg_gi;
integer    dbg_ai;
reg signed [15:0] dbg_rtl_max;
reg [7:0]         dbg_rtl_argmax;
reg [7:0]         dbg_g_argmax;
reg signed [15:0] dbg_g_max;
reg [7:0]         dbg_anchor_idx;

initial begin
    $readmemb({`BEH_ACT_DIR, "/box_head_after_forward_head_score_map_bi.txt"},
              dbg_score_golden);
    $readmemb({`BEH_ACT_DIR, "/box_head_after_forward_head_size_map_bi.txt"},
              dbg_size_golden);
    $readmemb({`BEH_ACT_DIR, "/box_head_after_forward_head_offset_map_bi.txt"},
              dbg_off_golden);
    dbg_tail_dumped = 1'b0;
end
`endif

`ifdef DUMP_BEH_STAGE_CMP
// Simulation only: compare conv1/conv2/ctr vs numpy trunk Activation golden.
// Phenomenon: argmax idx 120 vs golden 136 — score@136 too low (rtl 007d vs g 0087).
// Golden: box_head_shared_after_conv1_out_bi.txt, conv2, tail_ctr_after_conv_out_bi.txt
reg [15:0] dbg_sh1_golden [0:SH1_LEN-1];
reg [15:0] dbg_sh2_golden [0:SH2_LEN-1];
reg [15:0] dbg_raw_ctr_golden [0:FEAT_SZ-1];
reg [15:0] dbg_score_golden_stg [0:FEAT_SZ-1];
reg [15:0] dbg_opt_golden    [0:OPT_LEN-1];

function integer beh_abs_delta;
    input signed [31:0] a;
    input signed [31:0] b;
    reg signed [31:0] d;
    begin
        d = a - b;
        if (d < 0) d = -d;
        beh_abs_delta = d;
    end
endfunction

// Q8.8 saturate：對 48-bit 累加後 (acc>>>8)+bias 再取 16-bit（對齊 numpy fp）
function signed [15:0] beh_sat_q88;
    input signed [47:0] v;
    begin
        if (v > $signed(48'sd32767))
            beh_sat_q88 = 16'sh7FFF;
        else if (v < $signed(-48'sd32768))
            beh_sat_q88 = -16'sh8000;
        else
            beh_sat_q88 = v[15:0];
    end
endfunction

initial begin
    $readmemb({`BEH_ACT_DIR, "/box_head_shared_after_conv1_out_bi.txt"},
              dbg_sh1_golden);
    $readmemb({`BEH_ACT_DIR, "/box_head_shared_after_conv2_out_bi.txt"},
              dbg_sh2_golden);
    $readmemb({`BEH_ACT_DIR, "/box_head_tail_ctr_after_conv_out_bi.txt"},
              dbg_raw_ctr_golden);
    $readmemb({`BEH_ACT_DIR, "/box_head_after_forward_head_score_map_bi.txt"},
              dbg_score_golden_stg);
    $readmemb({`BEH_ACT_DIR, "/box_head_head_input_bi.txt"},
              dbg_opt_golden);
    $display("[BEH] DUMP_BEH_STAGE_CMP on: opt/conv golden loaded from %s/", `BEH_ACT_DIR);
end
`endif

// ---------------------------------------------------------------------------
// Weights（initial $readmemb 載入；對齊 numpy flatten 順序）
// ---------------------------------------------------------------------------
reg signed [15:0] c1_w [0:CONV1_W_N-1];
reg signed [15:0] c1_b [0:C_SH1-1];
reg signed [15:0] c2_w [0:CONV2_W_N-1];
reg signed [15:0] c2_b [0:C_SH2-1];
reg signed [15:0] ctr_w  [0:TAIL_W_CH-1];
reg signed [15:0] ctr_b  [0:0];
reg signed [15:0] sz_w   [0:2*TAIL_W_CH-1];
reg signed [15:0] sz_b   [0:1];
reg signed [15:0] of_w   [0:2*TAIL_W_CH-1];
reg signed [15:0] of_b   [0:1];

initial begin : beh_load_weights
    integer zi, zcnt;
    reg [1023:0] wpath;
    wpath = {`BEH_WEIGHT_DIR, "/box_head_shared_conv1_folded_weight_bi.txt"};
    $readmemb(wpath, c1_w);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_shared_conv1_folded_bias_bi.txt"},   c1_b);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_shared_conv2_folded_weight_bi.txt"}, c2_w);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_shared_conv2_folded_bias_bi.txt"},   c2_b);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_tail_ctr_weight_bi.txt"},            ctr_w);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_tail_ctr_bias_bi.txt"},              ctr_b);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_tail_size_weight_bi.txt"},           sz_w);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_tail_size_bias_bi.txt"},             sz_b);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_tail_offset_weight_bi.txt"},         of_w);
    $readmemb({`BEH_WEIGHT_DIR, "/box_head_tail_offset_bias_bi.txt"},           of_b);
    $display("[BEH] Weights loaded from %s/", `BEH_WEIGHT_DIR);
    $display("[BEH_WEIGHT_SANITY] c1_w[0]=%h c1_w[288]=%h c1_b[0]=%h c2_w[0]=%h ctr_w[0]=%h",
             c1_w[0], c1_w[288], c1_b[0], c2_w[0], ctr_w[0]);
    zcnt = 0;
    for (zi = 0; zi < 256; zi = zi + 1)
        if (c1_w[zi] === 16'sd0)
            zcnt = zcnt + 1;
    if (zcnt > 200)
        $display("[BEH_FATAL_HINT] c1_w[0:255] mostly ZERO (%0d/256); check CWD, file=%s",
                 zcnt, wpath);
    if ((c1_w[0] === 16'sd0) && (c1_w[1] === 16'sd0) && (c1_w[2] === 16'sd0))
        $display("[BEH_FATAL_HINT] c1_w[0..2] all zero; $readmemb failed? CWD must have TXT_File/Weight");
    if (^c1_w[0] === 1'bx)
        $display("[BEH_FATAL_HINT] c1_w[0] is X; weight file not loaded");
end

// ---------------------------------------------------------------------------
// Fill counters：S_FILL 接收 backbone stream（10240 筆），僅 search 段寫 opt
// stream order：token_idx*IN_CH + ic（token-major，與 numpy enc_opt transpose 對齊）
//   opt_mem[ic*FEAT_SZ + token_idx]  ←  a_i  (token_idx ∈ [0,255], ic ∈ [0,31])
// ---------------------------------------------------------------------------
reg [13:0] fill_cnt;
wire [13:0] fill_off    = fill_cnt - SKIP_VALS;
wire [4:0]  fill_ic     = fill_off[4:0];            // ic
wire [7:0]  fill_tok    = fill_off[12:5];           // token_idx
wire        fill_search = (fill_cnt >= SKIP_VALS) && (fill_cnt < TOT_VALS);

// ---------------------------------------------------------------------------
// Output-pixel counters（所有 conv 共用）
// ---------------------------------------------------------------------------
reg [6:0] cur_oc;
reg [3:0] cur_oh, cur_ow;

// ---------------------------------------------------------------------------
// Inner accumulation counters
// ---------------------------------------------------------------------------
reg [6:0]  mac_ic;       // up to 95 for conv2
reg [1:0]  mac_kh, mac_kw;
reg        mac_bp;       // 0=accumulate phase, 1=bias/write phase
reg signed [47:0] mac_acc;
// 最後一個 MAC tap 的 (acc+prod) 結果；在 mac_end 當拍鎖存，下一拍 mac_bp 寫 mem
reg signed [15:0] mac_wr_hold;

// 3×3 邊界 padding（與 numpy padding=1 一致；越界回傳 0）
// 6-bit signed：min=-1，max=15+2-1=16，需 range [-32,31]，避免 5-bit wrap
wire signed [5:0] ph_s = $signed({2'b00, cur_oh}) + $signed({4'b0, mac_kh}) - 6'sd1;
wire signed [5:0] pw_s = $signed({2'b00, cur_ow}) + $signed({4'b0, mac_kw}) - 6'sd1;
wire        pad_at_addr = (state==S_CONV1 || state==S_CONV2) &&
                          ((ph_s < 0) || (ph_s > 6'sd15) ||
                           (pw_s < 0) || (pw_s > 6'sd15));
wire [3:0]  ph = ph_s[3:0];
wire [3:0]  pw = pw_s[3:0];

// ---------------------------------------------------------------------------
// Feature read（無 SRAM 延遲）：依當前 state 從對應 mem 取值
// ---------------------------------------------------------------------------
wire [12:0] opt_rd = {mac_ic[4:0], ph, pw};              // ic*256 + (ph*16+pw)
wire [14:0] sh1_rd = {mac_ic[6:0], ph, pw};              // ic*256 + (ph*16+pw)，conv2 用
wire [13:0] sh2_rd = {mac_ic[5:0], cur_oh, cur_ow};      // tail 1x1：用本格 (oh,ow)

wire signed [15:0] feat_q =
    pad_at_addr                ? 16'sd0      :
    (state == S_CONV1)         ? opt_mem[opt_rd] :
    (state == S_CONV2)         ? sh1_mem[sh1_rd] :
    (state==S_CTR||state==S_SIZE||state==S_OFF) ? sh2_mem[sh2_rd] :
                                 16'sd0;

// ---------------------------------------------------------------------------
// Weight pick（純組合 mux；對齊 numpy flatten 順序）
// ---------------------------------------------------------------------------
// conv1: oc*32*9 + ic*9 + kh*3 + kw
wire [19:0] c1_wa = cur_oc * 7'd32 * 4'd9
                  + mac_ic * 4'd9
                  + mac_kh * 2'd3
                  + mac_kw;
// conv2: oc*96*9 + ic*9 + kh*3 + kw
wire [19:0] c2_wa = cur_oc * 7'd96 * 4'd9
                  + mac_ic * 4'd9
                  + mac_kh * 2'd3
                  + mac_kw;
// tail_ctr  : ic
// tail_size : oc*48 + ic  （numpy C-order；勿用 {oc[0],ic} 會變 64+ic 越界 sz_w[96]）
// tail_off  : oc*48 + ic
wire [6:0]  ctr_wa = mac_ic[5:0];
wire [6:0]  sz_wa  = (cur_oc[0] ? 7'd48 : 7'd0) + mac_ic[5:0];
wire [6:0]  of_wa  = (cur_oc[0] ? 7'd48 : 7'd0) + mac_ic[5:0];

wire signed [15:0] rom_q =
    (state == S_CONV1) ? c1_w[c1_wa] :
    (state == S_CONV2) ? c2_w[c2_wa] :
    (state == S_CTR)   ? ctr_w[ctr_wa] :
    (state == S_SIZE)  ? sz_w[sz_wa]   :
    (state == S_OFF)   ? of_w[of_wa]   :
                          16'sd0;

wire signed [15:0] bias_q =
    (state == S_CONV1) ? c1_b[cur_oc[6:0]] :
    (state == S_CONV2) ? c2_b[cur_oc[5:0]] :
    (state == S_CTR)   ? ctr_b[0]         :
    (state == S_SIZE)  ? sz_b[cur_oc[0]]  :
    (state == S_OFF)   ? of_b[cur_oc[0]]  :
                          16'sd0;

// ---------------------------------------------------------------------------
// MAC / bias / Q8.8 飽和 / ReLU / Sigmoid
// 對齊 numpy: acc>>>8 + bias → fp(saturate Q8.8) → relu or sigmoid
// ---------------------------------------------------------------------------
// 顯式 sign-extend 後 32-bit signed mul（避免 Verilog 寬度推導歧義）
wire signed [31:0] mac_prod = $signed(feat_q) * $signed(rom_q);

// 本拍若再累加一次乘積後的完整 acc（用於 mac_end 當拍鎖存寫出值）
wire signed [47:0] mac_acc_next = mac_acc + {{16{mac_prod[31]}}, mac_prod};
wire signed [47:0] mac_sh_next  = $signed(mac_acc_next) >>> 8;
wire signed [47:0] mac_bi_next    = mac_sh_next + {{32{bias_q[15]}}, bias_q};
wire signed [15:0] mac_cl_next   = beh_sat_q88(mac_bi_next);
wire signed [15:0] mac_relu_next  = mac_cl_next[15] ? 16'sd0 : mac_cl_next;

wire signed [47:0] mac_sh   = $signed(mac_acc) >>> 8;
wire signed [47:0] mac_bi   = mac_sh + {{32{bias_q[15]}}, bias_q};
wire signed [15:0] mac_cl   = beh_sat_q88(mac_bi);
wire signed [15:0] mac_relu = mac_cl[15] ? 16'sd0 : mac_cl;

wire [7:0] sig_out;
sigmoid_lut u_sig (.x_i(mac_cl), .y_o(sig_out));
wire signed [15:0] mac_sig = {8'b0, sig_out};

wire [7:0] sig_out_next;
sigmoid_lut u_sig_next (.x_i(mac_cl_next), .y_o(sig_out_next));
wire signed [15:0] mac_sig_next = {8'b0, sig_out_next};

// End-of-inner-loop（IN_CH×KH×KW 走完）
wire mac_end_3x3_c1 = (mac_ic == IN_CH_M1)  && (mac_kh == 2'd2) && (mac_kw == 2'd2);
wire mac_end_3x3_c2 = (mac_ic == C_SH1_M1)  && (mac_kh == 2'd2) && (mac_kw == 2'd2);
wire mac_end_1x1    = (mac_ic == C_SH2_M1);

// End-of-output-pixel（全 oc/oh/ow 走完）
wire all_done_c1  = (cur_oc == C_SH1_M1) && (cur_oh == FEAT_H_M1) && (cur_ow == FEAT_W_M1);
wire all_done_c2  = (cur_oc == C_SH2_M1) && (cur_oh == FEAT_H_M1) && (cur_ow == FEAT_W_M1);
wire all_done_ctr = (cur_oh == FEAT_H_M1) && (cur_ow == FEAT_W_M1);
wire all_done_2ch = (cur_oc == 7'd1) && (cur_oh == FEAT_H_M1) && (cur_ow == FEAT_W_M1);

// ---------------------------------------------------------------------------
// cal_bbox streaming：S_BBOX 將 score(256) + size(512) + off(512) 流向 cal_bbox
// ---------------------------------------------------------------------------
reg  [10:0] bcnt;
reg         bbox_start;
wire        bbox_busy, bbox_done;
wire [15:0] bbox_cx, bbox_cy, bbox_w, bbox_h;

// cal_bbox 狀態碼（與 cal_bbox.v / head_top.v 一致）
localparam CB_ARGMAX = 3'd1;
localparam CB_SIZE   = 3'd2;
localparam CB_OFFSET = 3'd3;

wire [2:0] bbox_u_st = u_bbox.state;

// bcnt 0..255: score；256..767: size；768..1279: offset
wire       bbox_phase_score = (bcnt < BBOX_N_SCORE);
wire       bbox_phase_size  = (bcnt >= BBOX_N_SCORE) &&
                              (bcnt <  BBOX_N_SCORE + BBOX_N_SIZE);
wire       bbox_phase_off   = (bcnt >= BBOX_N_SCORE + BBOX_N_SIZE) &&
                              (bcnt <= BBOX_STREAM_LAST);

// 僅在 cal_bbox 處於對應 phase 時才送 valid 並遞增 bcnt（否則永遠等不到 bbox_done）
wire       bbox_stream_advance = (
    (bbox_phase_score && (bbox_u_st == CB_ARGMAX)) ||
    (bbox_phase_size  && (bbox_u_st == CB_SIZE))   ||
    (bbox_phase_off   && (bbox_u_st == CB_OFFSET))
);

wire [7:0]  bcnt_score_idx = bcnt[7:0];
wire [8:0]  bcnt_size_idx  = bcnt - BBOX_N_SCORE;
wire [8:0]  bcnt_off_idx   = bcnt - (BBOX_N_SCORE + BBOX_N_SIZE);

wire signed [15:0] bbox_sc_q = score_mem[bcnt_score_idx];
wire signed [15:0] bbox_sz_q = size_mem[bcnt_size_idx];
wire signed [15:0] bbox_of_q = off_mem[bcnt_off_idx];

// 行為層 mem 為組合讀，無 SRAM 1-cycle 延遲；不可再 reg 延遲 bbox_cal_*（會與
// score_valid 同拍錯位，argmax 偏移、size1/off1 寫入錯格 → cy/h 為 X 或與 golden 差很大）
wire signed [15:0] bbox_cal_sc = bbox_sc_q;
wire signed [15:0] bbox_cal_sz = bbox_sz_q;
wire signed [15:0] bbox_cal_of = bbox_of_q;
wire               bbox_sc_v = (state == S_BBOX) && bbox_busy &&
                               bbox_phase_score && (bbox_u_st == CB_ARGMAX);
wire               bbox_sz_v = (state == S_BBOX) && bbox_busy &&
                               bbox_phase_size  && (bbox_u_st == CB_SIZE);
wire               bbox_of_v = (state == S_BBOX) && bbox_busy &&
                               bbox_phase_off   && (bbox_u_st == CB_OFFSET);

cal_bbox #(
    .FEAT_SZ  (FEAT_H),
    .FEAT_LEN (FEAT_SZ),
    .SHIFT    (4)
) u_bbox (
    .clk          (clk),
    .reset        (reset),
    .start        (bbox_start),
    .score_i      (bbox_cal_sc),
    .score_valid  (bbox_sc_v),
    .size_i       (bbox_cal_sz),
    .size_valid   (bbox_sz_v),
    .offset_i     (bbox_cal_of),
    .offset_valid (bbox_of_v),
    .busy         (bbox_busy),
    .done         (bbox_done),
    .cx_o         (bbox_cx),
    .cy_o         (bbox_cy),
    .w_o          (bbox_w),
    .h_o          (bbox_h)
);

assign cx_o = bbox_cx;
assign cy_o = bbox_cy;
assign w_o  = bbox_w;
assign h_o  = bbox_h;

// ---------------------------------------------------------------------------
// Main FSM / datapath
// 同一 always 區塊內：累加 / 寫 mem / 推進 counter；不需要 mac_dv pipeline
// ---------------------------------------------------------------------------
integer reset_i;

always @(posedge clk) begin
    done       <= 1'b0;
    bbox_start <= 1'b0;

    if (reset) begin
        state    <= S_IDLE;
        fill_cnt <= 14'd0;
        cur_oc   <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
        mac_ic   <= 7'd0; mac_kh <= 2'd0; mac_kw <= 2'd0;
        mac_acc  <= 48'sd0;
        mac_bp   <= 1'b0;
        bcnt     <= 11'd0;
        mac_wr_hold <= 16'sd0;
    end else begin
        case (state)

        // -------------------------------------------------------------------
        S_IDLE: begin
            fill_cnt <= 14'd0;
            cur_oc   <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
            mac_ic   <= 7'd0; mac_kh <= 2'd0; mac_kw <= 2'd0;
            mac_acc  <= 48'sd0;
            mac_bp   <= 1'b0;
            bcnt     <= 11'd0;
            if (start) state <= S_FILL;
        end

        // -------------------------------------------------------------------
        // S_FILL：接收 320 token × 32 ch，僅 search 段（後 256 token）寫 opt
        S_FILL: begin
            if (a_valid) begin
                if (fill_search)
                    opt_mem[{fill_ic, fill_tok}] <= a_i;
                if (fill_cnt < TOT_VALS - 1)
                    fill_cnt <= fill_cnt + 14'd1;
                else
                    state <= S_CONV1;
            end
        end

        // -------------------------------------------------------------------
        // S_CONV1：3×3, IN_CH→C_SH1, ReLU
        S_CONV1: begin
            if (!mac_bp) begin
                mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
`ifdef DUMP_BEH_TAP
                if (cur_oc == 7'd0 && mac_ic == 7'd0 && mac_kh == 2'd1 && mac_kw == 2'd1 &&
                    ({cur_oh, cur_ow} == 8'd3 || {cur_oh, cur_ow} == 8'd135 ||
                     {cur_oh, cur_ow} == 8'd136))
                    $display("[BEH_TAP] S_CONV1 idx=%0d ic=%0d kh=%0d kw=%0d ph=%0d pw=%0d pad=%b feat=%h rom=%h prod=%h acc=%h",
                             {cur_oh, cur_ow}, mac_ic, mac_kh, mac_kw, ph, pw,
                             pad_at_addr, feat_q, rom_q, mac_prod[31:0], mac_acc);
`endif
`ifdef DUMP_BEH_STAGE_CMP
                // 第一個 MAC tap（kh=kw=0）也印，避免只盯 kh=1,kw=1 漏 log
                if ((cur_oc == 7'd0) && (mac_ic == 7'd0) &&
                    (mac_kh == 2'd0) && (mac_kw == 2'd0) &&
                    (({cur_oh, cur_ow} == 8'd120) || ({cur_oh, cur_ow} == 8'd136)))
                    $display(
                        "[BEH_OPT_CMP] conv1 tap0 idx=%0d opt_rtl=%h opt_g=%h pad=%b rom=%h prod=%h",
                        {cur_oh, cur_ow}, feat_q,
                        dbg_opt_golden[{5'd0, cur_oh, cur_ow}],
                        pad_at_addr, rom_q, mac_prod[31:0]);
                if (({cur_oh, cur_ow} == 8'd136) && (cur_oc == 7'd0) &&
                    (mac_ic == 7'd0) && (mac_kh == 2'd1) && (mac_kw == 2'd1))
                    $display(
                        "[BEH_OPT_CMP] conv1 tap idx=136 ic0 kh=1 kw=1 ph=%0d pw=%0d opt_rtl=%h opt_g=%h",
                        ph, pw, feat_q, dbg_opt_golden[{5'd0, ph, pw}]);
`endif
                if (mac_end_3x3_c1) begin
                    mac_wr_hold <= mac_relu_next;
                    mac_bp      <= 1'b1;
                end else if (mac_kw == 2'd2) begin
                    mac_kw <= 2'd0;
                    if (mac_kh == 2'd2) begin
                        mac_kh <= 2'd0;
                        mac_ic <= mac_ic + 7'd1;
                    end else begin
                        mac_kh <= mac_kh + 2'd1;
                    end
                end else begin
                    mac_kw <= mac_kw + 2'd1;
                end
            end else begin
                // Bias / write phase
`ifdef DUMP_BEH_STAGE_CMP
                if ((cur_oc == 7'd0) &&
                    (({cur_oh, cur_ow} == 8'd120) || ({cur_oh, cur_ow} == 8'd136)))
                    $display(
                        "[BEH_OPT_CMP] conv1 wr idx=%0d opt_ch0=%h opt_g=%h mac_acc=%h bias=%h relu=%h",
                        {cur_oh, cur_ow}, opt_mem[{5'd0, cur_oh, cur_ow}],
                        dbg_opt_golden[{5'd0, cur_oh, cur_ow}],
                        mac_acc, bias_q, mac_wr_hold, mac_cl_next, mac_relu_next);
`endif
`ifdef DUMP_BEH_CONV1_PATCH
                if (({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136) &&
                    (cur_oc == 7'd0))
                    $display(
                        "[BEH_CONV1_PATCH] idx=%0d oc=%0d acc=%h acc_n=%h cl_n=%h relu_n=%h hold=%h",
                        {cur_oh, cur_ow}, cur_oc, mac_acc, mac_acc_next,
                        mac_cl_next, mac_relu_next, mac_wr_hold);
`endif
                sh1_mem[{cur_oc[6:0], cur_oh, cur_ow}] <= mac_wr_hold;
`ifdef DUMP_BEH_GOLDEN
                if ({cur_oh, cur_ow} == 8'd3 || {cur_oh, cur_ow} == 8'd135 ||
                    {cur_oh, cur_ow} == 8'd136)
                    $display("[BEH] S_CONV1 oc=%0d idx=%0d acc=%h sh=%h bias=%h cl=%h relu=%h",
                             cur_oc, {cur_oh, cur_ow}, mac_acc, mac_sh[15:0],
                             bias_q, mac_cl, mac_relu);
`endif
`ifdef DUMP_BEH_STAGE_CMP
                if (({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136) &&
                    (cur_oc == 7'd0 || cur_oc == 7'd17 || cur_oc == 7'd32))
                    $display(
                        "[BEH_STAGE_CMP] S_CONV1 wr oc=%0d idx=%0d hold=%h cl_n=%h golden=%h delta=%0d",
                        cur_oc, {cur_oh, cur_ow}, mac_wr_hold, mac_cl_next,
                        dbg_sh1_golden[{cur_oc[6:0], cur_oh, cur_ow}],
                        beh_abs_delta($signed(mac_wr_hold),
                            $signed(dbg_sh1_golden[{cur_oc[6:0], cur_oh, cur_ow}])));
`endif
                mac_acc <= 48'sd0;
                mac_bp  <= 1'b0;
                mac_ic  <= 7'd0; mac_kh <= 2'd0; mac_kw <= 2'd0;
                if (all_done_c1) begin
                    cur_oc <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
                    state  <= S_CONV2;
                end else if (cur_ow == FEAT_W_M1) begin
                    cur_ow <= 4'd0;
                    if (cur_oh == FEAT_H_M1) begin
                        cur_oh <= 4'd0;
                        cur_oc <= cur_oc + 7'd1;
                    end else begin
                        cur_oh <= cur_oh + 4'd1;
                    end
                end else begin
                    cur_ow <= cur_ow + 4'd1;
                end
            end
        end

        // -------------------------------------------------------------------
        // S_CONV2：3×3, C_SH1→C_SH2, ReLU
        S_CONV2: begin
            if (!mac_bp) begin
                mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                if (mac_end_3x3_c2) begin
                    mac_wr_hold <= mac_relu_next;
                    mac_bp      <= 1'b1;
                end else if (mac_kw == 2'd2) begin
                    mac_kw <= 2'd0;
                    if (mac_kh == 2'd2) begin
                        mac_kh <= 2'd0;
                        mac_ic <= mac_ic + 7'd1;
                    end else begin
                        mac_kh <= mac_kh + 2'd1;
                    end
                end else begin
                    mac_kw <= mac_kw + 2'd1;
                end
            end else begin
                sh2_mem[{cur_oc[5:0], cur_oh, cur_ow}] <= mac_wr_hold;
`ifdef DUMP_BEH_GOLDEN
                if ({cur_oh, cur_ow} == 8'd136 && (cur_oc == 7'd0 || cur_oc == 7'd47))
                    $display("[BEH] S_CONV2 oc=%0d idx=%0d acc=%h cl=%h relu=%h",
                             cur_oc, {cur_oh, cur_ow}, mac_acc, mac_cl, mac_relu);
`endif
`ifdef DUMP_BEH_STAGE_CMP
                if (({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136) &&
                    (cur_oc == 7'd0 || cur_oc == 7'd17 || cur_oc == 7'd21 ||
                     cur_oc == 7'd32 || cur_oc == 7'd47))
                    $display(
                        "[BEH_STAGE_CMP] S_CONV2 wr oc=%0d idx=%0d hold=%h cl_n=%h golden=%h delta=%0d",
                        cur_oc, {cur_oh, cur_ow}, mac_wr_hold, mac_cl_next,
                        dbg_sh2_golden[{cur_oc[5:0], cur_oh, cur_ow}],
                        beh_abs_delta($signed(mac_wr_hold),
                            $signed(dbg_sh2_golden[{cur_oc[5:0], cur_oh, cur_ow}])));
`endif
                mac_acc <= 48'sd0;
                mac_bp  <= 1'b0;
                mac_ic  <= 7'd0; mac_kh <= 2'd0; mac_kw <= 2'd0;
                if (all_done_c2) begin
                    cur_oc <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
                    state  <= S_CTR;
                end else if (cur_ow == FEAT_W_M1) begin
                    cur_ow <= 4'd0;
                    if (cur_oh == FEAT_H_M1) begin
                        cur_oh <= 4'd0;
                        cur_oc <= cur_oc + 7'd1;
                    end else begin
                        cur_oh <= cur_oh + 4'd1;
                    end
                end else begin
                    cur_ow <= cur_ow + 4'd1;
                end
            end
        end

        // -------------------------------------------------------------------
        // S_CTR：1×1, 48→1, sigmoid
        S_CTR: begin
            if (!mac_bp) begin
                mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
`ifdef DUMP_BEH_TAP
                if (mac_ic == 7'd0 &&
                    ({cur_oh, cur_ow} == 8'd136 || {cur_oh, cur_ow} == 8'd3))
                    $display("[BEH_TAP] S_CTR idx=%0d ic=%0d sh2=%h ctr_w=%h prod=%h",
                             {cur_oh, cur_ow}, mac_ic, feat_q, rom_q, mac_prod[31:0]);
`endif
`ifdef DUMP_BEH_STAGE_CMP
                if (({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136) &&
                    (mac_ic == 7'd0 || mac_ic == 7'd7 || mac_ic == 7'd15 ||
                     mac_ic == 7'd17 || mac_ic == 7'd21 || mac_ic == 7'd23 ||
                     mac_ic == 7'd31 || mac_ic == 7'd32 || mac_ic == 7'd39 ||
                     mac_ic == 7'd47))
                    $display(
                        "[BEH_CTR_STEP] idx=%0d ic=%0d sh2_rtl=%h sh2_g=%h ctr_w=%h prod=%h acc=%h",
                        {cur_oh, cur_ow}, mac_ic, feat_q,
                        dbg_sh2_golden[{mac_ic[5:0], cur_oh, cur_ow}],
                        rom_q, mac_prod[31:0], mac_acc);
`endif
                if (mac_end_1x1) begin
                    mac_wr_hold <= mac_sig_next;
                    mac_bp      <= 1'b1;
                end else begin
                    mac_ic <= mac_ic + 7'd1;
                end
            end else begin
                score_mem[{cur_oh, cur_ow}] <= mac_wr_hold;
`ifdef DUMP_BEH_GOLDEN
                if ({cur_oh, cur_ow} == 8'd3 || {cur_oh, cur_ow} == 8'd135 ||
                    {cur_oh, cur_ow} == 8'd136 || {cur_oh, cur_ow} == 8'd120)
                    $display("[BEH] S_CTR idx=%0d acc=%h cl=%h sig=%02h mac_sig=%h",
                             {cur_oh, cur_ow}, mac_acc, mac_cl, sig_out, mac_sig);
`endif
`ifdef DUMP_BEH_BBOX_DEBUG
                if ({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136)
                    $display("[BEH_SCORE_CMP] S_CTR idx=%0d rtl_sc=%h golden_sc=%h",
                             {cur_oh, cur_ow}, mac_sig, dbg_score_golden[{cur_oh, cur_ow}]);
`endif
`ifdef DUMP_BEH_STAGE_CMP
                if ({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136)
                    $display(
                        "[BEH_STAGE_CMP] S_CTR done idx=%0d acc=%h cl=%h sig=%02h sc=%h | raw_g=%h sc_g=%h",
                        {cur_oh, cur_ow}, mac_acc, mac_cl, sig_out, mac_sig,
                        dbg_raw_ctr_golden[{cur_oh, cur_ow}],
                        dbg_score_golden_stg[{cur_oh, cur_ow}]);
`endif
                mac_acc <= 48'sd0;
                mac_bp  <= 1'b0;
                mac_ic  <= 7'd0;
                if (all_done_ctr) begin
                    cur_oc <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
                    state  <= S_SIZE;
                end else if (cur_ow == FEAT_W_M1) begin
                    cur_ow <= 4'd0;
                    cur_oh <= cur_oh + 4'd1;
                end else begin
                    cur_ow <= cur_ow + 4'd1;
                end
            end
        end

        // -------------------------------------------------------------------
        // S_SIZE：1×1, 48→2, sigmoid (per output ch ∈ {0,1})
        S_SIZE: begin
            if (!mac_bp) begin
                mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                if (mac_end_1x1) begin
                    mac_wr_hold <= mac_sig_next;
                    mac_bp      <= 1'b1;
                end else begin
                    mac_ic <= mac_ic + 7'd1;
                end
            end else begin
                size_mem[{cur_oc[0], cur_oh, cur_ow}] <= mac_wr_hold;
`ifdef DUMP_BEH_TAIL_WR
                if ({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136)
                    $display(
                        "[BEH_TAIL_WR] S_SIZE wr oc=%0d idx=%0d sz_wa=%0d rom=%h feat=%h acc=%h cl=%h sig=%02h mac_sig=%h",
                        cur_oc[0], {cur_oh, cur_ow}, sz_wa, rom_q, feat_q,
                        mac_acc, mac_cl, sig_out, mac_sig);
`endif
                mac_acc <= 48'sd0;
                mac_bp  <= 1'b0;
                mac_ic  <= 7'd0;
                if (all_done_2ch) begin
                    cur_oc <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
                    state  <= S_OFF;
                end else if (cur_ow == FEAT_W_M1) begin
                    cur_ow <= 4'd0;
                    if (cur_oh == FEAT_H_M1) begin
                        cur_oh <= 4'd0;
                        cur_oc <= cur_oc + 7'd1;
                    end else begin
                        cur_oh <= cur_oh + 4'd1;
                    end
                end else begin
                    cur_ow <= cur_ow + 4'd1;
                end
            end
        end

        // -------------------------------------------------------------------
        // S_OFF：1×1, 48→2, raw Q8.8（不 sigmoid）
        S_OFF: begin
            if (!mac_bp) begin
                mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                if (mac_end_1x1) begin
                    mac_wr_hold <= mac_cl_next;
                    mac_bp      <= 1'b1;
                end else begin
                    mac_ic <= mac_ic + 7'd1;
                end
            end else begin
                off_mem[{cur_oc[0], cur_oh, cur_ow}] <= mac_wr_hold;
`ifdef DUMP_BEH_TAIL_WR
                if ({cur_oh, cur_ow} == 8'd120 || {cur_oh, cur_ow} == 8'd136)
                    $display(
                        "[BEH_TAIL_WR] S_OFF wr oc=%0d idx=%0d of_wa=%0d rom=%h feat=%h acc=%h mac_cl=%h",
                        cur_oc[0], {cur_oh, cur_ow}, of_wa, rom_q, feat_q, mac_acc, mac_cl);
`endif
                mac_acc <= 48'sd0;
                mac_bp  <= 1'b0;
                mac_ic  <= 7'd0;
                if (all_done_2ch) begin
                    cur_oc <= 7'd0; cur_oh <= 4'd0; cur_ow <= 4'd0;
                    state  <= S_BBOX;
                    bbox_start <= 1'b1;
                    bcnt   <= 11'd0;
`ifdef DUMP_BEH_BBOX_DEBUG
                    if (!dbg_tail_dumped) begin
                        dbg_tail_dumped <= 1'b1;
                        dbg_rtl_max    = 16'sh8000;
                        dbg_rtl_argmax = 8'd0;
                        dbg_g_max      = 16'sh8000;
                        dbg_g_argmax   = 8'd0;
                        for (dbg_ai = 0; dbg_ai < FEAT_SZ; dbg_ai = dbg_ai + 1) begin
                            if ($signed(score_mem[dbg_ai]) > dbg_rtl_max) begin
                                dbg_rtl_max    = score_mem[dbg_ai];
                                dbg_rtl_argmax = dbg_ai;
                            end
                            if ($signed(dbg_score_golden[dbg_ai]) > dbg_g_max) begin
                                dbg_g_max    = dbg_score_golden[dbg_ai];
                                dbg_g_argmax = dbg_ai;
                            end
                        end
                        $display("\n[BEH_TAIL_DUMP] before S_BBOX (score/size/off mem filled)");
                        $display("  RTL argmax idx=%0d (oh=%0d ow=%0d) max_sc=%h",
                                 dbg_rtl_argmax, dbg_rtl_argmax[7:4],
                                 dbg_rtl_argmax[3:0], dbg_rtl_max);
                        $display("  Golden argmax idx=%0d (oh=%0d ow=%0d) max_sc=%h",
                                 dbg_g_argmax, dbg_g_argmax[7:4],
                                 dbg_g_argmax[3:0], dbg_g_max);
                        for (dbg_gi = 0; dbg_gi < 3; dbg_gi = dbg_gi + 1) begin
                            if (dbg_gi == 0)
                                dbg_anchor_idx = 8'd3;
                            else if (dbg_gi == 1)
                                dbg_anchor_idx = 8'd135;
                            else
                                dbg_anchor_idx = 8'd136;
                            $display(
                                "  idx=%0d (oh=%0d ow=%0d) | sc rtl=%h g=%h | sz0 rtl=%h g=%h sz1 rtl=%h g=%h | of0 rtl=%h g=%h of1 rtl=%h g=%h",
                                dbg_anchor_idx, dbg_anchor_idx[7:4], dbg_anchor_idx[3:0],
                                score_mem[dbg_anchor_idx], dbg_score_golden[dbg_anchor_idx],
                                size_mem[dbg_anchor_idx], dbg_size_golden[dbg_anchor_idx],
                                size_mem[FEAT_SZ + dbg_anchor_idx],
                                dbg_size_golden[FEAT_SZ + dbg_anchor_idx],
                                off_mem[dbg_anchor_idx], dbg_off_golden[dbg_anchor_idx],
                                off_mem[FEAT_SZ + dbg_anchor_idx],
                                dbg_off_golden[FEAT_SZ + dbg_anchor_idx]);
                        end
                        $display("  flat idx=120 (oh=7 ow=8) sc rtl=%h g=%h | idx=136 sc rtl=%h g=%h",
                                 score_mem[8'd120], dbg_score_golden[8'd120],
                                 score_mem[8'd136], dbg_score_golden[8'd136]);
                        $display("  (cal_bbox argmax on streamed score; cy/h from size1/off1 at argmax idx)");
                    end
`endif
                end else if (cur_ow == FEAT_W_M1) begin
                    cur_ow <= 4'd0;
                    if (cur_oh == FEAT_H_M1) begin
                        cur_oh <= 4'd0;
                        cur_oc <= cur_oc + 7'd1;
                    end else begin
                        cur_oh <= cur_oh + 4'd1;
                    end
                end else begin
                    cur_ow <= cur_ow + 4'd1;
                end
            end
        end

        // -------------------------------------------------------------------
        // S_BBOX：串流餵 cal_bbox（相位須與 u_bbox.state 對齊，見 head_top.v）
        S_BBOX: begin
            if (bbox_busy) begin
`ifdef DUMP_BEH_BBOX_DEBUG
                if (bcnt == 11'd120 || bcnt == 11'd136 ||
                    bcnt == 11'd376 || bcnt == 11'd392 ||
                    bcnt == 11'd632 || bcnt == 11'd648 ||
                    bcnt == 11'd1144 || bcnt == 11'd1160)
                    $display(
                        "[BEH_BBOX_STRM] bcnt=%0d bbox_st=%0d sc_v=%b sz_v=%b of_v=%b | sc=%h sz=%h of=%h adv=%b",
                        bcnt, bbox_u_st, bbox_sc_v, bbox_sz_v, bbox_of_v,
                        bbox_sc_q, bbox_sz_q, bbox_of_q, bbox_stream_advance);
`endif
                if (bbox_stream_advance && (bcnt <= BBOX_STREAM_LAST))
                    bcnt <= bcnt + 11'd1;
            end
            if (bbox_done)
                state <= S_DONE;
        end

        // -------------------------------------------------------------------
        S_DONE: begin
            done  <= 1'b1;
            state <= S_IDLE;
        end

        default: state <= S_IDLE;

        endcase

        // 進入各 stage 時重置 inner counter（對齊 head_top.v stage-entry reset）
        if (state == S_FILL && a_valid && (fill_cnt == TOT_VALS_M1)) begin
            cur_oc  <= 7'd0;  cur_oh <= 4'd0;  cur_ow <= 4'd0;
            mac_ic  <= 7'd0;  mac_kh <= 2'd0;  mac_kw <= 2'd0;
            mac_acc <= 48'sd0; mac_bp <= 1'b0;
`ifdef DUMP_BEH_STAGE_CMP
            $display("[BEH_OPT_FILL] after S_FILL opt[120] rtl=%h g=%h | opt[136] rtl=%h g=%h",
                     opt_mem[13'd120], dbg_opt_golden[13'd120],
                     opt_mem[13'd136], dbg_opt_golden[13'd136]);
`endif
        end
        if (state == S_CONV1 && mac_bp && all_done_c1) begin
            cur_oc  <= 7'd0;  cur_oh <= 4'd0;  cur_ow <= 4'd0;
            mac_ic  <= 7'd0;  mac_kh <= 2'd0;  mac_kw <= 2'd0;
            mac_acc <= 48'sd0; mac_bp <= 1'b0;
        end
        if (state == S_CONV2 && mac_bp && all_done_c2) begin
            cur_oc  <= 7'd0;  cur_oh <= 4'd0;  cur_ow <= 4'd0;
            mac_ic  <= 7'd0;  mac_kh <= 2'd0;  mac_kw <= 2'd0;
            mac_acc <= 48'sd0; mac_bp <= 1'b0;
        end
        if (state == S_CTR && mac_bp && all_done_ctr) begin
            cur_oh  <= 4'd0;  cur_ow <= 4'd0;
            mac_ic  <= 7'd0;  mac_acc <= 48'sd0; mac_bp <= 1'b0;
        end
        if (state == S_SIZE && mac_bp && all_done_2ch) begin
            cur_oc  <= 7'd0;  cur_oh <= 4'd0;  cur_ow <= 4'd0;
            mac_ic  <= 7'd0;  mac_acc <= 48'sd0; mac_bp <= 1'b0;
        end
    end
end

assign busy = (state != S_IDLE);

// ---------------------------------------------------------------------------
// Optional FSM heartbeat（除錯用：確認卡在哪個 state）
// ---------------------------------------------------------------------------
`ifdef DUMP_BEH_FSM
reg [31:0] beh_hb_cnt;
always @(posedge clk) begin
    if (reset)
        beh_hb_cnt <= 32'd0;
    else if (state != S_IDLE && state != S_DONE) begin
        if (beh_hb_cnt == 32'd1_000_000) begin
            $display("[BEH_FSM] t=%0t state=%0d oc=%0d oh=%0d ow=%0d ic=%0d mac_bp=%b bcnt=%0d bbox_st=%0d",
                     $time, state, cur_oc, cur_oh, cur_ow, mac_ic, mac_bp, bcnt, bbox_u_st);
            beh_hb_cnt <= 32'd0;
        end else
            beh_hb_cnt <= beh_hb_cnt + 32'd1;
    end else
        beh_hb_cnt <= 32'd0;
end
`endif

// ---------------------------------------------------------------------------
// 預設將未使用的索引壓 0，避免 X 傳播（純 sim helper）
// ---------------------------------------------------------------------------
initial begin
    for (reset_i = 0; reset_i < OPT_LEN;     reset_i = reset_i + 1) opt_mem[reset_i]   = 16'sd0;
    for (reset_i = 0; reset_i < SH1_LEN;     reset_i = reset_i + 1) sh1_mem[reset_i]   = 16'sd0;
    for (reset_i = 0; reset_i < SH2_LEN;     reset_i = reset_i + 1) sh2_mem[reset_i]   = 16'sd0;
    for (reset_i = 0; reset_i < FEAT_SZ;     reset_i = reset_i + 1) score_mem[reset_i] = 16'sd0;
    for (reset_i = 0; reset_i < 2*FEAT_SZ;   reset_i = reset_i + 1) size_mem[reset_i]  = 16'sd0;
    for (reset_i = 0; reset_i < 2*FEAT_SZ;   reset_i = reset_i + 1) off_mem[reset_i]   = 16'sd0;
end

endmodule
