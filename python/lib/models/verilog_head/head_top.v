// =============================================================================
// head_top.v  — Shared-Trunk CenterPredictor Head (SRAM/ROM macro version)
//
// 對齊 run_backbone_numpy_shared_trunk.py head_shared_trunk()。
// 架構：opt_feat[32,16,16] → conv1(96,3x3) → conv2(48,3x3)
//        → tail_ctr(1) / tail_size(2) / tail_off(2) → cal_bbox
//
// ROM 對應（參照 rom.pdf）：
//   rom_box_head_shared_conv1_folded_weight1 / weight2 (same addr map as former rom_hd_conv1_w1/w2)
//   rom_box_head_shared_conv1_2_folded_bias  (conv1+conv2 bias ROM)
//   rom_box_head_shared_conv2_folded_weight1..3
//   rom_box_head_tail_ctr_offset_size_weight / _bias (tail packed ROM)
//
// SRAM 對應：
//   Sram_opt   8192×16-bit  A=13  opt_buf
//   Sram_sh1_lo 16384×16-bit  A=14  sh1_buf OC 0..63  (split from Sram_sh1 24576)
//   Sram_sh1_hi  8192×16-bit  A=13  sh1_buf OC 64..95
//   Sram_sh2  12288×16-bit  A=14  sh2_buf
//   Sram_score  256×16-bit  A=8   score_buf（例化 .A(score_sram_a[7:0])）
//   Sram_size   512×16-bit  A=9   size_buf（例化 .A(size_sram_a[8:0])）
//   Sram_off    512×16-bit  A=9   off_buf（例化 .A(off_sram_a[8:0])）
//
// SRAM/ROM 皆採 CLK=~clk（falling-edge）。
// mac_dv：0=SRAM 位址相、1=資料相；每個乘加後必須 mac_dv<=0 再進下一 tap（勿連續 mac_dv=1）。
//
// Simulation（僅供除錯，不可合成）：
//   +define+DUMP_HEAD_FSM
//       在 state=S_BBOX(7) 時每個 posedge 遞增 dbg_sbbox_hb，並每 16 clocks 印 heartbeat
//       （證明模擬未凍結、仍停在 state=7 時也在「跑時脈」）。
//   +define+DUMP_HEAD_S_BBOX_EACH_CLK（須一併開 DUMP_HEAD_FSM）
//       在 S_BBOX 內每個 posedge 都印一行（最嚴格；量很大）。
//   +define+DUMP_HEAD_BBOX_STREAM
//       在 S_BBOX 且 bbox_dv 採樣時，每 64 筆印 bcnt、三條 SRAM 位址、Q 與餵給 cal 的 phase valid（除錯 w/h=X、cy 爆量）。
//   +define+DUMP_HEAD_BBOX_STROBE（建議與 DUMP_HEAD_BBOX_STREAM 併用）
//       同上條件下 $strobe 印本拍 NBA 後 bbox_cal_* 與各 valid（與 score_q 對齊）。
//   +define+DUMP_HEAD_BBOX_SRAM_NEGEDGE
//       同上條件但在 negedge clk 印 sc/sz/of 位址與 Q（對齊 SRAM CLK=~clk，避免 posedge 上 A 已換 Q 仍舊的錯覺）。
//   +define+DUMP_HEAD_CTR_WRITE
//       在 S_CTR 寫 score SRAM 當拍（mac_bp&&mac_dv）每 16 格印 cur_oh/ow、mac_sig、score_sram_a（查是否每格皆 0001）。
//   +define+DUMP_HEAD_SIZE_SAT
//       在 S_SIZE 寫 size SRAM 當拍（mac_bp&&mac_dv）依線性索引每 N 格印一次：
//       oh, ow, cur_oc, mac_cl（sigmoid 前 Q8.8）, sig_out（LUT 8b）, size_sram_a。
//       間距 N 預設 16，編譯前可 +define+HEAD_SIZE_DUMP_STRIDE=32 覆寫（須為常數）。
//   +define+DUMP_HEAD_SCORE_CHECK
//       目的：argmax 與 golden 差 1 格或 score 偏弱時，對拍 score map 錨點與上游 sh2。
//       Golden: box_head_after_forward_head_score_map_bi.txt（flatten index = oh*16+ow）。
//       在 S_CONV2 寫 sh2（僅 oc=0 與 oc=47）、S_CTR 寫 score、S_BBOX 餵 cal_bbox 時，
//       僅印 flat_idx ∈ {3,135,136}。
//   +define+DUMP_HEAD_SCORE_DEEP（建議與 DUMP_HEAD_SCORE_CHECK 併用）
//       目的：S_CTR 的 mac_cl 與 golden 對拍，找出 sigmoid 前就已錯的節點。
//   +define+DUMP_HEAD_FEAT_TAP（與 SCORE_DEEP 併用）
//       S_CTR bias 當拍於 idx 3/135/136 印 opt_rd、feat_raw、feat_q、rom_q、mac_prod（查 MAC 輸入）。
//   [HEAD_BIAS_WR] / [HEAD_MAC_DBG]（SCORE_DEEP）：印 rom_q vs bias_lat、c12b_a、mac_sh、mac_bi48。
//       讀取 TXT_File/Activation/box_head_shared_after_conv2_out_bi.txt（flatten oc*256+oh*16+ow）
//       與 box_head_tail_ctr_after_conv_out_bi.txt（flatten oh*16+ow），
//       在 S_CONV2（oc0/47）/ S_CTR 錨點印 [HEAD_GOLDEN_CMP] PASS/FAIL（預設 ±2 LSB）。
//   +define+DUMP_TB_ARGMAX / +define+DUMP_TB_SCORE_DEEP（TEST_head.v）
//       TB 載入 score / conv2 / raw_ctr golden，結束時印錨點與 backbone 抽樣。
// =============================================================================

`ifndef HEAD_SIZE_DUMP_STRIDE
`define HEAD_SIZE_DUMP_STRIDE 16
`endif

`ifdef DUMP_HEAD_SCORE_CHECK
// flat index = oh*FEAT_W + ow（與 score_sram_a={oh,ow}、numpy C-order flatten 一致）
`define HEAD_SCORE_ANCHOR_IDX(idx) \
    ((idx) == 8'd3 || (idx) == 8'd135 || (idx) == 8'd136)
// conv2 預設僅 oc=0/47；idx=136 另印全部 oc（tail 1x1 加總用）
`define HEAD_SCORE_CONV2_DUMP_OC(cur_oc, oc_last, idx_flat) \
    ((cur_oc) == 7'd0 || (cur_oc) == (oc_last) || ((idx_flat) == 8'd136))
`endif

`ifndef HEAD_GOLDEN_TOL_LSB
`define HEAD_GOLDEN_TOL_LSB 2
`endif

module head_top #(
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
localparam FEAT_SZ   = FEAT_H * FEAT_W;          // 256
localparam SKIP_VALS = LENS_Z * IN_CH;            // 2048
localparam TOT_VALS  = N_TOKENS * IN_CH;          // 10240
localparam C_SH1_M1  = C_SH1 - 1;                // 95
localparam C_SH2_M1  = C_SH2 - 1;                // 47
localparam IN_CH_M1  = IN_CH - 1;                 // 31
localparam FEAT_H_M1 = FEAT_H - 1;               // 15
localparam FEAT_W_M1 = FEAT_W - 1;               // 15
localparam TOT_VALS_M1 = TOT_VALS - 1;           // fill_cnt last index
// S_BBOX → cal_bbox：依序 score(256) + size(512) + offset(512) = 1280 筆；最後一筆 bcnt=1279
localparam BBOX_N_SCORE  = FEAT_SZ;              // 256
localparam BBOX_N_SIZE   = 2 * FEAT_SZ;         // 512
localparam BBOX_N_OFF    = 2 * FEAT_SZ;         // 512
localparam BBOX_STREAM_LAST = BBOX_N_SCORE + BBOX_N_SIZE + BBOX_N_OFF - 1; // 1279

`ifdef DUMP_HEAD_SCORE_DEEP
reg [2:0] dbg_opt_in_logged;
// Golden: box_head_head_input — [C,H,W] → ic*256 + oh*16 + ow
localparam OPT_GOLDEN_LEN   = IN_CH * FEAT_SZ;
reg [15:0] opt_golden_mem [0:OPT_GOLDEN_LEN-1];
// Golden: box_head_shared_after_conv1_out — [C,H,W] → oc*256 + oh*16 + ow
localparam CONV1_GOLDEN_LEN = C_SH1 * FEAT_SZ;
reg [15:0] conv1_golden_mem [0:CONV1_GOLDEN_LEN-1];
// Golden: box_head_shared_after_conv2_out
localparam CONV2_GOLDEN_LEN = C_SH2 * FEAT_SZ;
reg [15:0] conv2_golden_mem [0:CONV2_GOLDEN_LEN-1];
// Golden: box_head_tail_ctr_after_conv_out — oh*16 + ow
reg [15:0] raw_ctr_golden_mem [0:FEAT_SZ-1];
`endif

// conv1 weight total = C_SH1*IN_CH*9 = 96*32*9 = 27648
// conv2 weight total = C_SH2*C_SH1*9 = 48*96*9 = 41472
localparam CONV2_B_BASE = C_SH2 * C_SH1 * 9;     // 41472 (unused; bias in shared ROM)

// ---------------------------------------------------------------------------
// FSM states
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

reg [3:0] state, next_state;

// ---------------------------------------------------------------------------
// Power-gating signals (SLP=0 = normal, SLP=1 = sleep)
// ---------------------------------------------------------------------------
reg pgen_opt, pgen_sh1, pgen_sh2, pgen_score, pgen_size, pgen_off;

// ---------------------------------------------------------------------------
// SRAM chip-enable / write-enable (active low)
// ---------------------------------------------------------------------------
reg mac_bp;
reg mac_dv;   // 1-cycle pipeline: 0=addr phase, 1=data valid phase
// SRAM CLK=~clk：寫入在 posedge 後 negedge；須在 mac_acc 清零前鎖存（見 always 內 blocking）
reg signed [15:0] sram_wdata_r;
// bias ROM CLK=~clk：mac_dv=0 設 c12b_a/csob_a，negedge 鎖存 Q；mac_dv=1 用 bias_q_lat（勿直接用 rom_q）
reg signed [15:0] bias_q_lat;

// fill_cnt / fill_search：須先於 ceb_opt、web_opt 宣告（Verilog 由上而下解析）
reg [13:0] fill_cnt;
wire [13:0] fill_off    = fill_cnt - SKIP_VALS;
wire [7:0]  fill_n      = fill_off[12:5];
wire [4:0]  fill_c      = fill_off[4:0];
wire        fill_search = (fill_cnt >= SKIP_VALS) &&
                          (fill_cnt <  TOT_VALS);
wire [12:0] opt_wr_comb = {fill_c, fill_n};

// Sram_opt CLK=~clk：寫入發生在 negedge，fill_cnt 不可與組合 opt_wr 同拍 posedge +1，
// 否則 negedge 時 fill_off 已 +1 → opt 特徵整體錯位（conv2/ tail 全偏、僅少數格 PASS）。
reg [12:0] opt_fill_a_r;
reg signed [15:0] opt_fill_d_r;
reg        opt_fill_we_r;
wire signed [15:0] opt_sram_d = opt_fill_we_r ? opt_fill_d_r : a_i;

// opt：僅在 conv1 讀取、或 S_FILL 的 search 段寫入時啟用；template 段不寫亦不讀
wire ceb_opt   = !(state==S_CONV1 || (state==S_FILL && (fill_search || opt_fill_we_r)));
wire ceb_sh1   = !(state==S_CONV1 || state==S_CONV2);
wire ceb_sh2   = !(state==S_CONV2 || state==S_CTR || state==S_SIZE || state==S_OFF);
wire ceb_score = !(state==S_CTR  || state==S_BBOX);
wire ceb_size  = !(state==S_SIZE || state==S_BBOX);
wire ceb_off   = !(state==S_OFF  || state==S_BBOX);

// Write enable: active (0) only on the bias-data cycle when writing result
// S_FILL：bb_y 串流為「template 2048 筆 + search 8192 筆」。僅 search 段可寫入 opt
//（與 numpy backbone_out[:, -256:, :] 一致）。若 template 段仍拉 web_opt，
// fill_off = fill_cnt - SKIP 無號下溢 → 亂寫 opt_buf，score 會全貼 sigmoid 下限。
wire web_opt   = !(state==S_FILL && opt_fill_we_r);
wire web_sh1   = !(state==S_CONV1 && mac_bp && mac_dv);
wire web_sh2   = !(state==S_CONV2 && mac_bp && mac_dv);
wire web_score = !(state==S_CTR  && mac_bp && mac_dv);
wire web_size  = !(state==S_SIZE && mac_bp && mac_dv);
wire web_off   = !(state==S_OFF  && mac_bp && mac_dv);

// ---------------------------------------------------------------------------
// Fill / streaming counters（fill_cnt / opt_wr 見上）
// ---------------------------------------------------------------------------
reg [10:0] bcnt;     // S_BBOX 串流索引 0..1279（對齊 cal_bbox 三階段共 1280 筆）
reg        bbox_dv;  // 1-cycle pipeline for bbox SRAM read

// ---------------------------------------------------------------------------
// Conv pixel / MAC counters
// ---------------------------------------------------------------------------
reg [6:0] cur_oc;
reg [3:0] cur_oh, cur_ow;
reg [6:0] mac_ic;
reg [1:0] mac_kh, mac_kw;
reg signed [47:0] mac_acc;

// ---------------------------------------------------------------------------
// Padding helper (5-bit unsigned wrap)
// ---------------------------------------------------------------------------
wire [4:0] ph  = {1'b0, cur_oh} + {1'b0, mac_kh} - 5'd1;
wire [4:0] pw  = {1'b0, cur_ow} + {1'b0, mac_kw} - 5'd1;
wire       pad = ph[4] | pw[4];
reg        pad_r;   // registered pad (valid when mac_dv=1)

// ---------------------------------------------------------------------------
// SRAM read addresses (combinational from registered counters)
// ---------------------------------------------------------------------------
wire [12:0] opt_rd  = {mac_ic[4:0], ph[3:0], pw[3:0]};
wire [14:0] sh1_rd  = {mac_ic[6:0], ph[3:0], pw[3:0]};
wire [13:0] sh2_rd  = {mac_ic[5:0], cur_oh,  cur_ow};

// SRAM address mux (read vs write)
wire [12:0] opt_sram_a  = (state==S_FILL) ? opt_fill_a_r : opt_rd;
wire [14:0] sh1_sram_a  = (state==S_CONV1 && mac_bp) ?
                           {cur_oc[6:0], cur_oh, cur_ow} : sh1_rd;
wire [13:0] sh2_sram_a  = (state==S_CONV2 && mac_bp) ?
                           {cur_oc[5:0], cur_oh, cur_ow} : sh2_rd;
// Pad to Sram_* macro address width (13b) to avoid PCWM lint
// S_BBOX：bcnt 0..255 讀 score；256..767 讀 size（位址 bcnt-256）；768..1279 讀 off（位址 bcnt-768）
wire [12:0] score_sram_a = {5'b0,
    (state==S_CTR) ? {cur_oh, cur_ow} : bcnt[7:0]};
wire [12:0] size_sram_a  = {4'b0,
    (state==S_SIZE && mac_bp) ? {cur_oc[0], cur_oh, cur_ow} :
    (state==S_BBOX && (bcnt >= BBOX_N_SCORE) &&
     (bcnt < (BBOX_N_SCORE + BBOX_N_SIZE))) ? (bcnt - BBOX_N_SCORE) :
    (state==S_BBOX) ? 9'd0 : bcnt[8:0]};
wire [12:0] off_sram_a   = {4'b0,
    (state==S_OFF && mac_bp) ? {cur_oc[0], cur_oh, cur_ow} :
    (state==S_BBOX && (bcnt >= (BBOX_N_SCORE + BBOX_N_SIZE)) &&
     (bcnt <= BBOX_STREAM_LAST)) ? (bcnt - (BBOX_N_SCORE + BBOX_N_SIZE)) :
    (state==S_BBOX) ? 9'd0 : bcnt[8:0]};

// ---------------------------------------------------------------------------
// SRAM Q outputs (registered on falling edge by macro)
// ---------------------------------------------------------------------------
wire signed [15:0] opt_q;
wire signed [15:0] sh1_q_lo, sh1_q_hi;
wire               sh1_hi_sel = sh1_sram_a[14];  // 1 when OC 64..95
wire signed [15:0] sh1_q      = sh1_hi_sel ? sh1_q_hi : sh1_q_lo;
wire signed [15:0] sh2_q;
wire signed [15:0] score_q;
wire signed [15:0] size_q;
wire signed [15:0] off_q;

// Feature mux（SRAM CLK=~clk）：mac_dv=0 拍設 opt_rd/sh* 位址；下一拍 mac_dv=1 時 opt_q 已對應該位址
// → 乘加直接用 feat_q（pad_r 為上一拍 pad），勿用 posedge/posedge-1 的 feat_q_lat（會錯拍）
wire signed [15:0] feat_raw =
    (state == S_CONV1) ? opt_q :
    (state == S_CONV2) ? sh1_q : sh2_q;
wire       feat_mac_st = (state == S_CONV1) || (state == S_CONV2) ||
                         (state == S_CTR)   || (state == S_SIZE) ||
                         (state == S_OFF);
wire       pad_at_addr = pad && (state != S_CTR) &&
                         (state != S_SIZE) && (state != S_OFF);
wire signed [15:0] feat_q = (pad_r && state != S_CTR &&
                              state != S_SIZE && state != S_OFF) ?
                              16'sd0 : feat_raw;
wire signed [15:0] feat_mac = (!mac_bp && mac_dv && feat_mac_st) ?
                              feat_q : 16'sd0;

// ---------------------------------------------------------------------------
// Weight-address shift-decomposition (multiplier-free)
//   288 = 256+32, 864 = 512+256+64+32, 9 = 8+1, 48 = 32+16
// ---------------------------------------------------------------------------
wire [19:0] OC288 = ({13'b0,cur_oc}<<8) + ({13'b0,cur_oc}<<5);
wire [19:0] OC864 = ({11'b0,cur_oc}<<9) + ({12'b0,cur_oc}<<8) +
                    ({14'b0,cur_oc}<<6) + ({15'b0,cur_oc}<<5);
wire [19:0] IC9   = ({17'b0,mac_ic}<<3) + {13'b0,mac_ic};
wire [19:0] KH3   = ({18'b0,mac_kh}<<1) + {18'b0,mac_kh};
wire [19:0] OC48  = ({15'b0,cur_oc}<<5) + ({16'b0,cur_oc}<<4);

wire [19:0] wa_c1_w = OC288 + IC9 + KH3 + {18'b0,mac_kw};
wire [19:0] wa_c2_w = OC864 + IC9 + KH3 + {18'b0,mac_kw};

// ---------------------------------------------------------------------------
// ROM address decoding
// ---------------------------------------------------------------------------
// conv1 weight split at 16384 (bit 14 of wa_c1_w)
wire        c1_use_w2 = wa_c1_w[14];
wire [13:0] c1w1_a    = wa_c1_w[13:0];
wire [13:0] c1w2_a    = wa_c1_w[13:0];  // offset from 16384 = lower 14 bits

// conv2 weight split at 16384/32768 (bits 14,15 of wa_c2_w)
wire        c2_use_w2 = wa_c2_w[14] & ~wa_c2_w[15];
wire        c2_use_w3 = wa_c2_w[15];
wire [13:0] c2w1_a    = wa_c2_w[13:0];
wire [13:0] c2w2_a    = wa_c2_w[13:0];
wire [13:0] c2w3_a    = wa_c2_w[13:0];

// conv1+conv2 bias ROM address
wire [7:0] c12b_a = (state==S_CONV1) ? {1'b0, cur_oc[6:0]} :
                    8'd96 + {2'b0, cur_oc[5:0]};

// tail combined weight ROM address
wire [7:0] csow_a =
    (state==S_CTR)  ? {2'b0, mac_ic[5:0]} :
    (state==S_OFF)  ? (8'd48  + (cur_oc[0] ? 8'd48 : 8'd0) + {2'b0, mac_ic[5:0]}) :
    (state==S_SIZE) ? (8'd144 + (cur_oc[0] ? 8'd48 : 8'd0) + {2'b0, mac_ic[5:0]}) :
                       8'd0;

// tail combined bias ROM address (pad to macro addr width)
wire [6:0] csob_a = {4'b0,
    (state==S_CTR)  ? 3'd0 :
    (state==S_OFF)  ? (cur_oc[0] ? 3'd2 : 3'd1) :
    (state==S_SIZE) ? (cur_oc[0] ? 3'd4 : 3'd3) :
                       3'd0};

// ROM chip-enables (active low; disable unused ROM for power saving)
wire ceb_c1w1  = !(state==S_CONV1 && !mac_bp && !c1_use_w2);
wire ceb_c1w2  = !(state==S_CONV1 && !mac_bp &&  c1_use_w2);
wire ceb_c12b  = !(state==S_CONV1 && mac_bp) && !(state==S_CONV2 && mac_bp);
wire ceb_c2w1  = !(state==S_CONV2 && !mac_bp && !c2_use_w2 && !c2_use_w3);
wire ceb_c2w2  = !(state==S_CONV2 && !mac_bp &&  c2_use_w2);
wire ceb_c2w3  = !(state==S_CONV2 && !mac_bp &&  c2_use_w3);
// tail 權重 ROM：mac_dv=0 設位址、mac_dv=1 讀 Q；勿在 mac_dv=1 關 ROM（否則 acc 恆為 0）
wire ceb_csow  = !(state==S_CTR || state==S_SIZE || state==S_OFF) || mac_bp;
wire ceb_csob  = !((state==S_CTR||state==S_SIZE||state==S_OFF) && mac_bp);

// ---------------------------------------------------------------------------
// ROM Q outputs
// ---------------------------------------------------------------------------
wire signed [15:0] c1w1_q, c1w2_q, c12b_q;
wire signed [15:0] c2w1_q, c2w2_q, c2w3_q;
wire signed [15:0] csow_q, csob_q;

// Weight ROM mux (select valid Q when mac_dv=1)
wire signed [15:0] conv1_wgt_q = c1_use_w2 ? c1w2_q : c1w1_q;
wire signed [15:0] conv2_wgt_q = c2_use_w3 ? c2w3_q :
                                  c2_use_w2 ? c2w2_q : c2w1_q;

// Combined weight/bias ROM Q for current state
wire signed [15:0] rom_q =
    (!mac_bp) ? (
        (state==S_CONV1) ? conv1_wgt_q :
        (state==S_CONV2) ? conv2_wgt_q :
                            csow_q
    ) : (
        (state==S_CONV1 || state==S_CONV2) ? c12b_q :
                                              csob_q
    );

// ---------------------------------------------------------------------------
// MAC product (Q8.8 × Q8.8 → Q16.16 in 32-bit)
// ---------------------------------------------------------------------------
wire signed [31:0] mac_prod = feat_mac * rom_q;

// ---------------------------------------------------------------------------
// Bias-phase combinational outputs
// ---------------------------------------------------------------------------
wire signed [47:0] mac_sh  = $signed(mac_acc) >>> 8;
wire signed [47:0] mac_bi  = mac_sh + {{32{bias_q_lat[15]}}, bias_q_lat};
// Q8.8 飽和：對完整 48-bit mac_bi 比較（對齊 numpy acc>>8+bias 後再 fp）
wire signed [47:0] mac_bi_s = $signed(mac_bi);
wire signed [15:0] mac_cl  =
    (mac_bi_s > $signed(48'sh7FFF)) ? 16'sh7FFF :
    (mac_bi_s < $signed(48'sh8000)) ? -16'sh8000 :
    mac_bi_s[15:0];
wire signed [15:0] mac_relu = mac_cl[15] ? 16'sd0 : mac_cl;

wire [7:0] sig_out;
sigmoid_lut u_sig (.x_i(mac_cl), .y_o(sig_out));
wire signed [15:0] mac_sig = {8'b0, sig_out};

// ---------------------------------------------------------------------------
// End-of-inner-loop conditions
// ---------------------------------------------------------------------------
wire mac_end_3x3 = (mac_ic==IN_CH_M1) && (mac_kh==2'd2) && (mac_kw==2'd2);
wire mac_end_c2  = (mac_ic==C_SH1_M1) && (mac_kh==2'd2) && (mac_kw==2'd2);
wire mac_end_1x1 = (mac_ic==C_SH2_M1);

wire all_done_c1  = (cur_oc==C_SH1_M1) && (cur_oh==FEAT_H_M1) &&
                    (cur_ow==FEAT_W_M1);
wire all_done_c2  = (cur_oc==C_SH2_M1) && (cur_oh==FEAT_H_M1) &&
                    (cur_ow==FEAT_W_M1);
wire all_done_ctr = (cur_oh==FEAT_H_M1) && (cur_ow==FEAT_W_M1);
wire all_done_2ch = (cur_oc==7'd1) && (cur_oh==FEAT_H_M1) &&
                    (cur_ow==FEAT_W_M1);

// ---------------------------------------------------------------------------
// S_BBOX → cal_bbox：SRAM macro CLK=~clk；本拍採樣 *_q 餵資料。
// score/size/offset 三路 valid 依 bcnt 分相（僅一路為 1），避免在 S_ARGMAX 仍對 size/off
// 送 valid=1 或相位錯亂時，模擬或下游對未用匯流排出現 X 傳播。
//
// 重要：bcnt 不可每拍自走。cal_bbox 只在 *_valid 為 1 時才在該相內遞增 cnt；若 head
// 較快會先進入 offset 相而 cal_bbox 仍卡在 S_SIZE，則 size_valid 永遠等不到足夠拍數
//（u_bbox.state 卡在 2、bcnt 已到 1279、sc_q/sz_q 在未讀 SRAM 相呈 X）。因此 bcnt
// 僅在「本拍送出的相」與 u_bbox.state 一致時才遞增；等候下游時凍結 bcnt 並維持當相
// valid 與 bbox_cal_*（與 cal_bbox.v 狀態碼一致）。
// ---------------------------------------------------------------------------
localparam CB_ARGMAX   = 3'd1;
localparam CB_SIZE     = 3'd2;
localparam CB_OFFSET   = 3'd3;

reg signed [15:0] bbox_cal_sc, bbox_cal_sz, bbox_cal_of;
reg               bbox_sc_valid, bbox_sz_valid, bbox_of_valid;

// ---------------------------------------------------------------------------
// cal_bbox
// ---------------------------------------------------------------------------
reg  bbox_start;
wire bbox_busy, bbox_done;

cal_bbox #(.FEAT_SZ(FEAT_H), .FEAT_LEN(FEAT_SZ)) u_bbox (
    .clk(clk), .reset(reset), .start(bbox_start),
    .score_i(bbox_cal_sc),  .score_valid (bbox_sc_valid),
    .size_i (bbox_cal_sz), .size_valid  (bbox_sz_valid),
    .offset_i(bbox_cal_of), .offset_valid(bbox_of_valid),
    .busy(bbox_busy), .done(bbox_done),
    .cx_o(cx_o), .cy_o(cy_o), .w_o(w_o), .h_o(h_o)
);

// S_BBOX 串流與 cal_bbox 相位對齊（須在 u_bbox 之後）
wire [2:0] bbox_u_st = u_bbox.state;
wire       bbox_phase_score = (bcnt <  BBOX_N_SCORE);
wire       bbox_phase_size  = (bcnt >= BBOX_N_SCORE) &&
                              (bcnt < (BBOX_N_SCORE + BBOX_N_SIZE));
wire       bbox_phase_off   = (bcnt >= (BBOX_N_SCORE + BBOX_N_SIZE)) &&
                              (bcnt <= BBOX_STREAM_LAST);
wire       bbox_stream_advance = (
    (bbox_phase_score && (bbox_u_st == CB_ARGMAX))  ||
    (bbox_phase_size  && (bbox_u_st == CB_SIZE))   ||
    (bbox_phase_off   && (bbox_u_st == CB_OFFSET))
);

// ---------------------------------------------------------------------------
// SRAM instances (CLK=~clk, 16-bit data width, BWEB=16'b0)
// ---------------------------------------------------------------------------
Sram_opt u_Sram_opt (
    .SLP(pgen_opt), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_opt), .WEB(web_opt),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(opt_sram_a), .D(opt_sram_d), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(opt_q));

// Sram_sh1_lo: 16384×16-bit  OC 0..63  CM=16  A[13:0]
Sram_sh1_lo u_Sram_sh1_lo (
    .SLP(pgen_sh1), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk),
    .CEB(ceb_sh1 | sh1_hi_sel),
    .WEB(web_sh1 | sh1_hi_sel),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(sh1_sram_a[13:0]), .D(sram_wdata_r), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(sh1_q_lo));

// Sram_sh1_hi: 8192×16-bit  OC 64..95  CM=8  A[12:0]
Sram_sh1_hi u_Sram_sh1_hi (
    .SLP(pgen_sh1), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk),
    .CEB(ceb_sh1 | ~sh1_hi_sel),
    .WEB(web_sh1 | ~sh1_hi_sel),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(sh1_sram_a[12:0]), .D(sram_wdata_r), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(sh1_q_hi));

Sram_sh2 u_Sram_sh2 (
    .SLP(pgen_sh2), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_sh2), .WEB(web_sh2),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(sh2_sram_a), .D(sram_wdata_r), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(sh2_q));

Sram_score u_Sram_score (
    .SLP(pgen_score), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_score), .WEB(web_score),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(score_sram_a), .D(sram_wdata_r), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(score_q));

Sram_size u_Sram_size (
    .SLP(pgen_size), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_size), .WEB(web_size),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(size_sram_a), .D(sram_wdata_r), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(size_q));

Sram_off u_Sram_off (
    .SLP(pgen_off), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_off), .WEB(web_off),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(off_sram_a), .D(sram_wdata_r), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(off_q));

// ---------------------------------------------------------------------------
// ROM instances (CLK=~clk, read-only)
// ---------------------------------------------------------------------------
rom_box_head_shared_conv1_folded_weight1 u_rom_c1w1 (
    .A(c1w1_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_c1w1), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(c1w1_q));

rom_box_head_shared_conv1_folded_weight2 u_rom_c1w2 (
    .A(c1w2_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_c1w2), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(c1w2_q));

rom_box_head_shared_conv1_2_folded_bias u_rom_c12b (
    .A(c12b_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_c12b), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(c12b_q));

rom_box_head_shared_conv2_folded_weight1 u_rom_c2w1 (
    .A(c2w1_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_c2w1), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(c2w1_q));

rom_box_head_shared_conv2_folded_weight2 u_rom_c2w2 (
    .A(c2w2_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_c2w2), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(c2w2_q));

rom_box_head_shared_conv2_folded_weight3 u_rom_c2w3 (
    .A(c2w3_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_c2w3), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(c2w3_q));

rom_box_head_tail_ctr_offset_size_weight u_rom_csow (
    .A(csow_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_csow), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(csow_q));

rom_box_head_tail_ctr_offset_size_bias u_rom_csob (
    .A(csob_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(ceb_csob), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(csob_q));

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:  next_state = start ? S_FILL : S_IDLE;
        S_FILL:  next_state = (a_valid && fill_cnt==TOT_VALS_M1) ?
                               S_CONV1 : S_FILL;
        S_CONV1: next_state = (mac_bp && mac_dv && all_done_c1) ?
                               S_CONV2 : S_CONV1;
        S_CONV2: next_state = (mac_bp && mac_dv && all_done_c2) ?
                               S_CTR   : S_CONV2;
        S_CTR:   next_state = (mac_bp && mac_dv && all_done_ctr) ?
                               S_SIZE  : S_CTR;
        S_SIZE:  next_state = (mac_bp && mac_dv && all_done_2ch) ?
                               S_OFF   : S_SIZE;
        S_OFF:   next_state = (mac_bp && mac_dv && all_done_2ch) ?
                               S_BBOX  : S_OFF;
        S_BBOX:  next_state = bbox_done ? S_DONE : S_BBOX;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: power-gating control
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        pgen_opt   <= 1'b0;
        pgen_sh1   <= 1'b0;
        pgen_sh2   <= 1'b0;
        pgen_score <= 1'b0;
        pgen_size  <= 1'b0;
        pgen_off   <= 1'b0;
    end else begin
        pgen_opt   <= !(state==S_CONV1 || (state==S_FILL && fill_search));
        pgen_sh1   <= !(state==S_CONV1  || state==S_CONV2);
        pgen_sh2   <= !(state==S_CONV2  || state==S_CTR ||
                        state==S_SIZE   || state==S_OFF);
        // score / size / off 三顆 map SRAM：在 tail 寫入完成後到 S_BBOX 讀串流前，
        // 不可僅在「自己的寫入 state」才解除 SLP。部分 SRAM macro 在 SLP=1 時
        // Q 會變 X，若鄰近 state（例如 score 在 S_SIZE/S_OFF 仍睡）進入 S_BBOX
        // 後 sc_q/sz_q 會一直為 xxxx；off 僅在 S_OFF 才睡，故常見 of_q 仍正常。
        // 對齊實際使用區間：自 S_CTR 起至 S_BBOX 結束，三顆一併保持喚醒（低功耗
        // 可在整段 head 完成後於 S_DONE 再睡）。
        pgen_score <= !(state==S_CTR || state==S_SIZE ||
                        state==S_OFF  || state==S_BBOX);
        pgen_size  <= !(state==S_CTR || state==S_SIZE ||
                        state==S_OFF  || state==S_BBOX);
        pgen_off   <= !(state==S_CTR || state==S_SIZE ||
                        state==S_OFF  || state==S_BBOX);
    end
end

// ---------------------------------------------------------------------------
// FSM segment 4: sequential datapath
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    done       <= 1'b0;
    bbox_start <= 1'b0;
    bbox_sc_valid <= 1'b0;
    bbox_sz_valid <= 1'b0;
    bbox_of_valid <= 1'b0;
    pad_r <= pad;
    opt_fill_we_r <= 1'b0;

    if (reset) begin
        opt_fill_a_r <= 13'd0;
        opt_fill_d_r <= 16'sd0;
`ifdef DUMP_HEAD_SCORE_DEEP
        dbg_opt_in_logged <= 3'b0;
`endif
        fill_cnt <= 14'd0;
        cur_oc   <= 7'd0;  cur_oh  <= 4'd0;  cur_ow  <= 4'd0;
        mac_ic   <= 7'd0;  mac_kh  <= 2'd0;  mac_kw  <= 2'd0;
        mac_acc  <= 48'sd0;
        mac_bp   <= 1'b0;  mac_dv  <= 1'b0;
        sram_wdata_r <= 16'sd0;
        bias_q_lat   <= 16'sd0;
        bcnt     <= 11'd0;  bbox_dv <= 1'b0;
    end else begin
        // bias 當拍：posedge 末 mac_acc<=0 會把 mac_relu/mac_sig 拉成 0；negedge SRAM 寫 sram_wdata_r
        if (mac_bp && mac_dv) begin
            case (state)
                S_CONV1, S_CONV2: sram_wdata_r = mac_relu;
                S_CTR, S_SIZE:    sram_wdata_r = mac_sig;
                S_OFF:            sram_wdata_r = mac_cl;
                default:          sram_wdata_r = 16'sd0;
            endcase
        end

        case (state)

        // ----------------------------------------------------------------
        S_IDLE: begin
            fill_cnt <= 14'd0;
            cur_oc   <= 7'd0;  cur_oh  <= 4'd0;  cur_ow  <= 4'd0;
            mac_ic   <= 7'd0;  mac_kh  <= 2'd0;  mac_kw  <= 2'd0;
            mac_acc  <= 48'sd0;
            mac_bp   <= 1'b0;  mac_dv  <= 1'b0;
            bcnt     <= 11'd0;  bbox_dv <= 1'b0;
        end

        // ----------------------------------------------------------------
        // Write search tokens into opt SRAM (template tokens discarded)
        S_FILL: begin
            if (a_valid) begin
                // 鎖存本拍 fill_cnt 對應位址，供同拍 negedge SRAM 寫入（見 opt_fill_* 註解）
                if (fill_search) begin
                    opt_fill_a_r  <= opt_wr_comb;
                    opt_fill_d_r  <= a_i;
                    opt_fill_we_r <= 1'b1;
                end
                if (fill_cnt < TOT_VALS)
                    fill_cnt <= fill_cnt + 14'd1;
            end
        end

        // ----------------------------------------------------------------
        // shared_conv1: 3×3, IN_CH→C_SH1, ReLU; mac_dv pipelines SRAM latency
        S_CONV1: begin
            if (!mac_bp) begin
                if (mac_dv) begin
`ifdef DUMP_HEAD_SCORE_DEEP
                    // 每錨點僅在 cur_oc==0 列印一次（避免 96 個 output channel 重複洗版）
                    if ((cur_oc == 7'd0) &&
                        (mac_ic == 7'd0) && (mac_kh == 2'd1) && (mac_kw == 2'd1) &&
                        (({cur_oh, cur_ow} == 8'd3   && !dbg_opt_in_logged[0]) ||
                         ({cur_oh, cur_ow} == 8'd135 && !dbg_opt_in_logged[1]) ||
                         ({cur_oh, cur_ow} == 8'd136 && !dbg_opt_in_logged[2]))) begin
                        begin : opt_in_golden_cmp
                            reg [15:0] g_opt;
                            integer  d_opt;
                            if ({cur_oh, cur_ow} == 8'd3)   dbg_opt_in_logged[0] <= 1'b1;
                            if ({cur_oh, cur_ow} == 8'd135) dbg_opt_in_logged[1] <= 1'b1;
                            if ({cur_oh, cur_ow} == 8'd136) dbg_opt_in_logged[2] <= 1'b1;
                            g_opt = opt_golden_mem[{mac_ic[4:0], ph[3:0], pw[3:0]}];
                            d_opt = $signed(feat_mac) - $signed(g_opt);
                            if ((d_opt > `HEAD_GOLDEN_TOL_LSB) ||
                                (d_opt < -`HEAD_GOLDEN_TOL_LSB))
                                $display(
                                    "[HEAD_GOLDEN_CMP] opt_in [FAIL] idx=%0d opt_a=%h rtl=%h golden=%h delta=%0d",
                                    {ph[3:0], pw[3:0]}, opt_rd, feat_mac, g_opt, d_opt);
                            else
                                $display(
                                    "[HEAD_GOLDEN_CMP] opt_in [PASS] idx=%0d opt_a=%h rtl=%h golden=%h",
                                    {ph[3:0], pw[3:0]}, opt_rd, feat_mac, g_opt);
                        end
                    end
`endif
                    mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
`ifdef DUMP_HEAD_FEAT_TAP
                    if ((cur_oc == 7'd0) && (mac_ic == 7'd0) &&
                        (mac_kh == 2'd1) && (mac_kw == 2'd1) &&
                        (({cur_oh, cur_ow} == 8'd3) ||
                         ({cur_oh, cur_ow} == 8'd135) ||
                         ({cur_oh, cur_ow} == 8'd136)))
                        $display(
                            "[HEAD_FEAT_TAP] S_CONV1 mac_dv=1 opt_rd=%h feat_raw=%h feat_q=%h rom_q=%h mac_prod=%h",
                            opt_rd, feat_raw, feat_q, rom_q, mac_prod[15:0]);
`endif
                    if (mac_end_3x3) begin
                        mac_bp <= 1'b1;
                        mac_dv <= 1'b0;
                    end else begin
                        if (mac_kw==2'd2) begin
                            mac_kw <= 2'd0;
                            if (mac_kh==2'd2) begin
                                mac_kh <= 2'd0;
                                mac_ic <= mac_ic + 7'd1;
                            end else mac_kh <= mac_kh + 2'd1;
                        end else mac_kw <= mac_kw + 2'd1;
                        mac_dv <= 1'b0;
                    end
                end else mac_dv <= 1'b1;
            end else begin
                // Bias phase: mac_dv=0→wait ROM, mac_dv=1→apply+write+advance
                if (mac_dv) begin
`ifdef DUMP_HEAD_SCORE_CHECK
                    if (`HEAD_SCORE_ANCHOR_IDX({cur_oh, cur_ow}) && (cur_oc == 7'd0)) begin
                        $display(
                            "[HEAD_SCORE_CHK] S_CONV1 wr sh1 oc=%0d oh=%0d ow=%0d idx=%0d mac_relu=%h",
                            cur_oc, cur_oh, cur_ow, {cur_oh, cur_ow}, mac_relu);
`ifdef DUMP_HEAD_SCORE_DEEP
                        begin : conv1_golden_cmp
                            reg [15:0] g_c1;
                            integer  d_c1;
                            g_c1 = conv1_golden_mem[{cur_oc, cur_oh, cur_ow}];
                            d_c1 = $signed(sram_wdata_r) - $signed(g_c1);
                            if ((d_c1 > `HEAD_GOLDEN_TOL_LSB) ||
                                (d_c1 < -`HEAD_GOLDEN_TOL_LSB))
                                $display(
                                    "[HEAD_GOLDEN_CMP] S_CONV1 [FAIL] oc=%0d idx=%0d rtl_wr=%h golden=%h delta=%0d",
                                    cur_oc, {cur_oh, cur_ow}, sram_wdata_r, g_c1, d_c1);
                            else
                                $display(
                                    "[HEAD_GOLDEN_CMP] S_CONV1 [PASS] oc=%0d idx=%0d rtl_wr=%h golden=%h",
                                    cur_oc, {cur_oh, cur_ow}, sram_wdata_r, g_c1);
                        end
                        $display(
                            "[HEAD_MAC_DBG] S_CONV1 bias oc=%0d idx=%0d mac_acc=%h mac_sh=%h c12b_a=%h rom_q=%h bias_lat=%h mac_bi48=%h mac_cl=%h comb_relu=%h sram_wdata=%h golden=%h",
                            cur_oc, {cur_oh, cur_ow}, mac_acc, mac_sh[15:0], c12b_a,
                            rom_q, bias_q_lat, mac_bi_s, mac_cl, mac_relu, sram_wdata_r,
                            conv1_golden_mem[{cur_oc, cur_oh, cur_ow}]);
                        $display(
                            "[HEAD_BIAS_WR] S_CONV1 idx=%0d comb_relu=%h sram_wdata=%h",
                            {cur_oh, cur_ow}, mac_relu, sram_wdata_r);
`endif
                    end
`endif
                    // sh1 SRAM write occurs via web_sh1/sh1_sram_a wires
                    mac_acc <= 48'sd0;
                    mac_bp  <= 1'b0;
                    mac_dv  <= 1'b0;
                    mac_ic  <= 7'd0; mac_kh <= 2'd0; mac_kw <= 2'd0;
                    if (cur_ow==FEAT_W_M1) begin
                        cur_ow <= 4'd0;
                        if (cur_oh==FEAT_H_M1) begin
                            cur_oh <= 4'd0;
                            if (cur_oc!=C_SH1_M1) cur_oc <= cur_oc + 7'd1;
                        end else cur_oh <= cur_oh + 4'd1;
                    end else cur_ow <= cur_ow + 4'd1;
                end else mac_dv <= 1'b1;
            end
        end

        // ----------------------------------------------------------------
        // shared_conv2: 3×3, C_SH1→C_SH2, ReLU
        S_CONV2: begin
            if (!mac_bp) begin
                if (mac_dv) begin
                    mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                    if (mac_end_c2) begin
                        mac_bp <= 1'b1;
                        mac_dv <= 1'b0;
                    end else begin
                        if (mac_kw==2'd2) begin
                            mac_kw <= 2'd0;
                            if (mac_kh==2'd2) begin
                                mac_kh <= 2'd0;
                                mac_ic <= mac_ic + 7'd1;
                            end else mac_kh <= mac_kh + 2'd1;
                        end else mac_kw <= mac_kw + 2'd1;
                        mac_dv <= 1'b0;
                    end
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
`ifdef DUMP_HEAD_SCORE_CHECK
                    if (`HEAD_SCORE_ANCHOR_IDX({cur_oh, cur_ow}) &&
                        `HEAD_SCORE_CONV2_DUMP_OC(cur_oc, C_SH2_M1, {cur_oh, cur_ow})) begin
                        $display(
                            "[HEAD_SCORE_CHK] S_CONV2 wr sh2 oc=%0d oh=%0d ow=%0d idx=%0d mac_relu=%h sh2_a=%h",
                            cur_oc, cur_oh, cur_ow, {cur_oh, cur_ow}, mac_relu, sh2_sram_a);
`ifdef DUMP_HEAD_SCORE_DEEP
                        begin : conv2_golden_cmp
                            reg [15:0] g_c2;
                            integer  d_c2;
                            g_c2 = conv2_golden_mem[{cur_oc, cur_oh, cur_ow}];
                            d_c2 = $signed(sram_wdata_r) - $signed(g_c2);
                            if ((d_c2 > `HEAD_GOLDEN_TOL_LSB) ||
                                (d_c2 < -`HEAD_GOLDEN_TOL_LSB))
                                $display(
                                    "[HEAD_GOLDEN_CMP] S_CONV2 [FAIL] oc=%0d idx=%0d rtl_wr=%h golden=%h delta=%0d",
                                    cur_oc, {cur_oh, cur_ow}, sram_wdata_r, g_c2, d_c2);
                            else
                                $display(
                                    "[HEAD_GOLDEN_CMP] S_CONV2 [PASS] oc=%0d idx=%0d rtl_wr=%h golden=%h",
                                    cur_oc, {cur_oh, cur_ow}, sram_wdata_r, g_c2);
                        end
`endif
                    end
`endif
                    mac_acc <= 48'sd0;
                    mac_bp  <= 1'b0;
                    mac_dv  <= 1'b0;
                    mac_ic  <= 7'd0; mac_kh <= 2'd0; mac_kw <= 2'd0;
                    if (cur_ow==FEAT_W_M1) begin
                        cur_ow <= 4'd0;
                        if (cur_oh==FEAT_H_M1) begin
                            cur_oh <= 4'd0;
                            if (cur_oc!=C_SH2_M1) cur_oc <= cur_oc + 7'd1;
                        end else cur_oh <= cur_oh + 4'd1;
                    end else cur_ow <= cur_ow + 4'd1;
                end else mac_dv <= 1'b1;
            end
        end

        // ----------------------------------------------------------------
        // tail_ctr: 1×1, C_SH2→1, sigmoid
        S_CTR: begin
            if (!mac_bp) begin
                if (mac_dv) begin
                    mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                    if (mac_end_1x1) begin
                        mac_bp <= 1'b1;
                        mac_dv <= 1'b0;
                    end else begin
                        mac_ic <= mac_ic + 7'd1;
                        mac_dv <= 1'b0;
                    end
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
`ifdef DUMP_HEAD_SCORE_CHECK
                    if (`HEAD_SCORE_ANCHOR_IDX({cur_oh, cur_ow})) begin
                        $display(
                            "[HEAD_SCORE_CHK] S_CTR wr score oh=%0d ow=%0d idx=%0d sc_a=%h | mac_sh=%h mac_cl=%h sig=%02h mac_sig=%h | feat_sh2=%h",
                            cur_oh, cur_ow, {cur_oh, cur_ow}, score_sram_a[7:0],
                            mac_sh[15:0], mac_cl, sig_out, mac_sig, feat_q);
`ifdef DUMP_HEAD_SCORE_DEEP
                        begin : ctr_raw_golden_cmp
                            reg [15:0] g_raw;
                            integer  d_raw;
                            g_raw = raw_ctr_golden_mem[{cur_oh, cur_ow}];
                            d_raw = $signed(mac_cl) - $signed(g_raw);
                            if ((d_raw > `HEAD_GOLDEN_TOL_LSB) ||
                                (d_raw < -`HEAD_GOLDEN_TOL_LSB))
                                $display(
                                    "[HEAD_GOLDEN_CMP] S_CTR raw [FAIL] idx=%0d rtl_mac_cl=%h golden=%h delta=%0d (sigmoid前)",
                                    {cur_oh, cur_ow}, mac_cl, g_raw, d_raw);
                            else
                                $display(
                                    "[HEAD_GOLDEN_CMP] S_CTR raw [PASS] idx=%0d rtl_mac_cl=%h golden=%h",
                                    {cur_oh, cur_ow}, mac_cl, g_raw);
                            $display(
                                "[HEAD_MAC_DBG] S_CTR bias idx=%0d mac_acc=%h mac_sh=%h csob_a=%h rom_q=%h bias_lat=%h mac_bi48=%h mac_cl=%h sig=%02h comb_sig=%h sram_wdata=%h golden_raw=%h",
                                {cur_oh, cur_ow}, mac_acc, mac_sh[15:0], csob_a, rom_q,
                                bias_q_lat, mac_bi_s, mac_cl, sig_out, mac_sig, sram_wdata_r,
                                raw_ctr_golden_mem[{cur_oh, cur_ow}]);
                            $display(
                                "[HEAD_BIAS_WR] S_CTR idx=%0d comb_sig=%h sram_wdata=%h feat_sh2=%h",
                                {cur_oh, cur_ow}, mac_sig, sram_wdata_r, feat_q);
`ifdef DUMP_HEAD_FEAT_TAP
                            if ((mac_ic == 7'd0) && (mac_kh == 2'd1) && (mac_kw == 2'd1) &&
                                (({cur_oh, cur_ow} == 8'd3) ||
                                 ({cur_oh, cur_ow} == 8'd135) ||
                                 ({cur_oh, cur_ow} == 8'd136)))
                                $display(
                                    "[HEAD_FEAT_TAP] S_CTR mac_dv=1 opt_rd=%h feat_raw=%h feat_q=%h rom_q=%h mac_prod=%h",
                                    opt_rd, feat_raw, feat_q, rom_q, mac_prod[15:0]);
`endif
                        end
`endif
                    end
`endif
                    // score SRAM write via web_score/score_sram_a wires
                    mac_acc <= 48'sd0;
                    mac_bp  <= 1'b0;
                    mac_dv  <= 1'b0;
                    mac_ic  <= 7'd0;
                    if (cur_ow==FEAT_W_M1) begin
                        cur_ow <= 4'd0;
                        if (cur_oh!=FEAT_H_M1) cur_oh <= cur_oh + 4'd1;
                    end else cur_ow <= cur_ow + 4'd1;
                end else mac_dv <= 1'b1;
            end
        end

        // ----------------------------------------------------------------
        // tail_size: 1×1, C_SH2→2, sigmoid
        S_SIZE: begin
            if (!mac_bp) begin
                if (mac_dv) begin
                    mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                    if (mac_end_1x1) begin
                        mac_bp <= 1'b1;
                        mac_dv <= 1'b0;
                    end else begin
                        mac_ic <= mac_ic + 7'd1;
                        mac_dv <= 1'b0;
                    end
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
`ifdef DUMP_HEAD_SIZE_SAT
                    // 線性序 oc*256+oh*16+ow，與 size_sram_a 展開一致（僅除錯）
                    if (({1'b0, cur_oc[0], cur_oh, cur_ow} % (`HEAD_SIZE_DUMP_STRIDE)) == 9'd0)
                        $display(
                            "[HEAD_SIZE_SAT] t=%0t oh=%0d ow=%0d cur_oc=%0d mac_cl=%h sig_out=%02h sz_a=%h",
                            $time, cur_oh, cur_ow, cur_oc,
                            mac_cl, sig_out, size_sram_a);
`endif
                    mac_acc <= 48'sd0;
                    mac_bp  <= 1'b0;
                    mac_dv  <= 1'b0;
                    mac_ic  <= 7'd0;
                    if (cur_ow==FEAT_W_M1) begin
                        cur_ow <= 4'd0;
                        if (cur_oh==FEAT_H_M1) begin
                            cur_oh <= 4'd0;
                            if (cur_oc!=7'd1) cur_oc <= cur_oc + 7'd1;
                        end else cur_oh <= cur_oh + 4'd1;
                    end else cur_ow <= cur_ow + 4'd1;
                end else mac_dv <= 1'b1;
            end
        end

        // ----------------------------------------------------------------
        // tail_off: 1×1, C_SH2→2, linear
        S_OFF: begin
            if (!mac_bp) begin
                if (mac_dv) begin
                    mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
                    if (mac_end_1x1) begin
                        mac_bp <= 1'b1;
                        mac_dv <= 1'b0;
                    end else begin
                        mac_ic <= mac_ic + 7'd1;
                        mac_dv <= 1'b0;
                    end
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
                    // off SRAM write via web_off/off_sram_a wires
                    mac_acc <= 48'sd0;
                    mac_bp  <= 1'b0;
                    mac_dv  <= 1'b0;
                    mac_ic  <= 7'd0;
                    if (cur_ow==FEAT_W_M1) begin
                        cur_ow <= 4'd0;
                        if (cur_oh==FEAT_H_M1) begin
                            cur_oh <= 4'd0;
                            if (cur_oc!=7'd1) cur_oc <= cur_oc + 7'd1;
                        end else cur_oh <= cur_oh + 4'd1;
                    end else cur_ow <= cur_ow + 4'd1;
                end else mac_dv <= 1'b1;
            end
        end

        // ----------------------------------------------------------------
        // Stream score/size/off SRAM to cal_bbox (1-cycle read latency)
        S_BBOX: begin
            // bbox_start 僅在 S_OFF→S_BBOX 轉換時脈衝一次（見下方 next_state 區塊）。
            // 不可每拍 !bbox_busy 再 start：cal_bbox 完成回 IDLE 後會二次 start，
            // 清掉 argmax_idx_snap，TB 在 top done 讀到 0／0x8000。
            if (bbox_busy) begin
                bbox_dv <= 1'b1;
                if (bbox_dv) begin
`ifdef DUMP_HEAD_SCORE_CHECK
                    if (`HEAD_SCORE_ANCHOR_IDX(bcnt[7:0]))
                        $display(
                            "[HEAD_SCORE_CHK] S_BBOX bcnt=%0d idx=%0d sc_a=%h sc_q=%h (feed cal_bbox)",
                            bcnt, bcnt[7:0], score_sram_a[7:0], score_q);
`endif
                    // 採樣的 *_q 對應上一 negedge 依目前 bcnt 讀出的資料（見模組頭註解）
                    bbox_cal_sc <= score_q;
                    bbox_cal_sz <= size_q;
                    bbox_cal_of <= off_q;
                    bbox_sc_valid  <= bbox_phase_score && (bbox_u_st == CB_ARGMAX);
                    bbox_sz_valid  <= bbox_phase_size  && (bbox_u_st == CB_SIZE);
                    bbox_of_valid  <= bbox_phase_off   && (bbox_u_st == CB_OFFSET);
                    if (bbox_stream_advance && (bcnt < BBOX_STREAM_LAST))
                        bcnt <= bcnt + 11'd1;
                end
            end
        end

        S_DONE: begin
            done <= 1'b1;
        end

        default: ;
        endcase

        // ----------------------------------------------------------------
        // Stage-entry counter reset on state transition
        // ----------------------------------------------------------------
        if (next_state==S_CONV1 && state!=S_CONV1) begin
            cur_oc<=7'd0; cur_oh<=4'd0; cur_ow<=4'd0;
            mac_ic<=7'd0; mac_kh<=2'd0; mac_kw<=2'd0;
            mac_acc<=48'sd0; mac_bp<=1'b0; mac_dv<=1'b0;
        end
        if (next_state==S_CONV2 && state!=S_CONV2) begin
            cur_oc<=7'd0; cur_oh<=4'd0; cur_ow<=4'd0;
            mac_ic<=7'd0; mac_kh<=2'd0; mac_kw<=2'd0;
            mac_acc<=48'sd0; mac_bp<=1'b0; mac_dv<=1'b0;
        end
        if (next_state==S_CTR && state!=S_CTR) begin
            cur_oc<=7'd0; cur_oh<=4'd0; cur_ow<=4'd0;
            mac_ic<=7'd0; mac_acc<=48'sd0; mac_bp<=1'b0; mac_dv<=1'b0;
        end
        if (next_state==S_SIZE && state!=S_SIZE) begin
            cur_oc<=7'd0; cur_oh<=4'd0; cur_ow<=4'd0;
            mac_ic<=7'd0; mac_acc<=48'sd0; mac_bp<=1'b0; mac_dv<=1'b0;
        end
        if (next_state==S_OFF && state!=S_OFF) begin
            cur_oc<=7'd0; cur_oh<=4'd0; cur_ow<=4'd0;
            mac_ic<=7'd0; mac_acc<=48'sd0; mac_bp<=1'b0; mac_dv<=1'b0;
        end
        if (next_state == S_BBOX && state == S_OFF) begin
            bbox_start <= 1'b1;
            bcnt       <= 11'd0;
            bbox_dv    <= 1'b0;
        end
    end
end

`ifdef DUMP_HEAD_FSM
// ---------------------------------------------------------------------------
// S_BBOX (state=7) heartbeat：只要 clk 在跑且 state 仍為 7，hb 每拍 +1 並定期 $display
// ---------------------------------------------------------------------------
reg [31:0] dbg_sbbox_hb;

always @(posedge clk) begin
    if (reset || (state != S_BBOX))
        dbg_sbbox_hb <= 32'd0;
    else
        dbg_sbbox_hb <= dbg_sbbox_hb + 32'd1;
end

always @(posedge clk) begin
    if (!reset) begin
        if (state == S_OFF && next_state == S_BBOX)
            $display("[HEAD] -> enter S_BBOX(state=7) BBOX_STREAM_LAST=%0d", BBOX_STREAM_LAST);
`ifdef DUMP_HEAD_S_BBOX_EACH_CLK
        if (state == S_BBOX)
            $display("[HEAD] S_BBOX hb=%0d t=%0t bcnt=%0d bbox_dv=%0b busy=%0b done=%0b u_st=%0d",
                     dbg_sbbox_hb, $time, bcnt, bbox_dv, bbox_busy, bbox_done, u_bbox.state);
`else
        if (state == S_BBOX && (dbg_sbbox_hb[3:0] == 4'd0))
            $display("[HEAD] S_BBOX hb=%0d t=%0t bcnt=%0d bbox_dv=%0b busy=%0b done=%0b u_st=%0d",
                     dbg_sbbox_hb, $time, bcnt, bbox_dv, bbox_busy, bbox_done, u_bbox.state);
`endif
        if (state == S_BBOX && next_state == S_DONE)
            $display("[HEAD] <- leave S_BBOX(state=7) bbox_done=%0b bcnt=%0d hb=%0d",
                     bbox_done, bcnt, dbg_sbbox_hb);
    end
end
`endif

`ifdef DUMP_HEAD_BBOX_STREAM
// ---------------------------------------------------------------------------
// S_BBOX 串流採樣：bcnt 為「本拍餵給 cal 的索引」（與 score/size/off_sram_a 對齊）
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!reset && (state == S_BBOX) && bbox_busy && bbox_dv) begin
        if ((bcnt[5:0] == 6'd0) || (bcnt == BBOX_N_SCORE) ||
            (bcnt == (BBOX_N_SCORE + BBOX_N_SIZE)) || (bcnt == BBOX_STREAM_LAST)) begin
            $display(
                "[HEAD_BBOX] t=%0t bcnt=%0d sc_a=%h sz_a=%h of_a=%h | sc_q=%h sz_q=%h of_q=%h | ph(sc,sz,of)=(%b,%b,%b) u_st=%0d",
                $time, bcnt, score_sram_a, size_sram_a, off_sram_a,
                score_q, size_q, off_q,
                (bcnt < BBOX_N_SCORE),
                (bcnt >= BBOX_N_SCORE) && (bcnt < (BBOX_N_SCORE + BBOX_N_SIZE)),
                (bcnt >= (BBOX_N_SCORE + BBOX_N_SIZE)) && (bcnt <= BBOX_STREAM_LAST),
                u_bbox.state);
            `ifdef DUMP_HEAD_BBOX_STROBE
            // NBA 後：本拍寫入 bbox_cal_*、valid 與 score_q 對齊
            $strobe(
                "[HEAD_BBOX_strobe] t=%0t bcnt=%0d sc_q=%h | cal_sc=%h cal_sz=%h cal_of=%h | v_sc=%b v_sz=%b v_of=%b",
                $time, bcnt, score_q,
                bbox_cal_sc, bbox_cal_sz, bbox_cal_of,
                bbox_sc_valid, bbox_sz_valid, bbox_of_valid);
            `endif
        end
    end
end
`endif

`ifdef DUMP_HEAD_BBOX_SRAM_NEGEDGE
// ---------------------------------------------------------------------------
// negedge：SRAM macro CLK=~clk，此時點位址與 Q 較一致（對照 posedge 的 [HEAD_BBOX]）
// ---------------------------------------------------------------------------
always @(negedge clk) begin
    if (!reset && (state == S_BBOX) && bbox_busy && bbox_dv) begin
        if ((bcnt[5:0] == 6'd0) || (bcnt == BBOX_N_SCORE) ||
            (bcnt == (BBOX_N_SCORE + BBOX_N_SIZE)) || (bcnt == BBOX_STREAM_LAST)) begin
            $display(
                "[HEAD_BBOX_n] t=%0t bcnt=%0d sc_a=%h sz_a=%h of_a=%h | sc_q=%h sz_q=%h of_q=%h",
                $time, bcnt, score_sram_a[7:0], size_sram_a[8:0], off_sram_a[8:0],
                score_q, size_q, off_q);
        end
    end
end
`endif

`ifdef DUMP_HEAD_CTR_WRITE
// ---------------------------------------------------------------------------
// S_CTR 寫入 score SRAM 當拍：確認 mac_sig 是否每格皆 sigmoid 下限等
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!reset && (state == S_CTR) && mac_bp && mac_dv) begin
        if (({cur_oh, cur_ow} & 6'h3f) == 6'd0)
            $display(
                "[HEAD_CTR_W] t=%0t oh=%0d ow=%0d sc_a=%h mac_sig=%h mac_cl=%h",
                $time, cur_oh, cur_ow, score_sram_a[7:0], mac_sig, mac_cl);
    end
end
`endif

`ifdef DUMP_HEAD_SCORE_DEEP
// ---------------------------------------------------------------------------
// 載入 conv2 / tail_ctr 線性輸出 golden（僅 simulation）
// ---------------------------------------------------------------------------
initial begin
    $readmemb("./TXT_File/Activation/box_head_head_input_bi.txt",
              opt_golden_mem);
    $readmemb("./TXT_File/Activation/box_head_shared_after_conv1_out_bi.txt",
              conv1_golden_mem);
    $readmemb("./TXT_File/Activation/box_head_shared_after_conv2_out_bi.txt",
              conv2_golden_mem);
    $readmemb("./TXT_File/Activation/box_head_tail_ctr_after_conv_out_bi.txt",
              raw_ctr_golden_mem);
    $display("[HEAD_GOLDEN_CMP] Loaded opt(%0d) conv1(%0d) conv2(%0d) raw_ctr(%0d)",
             OPT_GOLDEN_LEN, CONV1_GOLDEN_LEN, CONV2_GOLDEN_LEN, FEAT_SZ);
end
`endif

assign busy = (state != S_IDLE);

// ---------------------------------------------------------------------------
// negedge：bias ROM Q 在 mac_dv=0 位址相之後鎖存（mac_dv=1 時 rom_q 可能仍為上一拍權重）
// ---------------------------------------------------------------------------
always @(negedge clk) begin
    if (!reset && mac_bp && !mac_dv) begin
        case (state)
            S_CONV1, S_CONV2: bias_q_lat <= c12b_q;
            S_CTR, S_SIZE, S_OFF: bias_q_lat <= csob_q;
            default: bias_q_lat <= 16'sd0;
        endcase
    end
end

endmodule
