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
//   Sram_score  256×16-bit  A=8   score_buf
//   Sram_size   512×16-bit  A=9   size_buf
//   Sram_off    512×16-bit  A=9   off_buf
//
// SRAM/ROM 皆採 CLK=~clk（falling-edge）；mac_dv 為 1-cycle 讀取 pipeline。
// =============================================================================

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
localparam BBOX_BCNT_MAX = 2 * FEAT_SZ - 1;      // S_BBOX stream length - 1

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

wire ceb_opt   = !(state==S_FILL || state==S_CONV1);
wire ceb_sh1   = !(state==S_CONV1 || state==S_CONV2);
wire ceb_sh2   = !(state==S_CONV2 || state==S_CTR || state==S_SIZE || state==S_OFF);
wire ceb_score = !(state==S_CTR  || state==S_BBOX);
wire ceb_size  = !(state==S_SIZE || state==S_BBOX);
wire ceb_off   = !(state==S_OFF  || state==S_BBOX);

// Write enable: active (0) only on the bias-data cycle when writing result
wire web_opt   = !(state==S_FILL && a_valid);
wire web_sh1   = !(state==S_CONV1 && mac_bp && mac_dv);
wire web_sh2   = !(state==S_CONV2 && mac_bp && mac_dv);
wire web_score = !(state==S_CTR  && mac_bp && mac_dv);
wire web_size  = !(state==S_SIZE && mac_bp && mac_dv);
wire web_off   = !(state==S_OFF  && mac_bp && mac_dv);

// ---------------------------------------------------------------------------
// Fill / streaming counters
// ---------------------------------------------------------------------------
reg [13:0] fill_cnt;
reg [8:0]  bcnt;     // bbox streaming counter
reg        bbox_dv;  // 1-cycle pipeline for bbox SRAM read

wire [13:0] fill_off    = fill_cnt - SKIP_VALS;
wire [7:0]  fill_n      = fill_off[12:5];
wire [4:0]  fill_c      = fill_off[4:0];
wire        fill_search = (fill_cnt >= SKIP_VALS) &&
                          (fill_cnt <  TOT_VALS);
wire [12:0] opt_wr      = {fill_c, fill_n};

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
wire [12:0] opt_sram_a  = (state==S_FILL) ? opt_wr : opt_rd;
wire [14:0] sh1_sram_a  = (state==S_CONV1 && mac_bp) ?
                           {cur_oc[6:0], cur_oh, cur_ow} : sh1_rd;
wire [13:0] sh2_sram_a  = (state==S_CONV2 && mac_bp) ?
                           {cur_oc[5:0], cur_oh, cur_ow} : sh2_rd;
// Pad to Sram_* macro address width (13b) to avoid PCWM lint
wire [12:0] score_sram_a = {5'b0,
    (state==S_CTR) ? {cur_oh, cur_ow} : bcnt[7:0]};
wire [12:0] size_sram_a  = {4'b0,
    (state==S_SIZE && mac_bp) ? {cur_oc[0], cur_oh, cur_ow} : bcnt[8:0]};
wire [12:0] off_sram_a   = {4'b0,
    (state==S_OFF && mac_bp) ? {cur_oc[0], cur_oh, cur_ow} : bcnt[8:0]};

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

// Feature mux (valid when mac_dv=1; uses pad_r for correct timing)
wire signed [15:0] feat_raw =
    (state == S_CONV1) ? opt_q :
    (state == S_CONV2) ? sh1_q : sh2_q;
wire signed [15:0] feat_q = (pad_r && state != S_CTR &&
                              state != S_SIZE && state != S_OFF) ?
                              16'sd0 : feat_raw;

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
wire ceb_csow  = !(state==S_CTR || state==S_SIZE || state==S_OFF) ||
                  mac_bp || mac_dv;
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
wire signed [31:0] mac_prod = feat_q * rom_q;

// ---------------------------------------------------------------------------
// Bias-phase combinational outputs
// ---------------------------------------------------------------------------
wire signed [47:0] mac_sh  = $signed(mac_acc) >>> 8;
wire signed [47:0] mac_bi  = mac_sh + {{32{rom_q[15]}}, rom_q};
wire signed [15:0] mac_cl  = (mac_bi >  48'sh7FFF) ? 16'sh7FFF :
                              (mac_bi < -48'sh8000) ? -16'sh8000 :
                               mac_bi[15:0];
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
// cal_bbox
// ---------------------------------------------------------------------------
reg  bbox_start;
wire bbox_busy, bbox_done;
reg  bv_s, bv_sz, bv_of;

cal_bbox #(.FEAT_SZ(FEAT_H), .FEAT_LEN(FEAT_SZ)) u_bbox (
    .clk(clk), .reset(reset), .start(bbox_start),
    .score_i(score_q),  .score_valid (bv_s  && bbox_dv),
    .size_i (size_q),   .size_valid  (bv_sz && bbox_dv),
    .offset_i(off_q),   .offset_valid(bv_of && bbox_dv),
    .busy(bbox_busy), .done(bbox_done),
    .cx_o(cx_o), .cy_o(cy_o), .w_o(w_o), .h_o(h_o)
);

// ---------------------------------------------------------------------------
// SRAM instances (CLK=~clk, 16-bit data width, BWEB=16'b0)
// ---------------------------------------------------------------------------
Sram_opt u_Sram_opt (
    .SLP(pgen_opt), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_opt), .WEB(web_opt),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(opt_sram_a), .D(a_i), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(opt_q));

// Sram_sh1_lo: 16384×16-bit  OC 0..63  CM=16  A[13:0]
Sram_sh1_lo u_Sram_sh1_lo (
    .SLP(pgen_sh1), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk),
    .CEB(ceb_sh1 | sh1_hi_sel),
    .WEB(web_sh1 | sh1_hi_sel),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(sh1_sram_a[13:0]), .D(mac_relu), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(sh1_q_lo));

// Sram_sh1_hi: 8192×16-bit  OC 64..95  CM=8  A[12:0]
Sram_sh1_hi u_Sram_sh1_hi (
    .SLP(pgen_sh1), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk),
    .CEB(ceb_sh1 | ~sh1_hi_sel),
    .WEB(web_sh1 | ~sh1_hi_sel),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(sh1_sram_a[12:0]), .D(mac_relu), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(sh1_q_hi));

Sram_sh2 u_Sram_sh2 (
    .SLP(pgen_sh2), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_sh2), .WEB(web_sh2),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(sh2_sram_a), .D(mac_relu), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(sh2_q));

Sram_score u_Sram_score (
    .SLP(pgen_score), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_score), .WEB(web_score),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(score_sram_a), .D(mac_sig), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(score_q));

Sram_size u_Sram_size (
    .SLP(pgen_size), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_size), .WEB(web_size),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(size_sram_a), .D(mac_sig), .BWEB(16'b0),
    .AM(), .DM(), .BWEBM(16'b0),
    .RTSEL(2'b01), .WTSEL(2'b00), .Q(size_q));

Sram_off u_Sram_off (
    .SLP(pgen_off), .DSLP(1'b0), .SD(1'b0), .PUDELAY(),
    .CLK(~clk), .CEB(ceb_off), .WEB(web_off),
    .BIST(1'b0), .CEBM(), .WEBM(),
    .A(off_sram_a), .D(mac_cl), .BWEB(16'b0),
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
        pgen_opt   <= !(state==S_FILL   || state==S_CONV1);
        pgen_sh1   <= !(state==S_CONV1  || state==S_CONV2);
        pgen_sh2   <= !(state==S_CONV2  || state==S_CTR ||
                        state==S_SIZE   || state==S_OFF);
        pgen_score <= !(state==S_CTR    || state==S_BBOX);
        pgen_size  <= !(state==S_SIZE   || state==S_BBOX);
        pgen_off   <= !(state==S_OFF    || state==S_BBOX);
    end
end

// ---------------------------------------------------------------------------
// FSM segment 4: sequential datapath
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    done       <= 1'b0;
    bbox_start <= 1'b0;
    bv_s  <= 1'b0; bv_sz <= 1'b0; bv_of <= 1'b0;
    pad_r <= pad;

    if (reset) begin
        fill_cnt <= 14'd0;
        cur_oc   <= 7'd0;  cur_oh  <= 4'd0;  cur_ow  <= 4'd0;
        mac_ic   <= 7'd0;  mac_kh  <= 2'd0;  mac_kw  <= 2'd0;
        mac_acc  <= 48'sd0;
        mac_bp   <= 1'b0;  mac_dv  <= 1'b0;
        bcnt     <= 9'd0;  bbox_dv <= 1'b0;
    end else begin
        case (state)

        // ----------------------------------------------------------------
        S_IDLE: begin
            fill_cnt <= 14'd0;
            cur_oc   <= 7'd0;  cur_oh  <= 4'd0;  cur_ow  <= 4'd0;
            mac_ic   <= 7'd0;  mac_kh  <= 2'd0;  mac_kw  <= 2'd0;
            mac_acc  <= 48'sd0;
            mac_bp   <= 1'b0;  mac_dv  <= 1'b0;
            bcnt     <= 9'd0;  bbox_dv <= 1'b0;
        end

        // ----------------------------------------------------------------
        // Write search tokens into opt SRAM (template tokens discarded)
        S_FILL: begin
            if (a_valid) begin
                // Opt SRAM write: controlled by web_opt/ceb_opt wires;
                // here we just advance fill_cnt
                if (fill_cnt < TOT_VALS)
                    fill_cnt <= fill_cnt + 14'd1;
            end
        end

        // ----------------------------------------------------------------
        // shared_conv1: 3×3, IN_CH→C_SH1, ReLU; mac_dv pipelines SRAM latency
        S_CONV1: begin
            if (!mac_bp) begin
                if (mac_dv) begin
                    mac_acc <= mac_acc + {{16{mac_prod[31]}}, mac_prod};
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
                    end
                end else mac_dv <= 1'b1;
            end else begin
                // Bias phase: mac_dv=0→wait ROM, mac_dv=1→apply+write+advance
                if (mac_dv) begin
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
                    end
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
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
                    end else mac_ic <= mac_ic + 7'd1;
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
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
                    end else mac_ic <= mac_ic + 7'd1;
                end else mac_dv <= 1'b1;
            end else begin
                if (mac_dv) begin
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
                    end else mac_ic <= mac_ic + 7'd1;
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
            if (!bbox_busy) begin
                bbox_start <= 1'b1;
                bcnt       <= 9'd0;
                bbox_dv    <= 1'b0;
            end else begin
                bbox_dv <= 1'b1;
                if (bbox_dv) begin
                    bv_s  <= 1'b1;
                    bv_sz <= 1'b1;
                    bv_of <= 1'b1;
                    if (bcnt < BBOX_BCNT_MAX) bcnt <= bcnt + 9'd1;
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
    end
end

assign busy = (state != S_IDLE);

endmodule
