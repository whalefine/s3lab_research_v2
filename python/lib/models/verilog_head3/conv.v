// =============================================================================
// conv.v  --  Generic Conv2D (Q8.8 fixed point, FSM multi-cycle, ROM prefetch)
// -----------------------------------------------------------------------------
// 對應 numpy: run_backbone_numpy_shared_trunk.py -> conv2d() + bias + (ReLU) + fp()
//
// ROM read contract (verilog_rule.mdc §7.1 ; ROM macro CLK(~clk)) :
//   posedge T   : 送 w_addr / b_addr
//   posedge T+1 : w_i / b_i 有效；本拍寫 parent Sram_tok1 (wgt) / bias_r (negedge)
// wgt_buf: parent Sram_tok1 in head_top. WPRE phase1 write; MAC phase0 read, phase1 comb wgt_rd_i.
//
// WPRE: 2-phase ROM prefetch; MAC: 1-phase pipelined (read feat k+1 while MAC feat k).
// =============================================================================

module conv (
    clk           ,
    rst_n         ,
    start         ,
    busy          ,
    done          ,
    x_addr        ,
    x_i           ,
    y_valid       ,
    y_data        ,
    y_oc          ,
    y_oh          ,
    y_ow          ,
    mac_phase_o   ,
    x_addr_mac_rd ,
    wgt_wr_en     ,
    wgt_wr_addr   ,
    wgt_wr_data   ,
    wgt_rd_req    ,
    wgt_rd_addr   ,
    wgt_rd_i      ,
    stall         ,
    mac_active_o  ,
    cur_oh_o
);

parameter IN_CH    = 32 ;
parameter OUT_CH   = 96 ;
parameter IN_H     = 16 ;
parameter IN_W     = 16 ;
parameter K        = 3  ;
parameter PAD      = 1  ;
parameter HAS_RELU = 1  ;
parameter DATA_W   = 16 ;
parameter FRAC_W   = 8  ;
parameter ACC_W    = 32 ;

parameter OUT_H       = IN_H + 2*PAD - K + 1 ;
parameter OUT_W       = IN_W + 2*PAD - K + 1 ;
parameter KK          = K*K ;
parameter FEAT_PER_OC = IN_CH * KK ;

parameter X_AW    = 14 ;
parameter W_AW    = 16 ;
parameter B_AW    = 8  ;
parameter FEAT_AW = 10 ;
parameter OC_AW   = 8  ;
parameter HW_AW   = 5  ;
parameter ROM_PROFILE = 1 ;
parameter OC_PAR      = 1 ;
// 1: (acc+2^(FRAC_W-1))>>>FRAC_W+bias matches numpy fp() after conv; 0: trunc
parameter ROUND_Y     = 0 ;

input                       clk     ;
input                       rst_n   ;
input                       start   ;
output                      busy    ;
output                      done    ;
output [X_AW-1:0]           x_addr  ;
input  [DATA_W-1:0]         x_i     ;
output                      y_valid ;
output [DATA_W-1:0]         y_data  ;
output [OC_AW-1:0]          y_oc    ;
output [HW_AW-1:0]          y_oh    ;
output [HW_AW-1:0]          y_ow    ;
output                      mac_phase_o   ;
output [X_AW-1:0]           x_addr_mac_rd ;
output                      wgt_wr_en     ;
output [FEAT_AW-1:0]        wgt_wr_addr   ;
output [DATA_W-1:0]         wgt_wr_data   ;
output                      wgt_rd_req    ;
output [FEAT_AW-1:0]        wgt_rd_addr   ;
input  [DATA_W-1:0]         wgt_rd_i      ;
input                       stall         ;
output                      mac_active_o  ;
output [HW_AW-1:0]          cur_oh_o      ;

parameter S_IDLE = 3'd0 ;
parameter S_WPRE = 3'd1 ;
parameter S_MAC  = 3'd2 ;
parameter S_SAT  = 3'd3 ;
parameter S_DONE = 3'd4 ;

reg  [2:0]               CS, NS ;
reg  [OC_AW-1:0]         oc_base_r ;
reg  [HW_AW-1:0]         oh_r, ow_r ;

reg                      wpre_phase ;
reg  [FEAT_AW-1:0]       wpre_feat ;
reg  [3:0]               wpre_lane ;
reg                      wpre_done ;
reg                      bpre_phase ;
reg  [3:0]               bpre_lane ;
reg                      bpre_done ;

reg                      mac_fill ;
reg  [FEAT_AW-1:0]       mac_feat ;
reg                      mac_done ;
reg                      mac_xi_pad_r ;

reg  signed [DATA_W-1:0] bias_r [0:OC_PAR-1] ;
reg  signed [DATA_W-1:0] wgt_buf [0:OC_PAR-1][0:FEAT_PER_OC-1] ;
reg  signed [ACC_W-1:0]  acc_r [0:OC_PAR-1] ;
reg  signed [ACC_W-1:0]  acc_sat_r [0:OC_PAR-1] ;
reg  [X_AW-1:0]          x_addr_r ;
reg  [W_AW-1:0]          w_addr_r ;
reg  [B_AW-1:0]          b_addr_r ;

reg                      y_valid_r ;
reg  signed [DATA_W-1:0] y_data_r ;
reg  [OC_AW-1:0]         y_oc_r ;
reg  [HW_AW-1:0]         y_oh_r, y_ow_r ;

reg                      busy_r, done_r ;

wire                     ow_last ;
wire                     oh_last ;
wire                     oc_last ;
reg  [3:0]               sat_lane ;
wire                     sat_lane_valid ;

wire [6:0]               mac_ic ;
wire [3:0]               mac_kh ;
wire [3:0]               mac_kw ;
wire signed [HW_AW:0]    ih_s ;
wire signed [HW_AW:0]    iw_s ;
wire [X_AW-1:0]          x_addr_nxt ;
wire                     pad_nxt ;
wire                     mac_feat_last ;

wire signed [DATA_W-1:0]       mac_x_op ;
integer i_lane ;
reg signed [DATA_W-1:0]  mac_w_op [0:OC_PAR-1] ;
reg signed [2*DATA_W-1:0] mac_prod [0:OC_PAR-1] ;
reg signed [ACC_W-1:0]    acc_next [0:OC_PAR-1] ;
reg signed [ACC_W-1:0]    acc_shifted ;
reg signed [ACC_W-1:0]    y_pre_sat ;

reg signed [DATA_W-1:0]   y_relu ;

wire mac_stall = stall && (CS == S_MAC) && !mac_done ;
wire cs_en     = !mac_stall ;

wire                 rom_c1_use_w2 ;
wire                 rom_c2_use_w2 ;
wire                 rom_c2_use_w3 ;
wire [13:0]          rom_w_a ;
wire [7:0]           rom_c12b_a ;
wire                 rom_ceb_w1 ;
wire                 rom_ceb_w2 ;
wire                 rom_ceb_w3 ;
wire                 rom_ceb_b ;
wire signed [15:0]   rom_w1_q ;
wire signed [15:0]   rom_w2_q ;
wire signed [15:0]   rom_w3_q ;
wire signed [15:0]   rom_c12b_q ;
wire signed [15:0]   w_i ;

assign rom_c1_use_w2 = w_addr_r[14] ;
assign rom_c2_use_w3 = w_addr_r[15] ;
assign rom_c2_use_w2 = w_addr_r[14] & ~w_addr_r[15] ;
assign rom_w_a       = w_addr_r[13:0] ;
assign rom_ceb_b     = !busy_r ;

generate
    if (ROM_PROFILE == 1) begin : gen_rom_c1
        assign rom_c12b_a = {1'b0, b_addr_r[6:0]} ;
        assign rom_ceb_w1 = !(busy_r && !rom_c1_use_w2) ;
        assign rom_ceb_w2 = !(busy_r &&  rom_c1_use_w2) ;
        assign rom_ceb_w3 = 1'b1 ;
        assign w_i        = rom_c1_use_w2 ? rom_w2_q : rom_w1_q ;
        rom_box_head_shared_conv1_folded_weight1 u_rom_w1 (
            .A(rom_w_a), .AM(), .CEBM(), .BIST(1'b0),
            .CEB(rom_ceb_w1), .CLK(~clk),
            .SD(1'b0), .PUDELAY(),
            .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
            .Q(rom_w1_q));
        rom_box_head_shared_conv1_folded_weight2 u_rom_w2 (
            .A(rom_w_a), .AM(), .CEBM(), .BIST(1'b0),
            .CEB(rom_ceb_w2), .CLK(~clk),
            .SD(1'b0), .PUDELAY(),
            .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
            .Q(rom_w2_q));
        assign rom_w3_q = 16'sd0 ;
    end else begin : gen_rom_c2
        assign rom_c12b_a = 8'd96 + {2'b0, b_addr_r[5:0]} ;
        assign rom_ceb_w1 = !(busy_r && !rom_c2_use_w2 && !rom_c2_use_w3) ;
        assign rom_ceb_w2 = !(busy_r &&  rom_c2_use_w2) ;
        assign rom_ceb_w3 = !(busy_r &&  rom_c2_use_w3) ;
        assign w_i        = rom_c2_use_w3 ? rom_w3_q :
                            rom_c2_use_w2 ? rom_w2_q : rom_w1_q ;
        rom_box_head_shared_conv2_folded_weight1 u_rom_w1 (
            .A(rom_w_a), .AM(), .CEBM(), .BIST(1'b0),
            .CEB(rom_ceb_w1), .CLK(~clk),
            .SD(1'b0), .PUDELAY(),
            .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
            .Q(rom_w1_q));
        rom_box_head_shared_conv2_folded_weight2 u_rom_w2 (
            .A(rom_w_a), .AM(), .CEBM(), .BIST(1'b0),
            .CEB(rom_ceb_w2), .CLK(~clk),
            .SD(1'b0), .PUDELAY(),
            .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
            .Q(rom_w2_q));
        rom_box_head_shared_conv2_folded_weight3 u_rom_w3 (
            .A(rom_w_a), .AM(), .CEBM(), .BIST(1'b0),
            .CEB(rom_ceb_w3), .CLK(~clk),
            .SD(1'b0), .PUDELAY(),
            .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
            .Q(rom_w3_q));
    end
endgenerate

rom_box_head_shared_conv1_2_folded_bias u_rom_c12b (
    .A(rom_c12b_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(rom_ceb_b), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(rom_c12b_q));

assign busy          = busy_r ;
assign done          = done_r ;
assign x_addr        = x_addr_r ;
// mac_phase_o=1: parent should feed x_i (SRAM Q from prior read); 0: fill cycle only
assign mac_phase_o   = (CS == S_MAC) && !mac_done && !mac_fill ;
// Issue read for feat0 on fill; else read feat mac_feat+1 (Q valid next cycle for MAC)
wire [FEAT_AW-1:0]       mac_feat_rd   = mac_feat + 1 ;
wire [6:0]               rd_ic         = mac_feat_rd / KK ;
wire [3:0]               rd_kh         = (mac_feat_rd % KK) / K ;
wire [3:0]               rd_kw         = (mac_feat_rd % KK) % K ;
wire signed [HW_AW:0]    rd_ih_s       = $signed({1'b0, oh_r}) + $signed({2'b00, rd_kh}) - PAD ;
wire signed [HW_AW:0]    rd_iw_s       = $signed({1'b0, ow_r}) + $signed({2'b00, rd_kw}) - PAD ;
wire                     rd_pad_nxt    = (rd_ih_s < 0) || (rd_ih_s >= IN_H) || (rd_iw_s < 0) || (rd_iw_s >= IN_W) ;
wire [X_AW-1:0]          x_addr_rd_nxt = rd_pad_nxt ? {X_AW{1'b0}} :
                         (rd_ic * (IN_H*IN_W) + rd_ih_s[HW_AW-1:0] * IN_W + rd_iw_s[HW_AW-1:0]) ;
wire [X_AW-1:0] mac_rd_addr = mac_fill ? x_addr_nxt :
                              mac_feat_last ? x_addr_r : x_addr_rd_nxt ;
assign x_addr_mac_rd = (CS == S_MAC && !mac_done) ? mac_rd_addr : x_addr_r ;
assign wgt_wr_en   = 1'b0 ;
assign wgt_wr_addr = {FEAT_AW{1'b0}} ;
assign wgt_wr_data = {DATA_W{1'b0}} ;
assign wgt_rd_req  = 1'b0 ;
assign wgt_rd_addr = {FEAT_AW{1'b0}} ;
assign y_valid = y_valid_r ;
assign y_data  = y_data_r ;
assign y_oc    = y_oc_r ;
assign y_oh    = y_oh_r ;
assign y_ow    = y_ow_r ;

assign ow_last       = (ow_r == OUT_W - 1) ;
assign oh_last       = (oh_r == OUT_H - 1) ;
assign oc_last       = (oc_base_r + OC_PAR >= OUT_CH) ;
assign sat_lane_valid = (oc_base_r + sat_lane < OUT_CH) ;
assign mac_active_o   = (CS == S_MAC) ;
assign cur_oh_o       = oh_r ;
assign mac_ic        = mac_feat / KK ;
assign mac_kh        = (mac_feat % KK) / K ;
assign mac_kw        = (mac_feat % KK) % K ;
assign ih_s          = $signed({1'b0, oh_r}) + $signed({2'b00, mac_kh}) - PAD ;
assign iw_s          = $signed({1'b0, ow_r}) + $signed({2'b00, mac_kw}) - PAD ;
assign pad_nxt       = (ih_s < 0) || (ih_s >= IN_H) || (iw_s < 0) || (iw_s >= IN_W) ;
assign x_addr_nxt    = pad_nxt ? {X_AW{1'b0}} :
                       (mac_ic * (IN_H*IN_W) + ih_s[HW_AW-1:0] * IN_W + iw_s[HW_AW-1:0]) ;
assign mac_feat_last = (mac_feat == FEAT_PER_OC - 1) ;

// pad locked when read issued (same as 2-phase phase0 x_in_pad -> phase1 MAC)
assign mac_x_op  = mac_xi_pad_r ? {DATA_W{1'b0}} : $signed(x_i) ;
// FSM CS
always @(posedge clk) begin
    if (!rst_n)
        CS <= S_IDLE ;
    else if (cs_en)
        CS <= NS ;
end

// FSM NS
always @(*) begin
    NS = CS ;
    case (CS)
        S_IDLE : if (start)     NS = S_WPRE ;
        S_WPRE : if (wpre_done && bpre_done) NS = S_MAC ;
        S_MAC  : if (mac_done)  NS = S_SAT ;
        S_SAT  : begin
            if (sat_lane_valid && (sat_lane < OC_PAR-1)) NS = S_SAT ;
            else if (ow_last && oh_last && oc_last) NS = S_DONE ;
            else if (ow_last && oh_last)       NS = S_WPRE ;
            else                               NS = S_MAC ;
        end
        S_DONE :                NS = S_IDLE ;
        default :               NS = S_IDLE ;
    endcase
end

// busy_r, done_r
always @(posedge clk) begin
    if (!rst_n) begin
        busy_r <= 1'b0 ;
        done_r <= 1'b0 ;
    end else begin
        busy_r <= (NS != S_IDLE) && (NS != S_DONE) ;
        done_r <= (NS == S_DONE) ;
    end
end

// oc_base_r, oh_r, ow_r
wire sat_spatial_step = cs_en && (CS == S_SAT) &&
                        !(sat_lane_valid && (sat_lane < OC_PAR - 1)) ;

always @(posedge clk) begin
    if (!rst_n) begin
        oc_base_r <= 0 ;
        oh_r      <= 0 ;
        ow_r      <= 0 ;
    end else if (CS == S_IDLE) begin
        oc_base_r <= 0 ;
        oh_r      <= 0 ;
        ow_r      <= 0 ;
    end else if (sat_spatial_step && ow_last && oh_last && !oc_last) begin
        oc_base_r <= oc_base_r + OC_PAR ;
        oh_r      <= 0 ;
        ow_r      <= 0 ;
    end else if (sat_spatial_step && ow_last && !oh_last) begin
        oh_r <= oh_r + 1 ;
        ow_r <= 0 ;
    end else if (sat_spatial_step && !ow_last)
        ow_r <= ow_r + 1 ;
end

// weight/bias prefetch: wpre_*, w_addr_r, b_addr_r (flat if / else if)
wire wpre_clr      = cs_en && (CS == S_IDLE) ;
wire wpre_w_rom_a0 = cs_en && (CS == S_WPRE) && !wpre_done && (wpre_phase == 1'b0) ;
wire wpre_w_rom_a1_last = cs_en && (CS == S_WPRE) && !wpre_done && (wpre_phase == 1'b1) &&
                            (wpre_lane == OC_PAR - 1) && (wpre_feat == FEAT_PER_OC - 1) ;
wire wpre_w_rom_a1_more = cs_en && (CS == S_WPRE) && !wpre_done && (wpre_phase == 1'b1) &&
                            (wpre_lane == OC_PAR - 1) && (wpre_feat != FEAT_PER_OC - 1) ;
wire wpre_w_rom_a1_lane = cs_en && (CS == S_WPRE) && !wpre_done && (wpre_phase == 1'b1) &&
                            (wpre_lane != OC_PAR - 1) ;
wire bpre_rom_a0   = cs_en && (CS == S_WPRE) && wpre_done && !bpre_done && (bpre_phase == 1'b0) ;
wire bpre_rom_a1_last = cs_en && (CS == S_WPRE) && wpre_done && !bpre_done && (bpre_phase == 1'b1) &&
                          (bpre_lane == OC_PAR - 1) ;
wire bpre_rom_a1_more = cs_en && (CS == S_WPRE) && wpre_done && !bpre_done && (bpre_phase == 1'b1) &&
                          (bpre_lane != OC_PAR - 1) ;
wire wpre_sat_wrap   = cs_en && (CS == S_SAT) && sat_spatial_step &&
                       ow_last && oh_last && !oc_last ;

always @(posedge clk) begin
    if (!rst_n) begin
        wpre_phase <= 1'b0 ;
        wpre_feat  <= 0 ;
        wpre_lane  <= 0 ;
        wpre_done  <= 1'b0 ;
        bpre_phase <= 1'b0 ;
        bpre_lane  <= 0 ;
        bpre_done  <= 1'b0 ;
        w_addr_r   <= 0 ;
        b_addr_r   <= 0 ;
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1)
            bias_r[i_lane] <= 0 ;
    end else if (wpre_clr) begin
        wpre_phase <= 1'b0 ;
        wpre_feat  <= 0 ;
        wpre_lane  <= 0 ;
        wpre_done  <= 1'b0 ;
        bpre_phase <= 1'b0 ;
        bpre_lane  <= 0 ;
        bpre_done  <= 1'b0 ;
    end else if (wpre_w_rom_a0) begin
        w_addr_r   <= (oc_base_r + wpre_lane) * FEAT_PER_OC + wpre_feat ;
        wpre_phase <= 1'b1 ;
    end else if (wpre_w_rom_a1_last) begin
        wgt_buf[wpre_lane][wpre_feat] <= w_i ;
        wpre_lane  <= 0 ;
        wpre_done  <= 1'b1 ;
        wpre_phase <= 1'b0 ;
    end else if (wpre_w_rom_a1_more) begin
        wgt_buf[wpre_lane][wpre_feat] <= w_i ;
        wpre_lane  <= 0 ;
        wpre_feat  <= wpre_feat + 1 ;
        wpre_phase <= 1'b0 ;
    end else if (wpre_w_rom_a1_lane) begin
        wgt_buf[wpre_lane][wpre_feat] <= w_i ;
        wpre_lane  <= wpre_lane + 1 ;
        wpre_phase <= 1'b0 ;
    end else if (bpre_rom_a0) begin
        b_addr_r   <= oc_base_r + bpre_lane ;
        bpre_phase <= 1'b1 ;
    end else if (bpre_rom_a1_last) begin
        bias_r[bpre_lane] <= rom_c12b_q ;
        bpre_done  <= 1'b1 ;
        bpre_phase <= 1'b0 ;
    end else if (bpre_rom_a1_more) begin
        bias_r[bpre_lane] <= rom_c12b_q ;
        bpre_lane  <= bpre_lane + 1 ;
        bpre_phase <= 1'b0 ;
    end else if (wpre_sat_wrap) begin
        wpre_phase <= 1'b0 ;
        wpre_feat  <= 0 ;
        wpre_lane  <= 0 ;
        wpre_done  <= 1'b0 ;
        bpre_phase <= 1'b0 ;
        bpre_lane  <= 0 ;
        bpre_done  <= 1'b0 ;
    end
end

// MAC: mac_*, x_addr_r, acc_r, acc_sat_r (flat if / else if)
wire mac_wpre_arm   = cs_en && (CS == S_WPRE) && wpre_done && bpre_done ;
wire mac_fill_rd    = cs_en && (CS == S_MAC) && !mac_done && mac_fill ;
wire mac_accum_last = cs_en && (CS == S_MAC) && !mac_done && !mac_fill && mac_feat_last ;
wire mac_accum_more = cs_en && (CS == S_MAC) && !mac_done && !mac_fill && !mac_feat_last ;
wire mac_sat_lane   = cs_en && (CS == S_SAT) && sat_lane_valid && (sat_lane < OC_PAR - 1) ;
wire mac_sat_wrap   = cs_en && (CS == S_SAT) &&
                        !(sat_lane_valid && (sat_lane < OC_PAR - 1)) ;

always @(posedge clk) begin
    if (!rst_n) begin
        mac_fill     <= 1'b1 ;
        mac_feat     <= 0 ;
        mac_done     <= 1'b0 ;
        x_addr_r     <= 0 ;
        mac_xi_pad_r <= 1'b0 ;
        sat_lane     <= 0 ;
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1) begin
            acc_r[i_lane]     <= 0 ;
            acc_sat_r[i_lane] <= 0 ;
        end
    end else if (mac_wpre_arm) begin
        mac_fill  <= 1'b1 ;
        mac_feat  <= 0 ;
        mac_done  <= 1'b0 ;
        sat_lane  <= 0 ;
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1)
            acc_r[i_lane] <= 0 ;
    end else if (mac_fill_rd) begin
        x_addr_r     <= x_addr_nxt ;
        mac_xi_pad_r <= pad_nxt ;
        mac_fill     <= 1'b0 ;
    end else if (mac_accum_last) begin
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1)
            acc_r[i_lane] <= (oc_base_r + i_lane < OUT_CH) ? acc_next[i_lane] : acc_r[i_lane] ;
        mac_done <= 1'b1 ;
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1)
            acc_sat_r[i_lane] <= (oc_base_r + i_lane < OUT_CH) ? acc_next[i_lane] : acc_sat_r[i_lane] ;
    end else if (mac_accum_more) begin
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1)
            acc_r[i_lane] <= (oc_base_r + i_lane < OUT_CH) ? acc_next[i_lane] : acc_r[i_lane] ;
        mac_feat     <= mac_feat + 1 ;
        mac_xi_pad_r <= rd_pad_nxt ;
    end else if (mac_sat_lane)
        sat_lane <= sat_lane + 1 ;
    else if (mac_sat_wrap) begin
        mac_fill  <= 1'b1 ;
        mac_feat  <= 0 ;
        mac_done  <= 1'b0 ;
        sat_lane  <= 0 ;
        for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1)
            acc_r[i_lane] <= 0 ;
    end
end

// mac comb + sat16
reg signed [DATA_W-1:0] y_sat_w ;

always @(*) begin
    for (i_lane = 0; i_lane < OC_PAR; i_lane = i_lane + 1) begin
        mac_w_op[i_lane] = wgt_buf[i_lane][mac_feat] ;
        mac_prod[i_lane] = mac_x_op * mac_w_op[i_lane] ;
        acc_next[i_lane] = acc_r[i_lane] + mac_prod[i_lane] ;
    end
    acc_shifted = ROUND_Y ? ((acc_sat_r[sat_lane] + (32'sd1 << (FRAC_W-1))) >>> FRAC_W)
                          : (acc_sat_r[sat_lane] >>> FRAC_W) ;
    y_pre_sat   = acc_shifted + bias_r[sat_lane] ;
    if (y_pre_sat >  32'sd32767)
        y_sat_w = 16'sh7fff ;
    else if (y_pre_sat < -32'sd32768)
        y_sat_w = 16'sh8000 ;
    else
        y_sat_w = y_pre_sat[DATA_W-1:0] ;
end

always @(*) begin
    if ((HAS_RELU != 0) && y_sat_w[DATA_W-1])
        y_relu = {DATA_W{1'b0}} ;
    else
        y_relu = y_sat_w ;
end

// y_valid_r, y_data_r, y_oc_r, y_oh_r, y_ow_r
always @(posedge clk) begin
    if (!rst_n) begin
        y_valid_r <= 1'b0 ;
        y_data_r  <= 0 ;
        y_oc_r    <= 0 ;
        y_oh_r    <= 0 ;
        y_ow_r    <= 0 ;
    end else if (CS == S_SAT && sat_lane_valid) begin
        y_valid_r <= 1'b1 ;
        y_data_r  <= y_relu ;
        y_oc_r    <= oc_base_r + sat_lane ;
        y_oh_r    <= oh_r ;
        y_ow_r    <= ow_r ;
    end else
        y_valid_r <= 1'b0 ;
end

endmodule
