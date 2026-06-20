// =============================================================================
// tail.v -- head tail (1x1 conv) ctr / offset / size + sigmoid on ctr/size
// -----------------------------------------------------------------------------
// numpy: run_backbone_numpy_shared_trunk.py L597-624
// tail_unit: K=1 PAD=0, 2-phase WPRE, 1-phase pipelined MAC, negedge bias_r
// tail top: ctr -> off -> size -> T_DRAIN_SIG -> done ; shared weight/bias ROM
// Shared ROMs use CLK(clk) (posedge, addr@T -> Q@T+1); tail_unit presents the WPRE
// addr combinationally in phase0 (tu_wpre_p0) so capture timing/cycle count are kept.
// =============================================================================

// tail_unit : single 1x1 conv (ROM addr includes WEIGHT_BASE / BIAS_BASE)
module tail_unit (
    clk      ,
    rst_n    ,
    start    ,
    busy     ,
    done     ,
    x_addr   ,
    x_i      ,
    w_addr   ,
    w_i      ,
    b_addr   ,
    b_i      ,
    y_valid  ,
    y_data   ,
    y_oc     ,
    y_oh     ,
    y_ow     ,
    mac_phase_o   ,
    mac_active_o  ,
    x_addr_mac_rd
);

parameter IN_CH       = 48 ;
parameter OUT_CH      = 1  ;
parameter IN_H        = 16 ;
parameter IN_W        = 16 ;
parameter WEIGHT_BASE = 0  ;
parameter BIAS_BASE   = 0  ;
parameter DATA_W      = 16 ;
parameter FRAC_W      = 8  ;
parameter ACC_W       = 32 ;
// 1: classic 2-phase MAC; 0: 1-phase pipelined
parameter MAC_2PHASE  = 0 ;
// 1: round(acc>>>FRAC_W) before +bias (numpy fp on tail conv); size u_siz only
parameter ROUND_Y     = 0 ;
// 1: size ch1 (oc!=0) +1 on pre-sat y (deprecated; use tail SIG_H_INC post-sigmoid)
parameter ROUND_H_INC = 0 ;

parameter X_AW    = 15 ;
parameter W_AW    = 8  ;
parameter B_AW    = 4  ;
parameter FEAT_AW = 6  ;
parameter OC_AW   = 2  ;
parameter HW_AW   = 5  ;

input                       clk     ;
input                       rst_n   ;
input                       start   ;
output                      busy    ;
output                      done    ;
output [X_AW-1:0]           x_addr  ;
input  signed [DATA_W-1:0]  x_i     ;
output [W_AW-1:0]           w_addr  ;
input  signed [DATA_W-1:0]  w_i     ;
output [B_AW-1:0]           b_addr  ;
input  signed [DATA_W-1:0]  b_i     ;
output                      y_valid ;
output signed [DATA_W-1:0]  y_data  ;
output [OC_AW-1:0]          y_oc    ;
output [HW_AW-1:0]          y_oh    ;
output [HW_AW-1:0]          y_ow    ;
output                      mac_phase_o   ;
output                      mac_active_o  ;
output [X_AW-1:0]           x_addr_mac_rd ;

parameter S_IDLE = 3'd0 ;
parameter S_WPRE = 3'd1 ;
parameter S_MAC  = 3'd2 ;
parameter S_SAT  = 3'd3 ;
parameter S_DONE = 3'd4 ;

reg  [2:0]               CS, NS ;
reg  [OC_AW-1:0]         oc_r ;
reg  [HW_AW-1:0]         oh_r, ow_r ;

reg                      wpre_phase ;
reg  [FEAT_AW-1:0]       wpre_feat ;
reg                      wpre_done ;
reg                      wpre_bias_ce ;

reg                      mac_fill ;
reg                      mac_phase ;
reg  [FEAT_AW-1:0]       mac_feat ;
reg                      mac_done ;

reg  signed [DATA_W-1:0] wgt_buf [0:IN_CH-1] ;
reg  signed [DATA_W-1:0] bias_r ;
reg  signed [ACC_W-1:0]  acc_r ;
reg  signed [ACC_W-1:0]  acc_sat_r ;

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

wire [FEAT_AW-1:0]       mac_ic ;
wire [HW_AW-1:0]         ih_r ;
wire [HW_AW-1:0]         iw_r ;
wire [X_AW-1:0]          x_addr_nxt ;
wire                     mac_feat_last ;

wire signed [DATA_W-1:0]       mac_x_op ;
wire signed [DATA_W-1:0]       mac_w_op ;
wire signed [2*DATA_W-1:0]     mac_prod ;
wire signed [ACC_W-1:0]        acc_next ;

// weight-prefetch phase0 (drive ROM addr combinationally for CLK(clk) macro)
wire tu_wpre_p0 = (CS == S_WPRE) && !wpre_done && (wpre_phase == 1'b0) ;

reg  signed [DATA_W-1:0]       y_sat ;

assign busy    = busy_r ;
assign done    = done_r ;
assign x_addr  = x_addr_r ;
// CLK(clk) shared ROM (tail top): present the prefetch addr combinationally during
// phase0 (one cycle earlier than registered w_addr_r/b_addr_r) so the posedge macro
// registers Q by the phase1 capture edge. w_addr_r/b_addr_r hold the same value in
// phase1; 2-phase WPRE cycle count is unchanged. (b_addr is constant per unit, so the
// negedge bias_r capture is unaffected either way.)
assign w_addr  = (tu_wpre_p0) ? (WEIGHT_BASE + oc_r * IN_CH + wpre_feat) : w_addr_r ;
assign b_addr  = (tu_wpre_p0) ? (BIAS_BASE + oc_r)                       : b_addr_r ;
assign y_valid = y_valid_r ;
assign y_data  = y_data_r ;
assign y_oc    = y_oc_r ;
assign y_oh    = y_oh_r ;
assign y_ow    = y_ow_r ;

assign mac_phase_o   = MAC_2PHASE ? mac_phase
                      : ((CS == S_MAC) && !mac_done && !mac_fill) ;
assign mac_active_o  = MAC_2PHASE ? ((CS == S_MAC) && !mac_done && !mac_phase)
                      : (CS == S_MAC) ;
wire [FEAT_AW-1:0]       mac_feat_rd   = mac_feat + 1 ;
wire [X_AW-1:0]          x_addr_rd_nxt = mac_feat_rd * (IN_H*IN_W) + ih_r * IN_W + iw_r ;
wire [X_AW-1:0] mac_rd_addr_1p = mac_fill ? x_addr_nxt :
                              mac_feat_last ? x_addr_r : x_addr_rd_nxt ;
assign x_addr_mac_rd = (CS == S_MAC && !mac_done)
                       ? (MAC_2PHASE ? ((!mac_phase) ? x_addr_nxt : x_addr_r)
                                     : mac_rd_addr_1p)
                       : x_addr_r ;

assign ow_last       = (ow_r == IN_W - 1) ;
assign oh_last       = (oh_r == IN_H - 1) ;
assign oc_last       = (oc_r == OUT_CH - 1) ;
assign mac_ic        = mac_feat ;
assign ih_r          = oh_r ;
assign iw_r          = ow_r ;
assign x_addr_nxt    = mac_ic * (IN_H*IN_W) + ih_r * IN_W + iw_r ;
assign mac_feat_last = (mac_feat == IN_CH - 1) ;

assign mac_x_op  = $signed(x_i) ;
assign mac_w_op  = wgt_buf[mac_feat] ;
assign mac_prod  = mac_x_op * mac_w_op ;
assign acc_next  = acc_r + mac_prod ;

wire signed [ACC_W-1:0] acc_shift_trunc = acc_sat_r >>> FRAC_W ;
wire signed [ACC_W-1:0] acc_shift_round =
    (acc_sat_r + (32'sd1 << (FRAC_W - 1))) >>> FRAC_W ;
wire signed [ACC_W-1:0] acc_shifted     = ROUND_Y ? acc_shift_round : acc_shift_trunc ;
wire signed [ACC_W-1:0] y_base   = acc_shifted + bias_r ;
wire signed [ACC_W-1:0] y_pre_sat = (ROUND_H_INC && (OUT_CH > 1) && (oc_r != {OC_AW{1'b0}}))
                                    ? (y_base + 32'sd1) : y_base ;

// FSM CS
always @(posedge clk) begin
    if (!rst_n)
        CS <= S_IDLE ;
    else
        CS <= NS ;
end

// FSM NS
always @(*) begin
    NS = CS ;
    case (CS)
        S_IDLE : if (start)     NS = S_WPRE ;
        S_WPRE : if (wpre_done) NS = S_MAC ;
        S_MAC  : if (mac_done)  NS = S_SAT ;
        S_SAT  : begin
            if (ow_last && oh_last && oc_last) NS = S_DONE ;
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

// oc_r, oh_r, ow_r
always @(posedge clk) begin
    if (!rst_n) begin
        oc_r <= 0 ;
        oh_r <= 0 ;
        ow_r <= 0 ;
    end else if (CS == S_IDLE) begin
        oc_r <= 0 ;
        oh_r <= 0 ;
        ow_r <= 0 ;
    end else if (CS == S_SAT && ow_last && oh_last && !oc_last) begin
        oc_r <= oc_r + 1 ;
        oh_r <= 0 ;
        ow_r <= 0 ;
    end else if (CS == S_SAT && ow_last && !oh_last) begin
        oh_r <= oh_r + 1 ;
        ow_r <= 0 ;
    end else if (CS == S_SAT && !ow_last)
        ow_r <= ow_r + 1 ;
end

// weight prefetch: wpre_*, w_addr_r, b_addr_r
always @(posedge clk) begin
    if (!rst_n) begin
        wpre_phase   <= 1'b0 ;
        wpre_feat    <= 0 ;
        wpre_done    <= 1'b0 ;
        wpre_bias_ce <= 1'b0 ;
        w_addr_r     <= 0 ;
        b_addr_r     <= 0 ;
    end else begin
        case (CS)
            S_IDLE : begin
                wpre_phase   <= 1'b0 ;
                wpre_feat    <= 0 ;
                wpre_done    <= 1'b0 ;
                wpre_bias_ce <= 1'b0 ;
            end
            S_WPRE : begin
                wpre_bias_ce <= 1'b0 ;
                if (wpre_phase == 1'b0) begin
                    w_addr_r   <= WEIGHT_BASE + oc_r * IN_CH + wpre_feat ;
                    b_addr_r   <= BIAS_BASE + oc_r ;
                    wpre_phase <= 1'b1 ;
                end else if (wpre_feat == IN_CH - 1) begin
                    wpre_done    <= 1'b1 ;
                    wpre_bias_ce <= 1'b1 ;
                    wpre_phase   <= 1'b0 ;
                end else begin
                    wpre_feat    <= wpre_feat + 1 ;
                    wpre_phase   <= 1'b0 ;
                end
            end
            S_SAT : begin
                if (ow_last && oh_last && !oc_last) begin
                    wpre_phase <= 1'b0 ;
                    wpre_feat  <= 0 ;
                    wpre_done  <= 1'b0 ;
                end
            end
            default : ;
        endcase
    end
end

// wgt_buf
always @(posedge clk) begin
    if (CS == S_WPRE && wpre_phase == 1'b1)
        wgt_buf[wpre_feat] <= w_i ;
end

// bias_r: async reset on negedge FF (reset -> CDN, not D-mux critical path)
always @(negedge clk or negedge rst_n) begin
    if (!rst_n)
        bias_r <= 16'sd0 ;
    else if (wpre_bias_ce)
        bias_r <= b_i ;
end

// MAC: mac_*, x_addr_r, acc_r, acc_sat_r
always @(posedge clk) begin
    if (!rst_n) begin
        mac_fill  <= 1'b1 ;
        mac_phase <= 1'b0 ;
        mac_feat  <= 0 ;
        mac_done  <= 1'b0 ;
        x_addr_r  <= 0 ;
        acc_r     <= 0 ;
        acc_sat_r <= 0 ;
    end else begin
        case (CS)
            S_WPRE : begin
                if (wpre_done) begin
                    mac_fill  <= 1'b1 ;
                    mac_phase <= 1'b0 ;
                    mac_feat  <= 0 ;
                    mac_done  <= 1'b0 ;
                    acc_r     <= 0 ;
                end
            end
            S_MAC : begin
                if (!mac_done && MAC_2PHASE && (mac_phase == 1'b0)) begin
                    x_addr_r  <= x_addr_nxt ;
                    mac_phase <= 1'b1 ;
                end else if (!mac_done && MAC_2PHASE && (mac_phase == 1'b1) && mac_feat_last) begin
                    acc_r     <= acc_next ;
                    mac_done  <= 1'b1 ;
                    acc_sat_r <= acc_next ;
                end else if (!mac_done && MAC_2PHASE && (mac_phase == 1'b1)) begin
                    acc_r     <= acc_next ;
                    mac_feat  <= mac_feat + 1 ;
                    mac_phase <= 1'b0 ;
                end else if (!mac_done && !MAC_2PHASE && mac_fill) begin
                    x_addr_r <= x_addr_nxt ;
                    mac_fill <= 1'b0 ;
                end else if (!mac_done && !MAC_2PHASE && mac_feat_last) begin
                    acc_r     <= acc_next ;
                    mac_done  <= 1'b1 ;
                    acc_sat_r <= acc_next ;
                end else if (!mac_done && !MAC_2PHASE) begin
                    acc_r    <= acc_next ;
                    mac_feat <= mac_feat + 1 ;
                end
            end
            S_SAT : begin
                mac_fill  <= 1'b1 ;
                mac_phase <= 1'b0 ;
                mac_feat  <= 0 ;
                mac_done  <= 1'b0 ;
                acc_r     <= 0 ;
            end
            default : ;
        endcase
    end
end

// y_sat (sat16, no ReLU)
always @(*) begin
    if (y_pre_sat >  32'sd32767)
        y_sat = 16'sh7fff ;
    else if (y_pre_sat < -32'sd32768)
        y_sat = 16'sh8000 ;
    else
        y_sat = y_pre_sat[DATA_W-1:0] ;
end

// y_valid_r, y_data_r, y_oc_r, y_oh_r, y_ow_r
always @(posedge clk) begin
    if (!rst_n) begin
        y_valid_r <= 1'b0 ;
        y_data_r  <= 0 ;
        y_oc_r    <= 0 ;
        y_oh_r    <= 0 ;
        y_ow_r    <= 0 ;
    end else if (CS == S_SAT) begin
        y_valid_r <= 1'b1 ;
        y_data_r  <= y_sat ;
        y_oc_r    <= oc_r ;
        y_oh_r    <= oh_r ;
        y_ow_r    <= ow_r ;
    end else
        y_valid_r <= 1'b0 ;
end

endmodule


// tail top: 3x tail_unit + shared ROM + sigmoid_lut (ctr, size)
module tail (
    clk           ,
    rst_n         ,
    start         ,
    busy          ,
    done          ,
    x_addr        ,
    x_i           ,
    mac_phase_o   ,
    mac_active_o  ,
    x_addr_mac_rd ,
    ctr_raw_y_valid ,
    ctr_raw_y_data  ,
    ctr_raw_y_oh    ,
    ctr_raw_y_ow    ,
    ctr_y_valid     ,
    ctr_y_data      ,
    off_y_valid     ,
    off_y_data      ,
    off_y_sub       ,
    off_y_oh        ,
    off_y_ow        ,
    size_raw_y_valid,
    size_raw_y_data ,
    size_raw_y_sub  ,
    size_raw_y_oh   ,
    size_raw_y_ow   ,
    size_y_valid    ,
    size_y_data
);

parameter DATA_W = 16 ;
parameter X_AW   = 15 ;

parameter CTR_WEIGHT_BASE = 0   ;
parameter OFF_WEIGHT_BASE = 48  ;
parameter SIZ_WEIGHT_BASE = 144 ;
parameter CTR_BIAS_BASE   = 0   ;
parameter OFF_BIAS_BASE   = 1   ;
parameter SIZ_BIAS_BASE   = 3   ;

input                       clk             ;
input                       rst_n           ;
input                       start           ;
output                      busy            ;
output                      done            ;
output [X_AW-1:0]           x_addr          ;
input  signed [DATA_W-1:0]  x_i             ;
output                      mac_phase_o     ;
output                      mac_active_o    ;
output [X_AW-1:0]           x_addr_mac_rd   ;

output                      ctr_raw_y_valid ;
output signed [DATA_W-1:0]  ctr_raw_y_data  ;
output [4:0]                ctr_raw_y_oh    ;
output [4:0]                ctr_raw_y_ow    ;

output                      ctr_y_valid     ;
output signed [DATA_W-1:0]  ctr_y_data      ;

output                      off_y_valid     ;
output signed [DATA_W-1:0]  off_y_data      ;
output                      off_y_sub       ;
output [4:0]                off_y_oh        ;
output [4:0]                off_y_ow        ;

output                      size_raw_y_valid ;
output signed [DATA_W-1:0]  size_raw_y_data  ;
output                      size_raw_y_sub   ;
output [4:0]                size_raw_y_oh    ;
output [4:0]                size_raw_y_ow    ;

output                      size_y_valid    ;
output signed [DATA_W-1:0]  size_y_data     ;

parameter T_IDLE      = 3'd0 ;
parameter T_CTR       = 3'd1 ;
parameter T_OFF       = 3'd2 ;
parameter T_SIZE      = 3'd3 ;
parameter T_DRAIN_SIG = 3'd4 ;
parameter T_DONE      = 3'd5 ;

reg  [2:0]                  TS, TNS ;
reg                         busy_r, done_r ;
reg                         ctr_start_r ;
reg                         off_start_r ;
reg                         siz_start_r ;
reg                         drain_cnt ;
reg                         siz_is_h_d1 ;  // delayed siz_yc[0] for h-ch sigmoid +1

wire [7:0]                  ctr_w_addr ;
wire [7:0]                  off_w_addr ;
wire [7:0]                  siz_w_addr ;
wire [3:0]                  ctr_b_addr ;
wire [3:0]                  off_b_addr ;
wire [3:0]                  siz_b_addr ;
wire signed [DATA_W-1:0]    w_i ;
wire signed [DATA_W-1:0]    b_i ;

wire [7:0]                  rom_w_a ;
wire [6:0]                  rom_b_a ;
wire signed [DATA_W-1:0]    rom_w_q ;
wire signed [DATA_W-1:0]    rom_b_q ;
wire                        rom_ceb_w ;
wire                        rom_ceb_b ;

wire [X_AW-1:0]             ctr_x_addr ;
wire [X_AW-1:0]             off_x_addr ;
wire [X_AW-1:0]             siz_x_addr ;
wire [X_AW-1:0]             x_addr_r ;

wire                        ctr_mac_phase ;
wire                        off_mac_phase ;
wire                        siz_mac_phase ;
wire                        ctr_mac_active ;
wire                        off_mac_active ;
wire                        siz_mac_active ;
wire [X_AW-1:0]             ctr_x_addr_mac ;
wire [X_AW-1:0]             off_x_addr_mac ;
wire [X_AW-1:0]             siz_x_addr_mac ;

wire                        ctr_busy, ctr_done, ctr_yv ;
wire signed [DATA_W-1:0]    ctr_yd ;
wire [1:0]                  ctr_yc ;
wire [4:0]                  ctr_yh, ctr_yw ;

wire                        off_busy, off_done, off_yv ;
wire signed [DATA_W-1:0]    off_yd ;
wire [1:0]                  off_yc ;
wire [4:0]                  off_yh, off_yw ;

wire                        siz_busy, siz_done, siz_yv ;
wire signed [DATA_W-1:0]    siz_yd ;
wire [1:0]                  siz_yc ;
wire [4:0]                  siz_yh, siz_yw ;

wire                        ctr_sig_v ;
wire signed [DATA_W-1:0]    ctr_sig_d ;
wire                        siz_sig_v ;
wire signed [DATA_W-1:0]    siz_sig_d ;

assign busy             = busy_r ;
assign done             = done_r ;
assign x_addr           = x_addr_r ;

assign ctr_raw_y_valid  = ctr_yv ;
assign ctr_raw_y_data   = ctr_yd ;
assign ctr_raw_y_oh     = ctr_yh ;
assign ctr_raw_y_ow     = ctr_yw ;

assign ctr_y_valid      = ctr_sig_v ;
assign ctr_y_data       = ctr_sig_d ;

assign off_y_valid      = off_yv ;
assign off_y_data       = off_yd ;
assign off_y_sub        = off_yc[0] ;
assign off_y_oh         = off_yh ;
assign off_y_ow         = off_yw ;

assign size_raw_y_valid = siz_yv ;
assign size_raw_y_data  = siz_yd ;
assign size_raw_y_sub   = siz_yc[0] ;
assign size_raw_y_oh    = siz_yh ;
assign size_raw_y_ow    = siz_yw ;

// h ch only: +1 LSB after sigmoid (numpy fp(sigmoid); w uses raw siz_sig_d)
always @(posedge clk) begin
    if (!rst_n)
        siz_is_h_d1 <= 1'b0 ;
    else if (siz_yv)
        siz_is_h_d1 <= siz_yc[0] ;
end

wire signed [DATA_W-1:0] siz_sig_h_bump =
    (siz_sig_d >= 16'sd255) ? 16'sd255 : (siz_sig_d + 16'sd1) ;

assign size_y_valid = siz_sig_v ;
assign size_y_data  = (siz_sig_v && siz_is_h_d1) ? siz_sig_h_bump : siz_sig_d ;

assign rom_w_a = (TS == T_CTR) ? ctr_w_addr :
                 (TS == T_OFF) ? off_w_addr :
                 (TS == T_SIZE) ? siz_w_addr : 8'd0 ;

assign rom_b_a = (TS == T_CTR) ? {3'b000, ctr_b_addr} :
                 (TS == T_OFF) ? {3'b000, off_b_addr} :
                 (TS == T_SIZE) ? {3'b000, siz_b_addr} : 7'd0 ;

assign x_addr_r = (TS == T_CTR) ? ctr_x_addr :
                  (TS == T_OFF) ? off_x_addr :
                  (TS == T_SIZE) ? siz_x_addr : {X_AW{1'b0}} ;

assign mac_phase_o = (TS == T_CTR) ? ctr_mac_phase :
                     (TS == T_OFF) ? off_mac_phase :
                     (TS == T_SIZE) ? siz_mac_phase : 1'b0 ;

assign mac_active_o = (TS == T_CTR) ? ctr_mac_active :
                      (TS == T_OFF) ? off_mac_active :
                      (TS == T_SIZE) ? siz_mac_active : 1'b0 ;

assign x_addr_mac_rd = (TS == T_CTR) ? ctr_x_addr_mac :
                       (TS == T_OFF) ? off_x_addr_mac :
                       (TS == T_SIZE) ? siz_x_addr_mac : {X_AW{1'b0}};

assign rom_ceb_w = !(ctr_busy || off_busy || siz_busy) ;
assign rom_ceb_b = !(ctr_busy || off_busy || siz_busy) ;

assign w_i = rom_w_q ;
assign b_i = rom_b_q ;

// FSM TS
always @(posedge clk) begin
    if (!rst_n)
        TS <= T_IDLE ;
    else
        TS <= TNS ;
end

// FSM TNS
always @(*) begin
    TNS = TS ;
    case (TS)
        T_IDLE      : if (start)        TNS = T_CTR ;
        T_CTR       : if (ctr_done)     TNS = T_OFF ;
        T_OFF       : if (off_done)     TNS = T_SIZE ;
        T_SIZE      : if (siz_done)     TNS = T_DRAIN_SIG ;
        T_DRAIN_SIG : if (drain_cnt)    TNS = T_DONE ;
        T_DONE      :                   TNS = T_IDLE ;
        default     :                   TNS = T_IDLE ;
    endcase
end

// drain_cnt
always @(posedge clk) begin
    if (!rst_n)
        drain_cnt <= 1'b0 ;
    else if (TS == T_SIZE && siz_done)
        drain_cnt <= 1'b0 ;
    else if (TS == T_DRAIN_SIG)
        drain_cnt <= 1'b1 ;
end

// busy_r, done_r, start pulses
always @(posedge clk) begin
    if (!rst_n) begin
        busy_r      <= 1'b0 ;
        done_r      <= 1'b0 ;
        ctr_start_r <= 1'b0 ;
        off_start_r <= 1'b0 ;
        siz_start_r <= 1'b0 ;
    end else begin
        busy_r <= (TNS != T_IDLE) && (TNS != T_DONE) ;
        done_r <= (TNS == T_DONE) ;
        ctr_start_r <= (TS == T_IDLE) && start ;
        off_start_r <= (TS == T_CTR)  && ctr_done ;
        siz_start_r <= (TS == T_OFF)  && off_done ;
    end
end

tail_unit #(
    .IN_CH        (48              ),
    .OUT_CH       (1               ),
    .IN_H         (16              ),
    .IN_W         (16              ),
    .WEIGHT_BASE  (CTR_WEIGHT_BASE ),
    .BIAS_BASE    (CTR_BIAS_BASE   ),
    .DATA_W       (DATA_W          ),
    .FRAC_W       (8               ),
    .ACC_W        (32              ),
    .X_AW         (X_AW            ),
    .W_AW         (8               ),
    .B_AW         (4               ),
    .FEAT_AW      (6               ),
    .OC_AW        (2               ),
    .HW_AW        (5               )
) u_ctr (
    .clk     (clk           ),
    .rst_n   (rst_n         ),
    .start   (ctr_start_r   ),
    .busy    (ctr_busy      ),
    .done    (ctr_done      ),
    .x_addr  (ctr_x_addr    ),
    .x_i     (x_i           ),
    .w_addr  (ctr_w_addr    ),
    .w_i     (w_i           ),
    .b_addr  (ctr_b_addr    ),
    .b_i     (b_i           ),
    .y_valid (ctr_yv        ),
    .y_data  (ctr_yd        ),
    .y_oc    (ctr_yc        ),
    .y_oh    (ctr_yh        ),
    .y_ow    (ctr_yw        ),
    .mac_phase_o   (ctr_mac_phase   ),
    .mac_active_o  (ctr_mac_active  ),
    .x_addr_mac_rd (ctr_x_addr_mac  )
);

tail_unit #(
    .IN_CH        (48              ),
    .OUT_CH       (2               ),
    .IN_H         (16              ),
    .IN_W         (16              ),
    .WEIGHT_BASE  (OFF_WEIGHT_BASE ),
    .BIAS_BASE    (OFF_BIAS_BASE   ),
    .DATA_W       (DATA_W          ),
    .FRAC_W       (8               ),
    .ACC_W        (32              ),
    .X_AW         (X_AW            ),
    .W_AW         (8               ),
    .B_AW         (4               ),
    .FEAT_AW      (6               ),
    .OC_AW        (2               ),
    .HW_AW        (5               )
) u_off (
    .clk     (clk           ),
    .rst_n   (rst_n         ),
    .start   (off_start_r   ),
    .busy    (off_busy      ),
    .done    (off_done      ),
    .x_addr  (off_x_addr    ),
    .x_i     (x_i           ),
    .w_addr  (off_w_addr    ),
    .w_i     (w_i           ),
    .b_addr  (off_b_addr    ),
    .b_i     (b_i           ),
    .y_valid (off_yv        ),
    .y_data  (off_yd        ),
    .y_oc    (off_yc        ),
    .y_oh    (off_yh        ),
    .y_ow    (off_yw        ),
    .mac_phase_o   (off_mac_phase   ),
    .mac_active_o  (off_mac_active  ),
    .x_addr_mac_rd (off_x_addr_mac  )
);

tail_unit #(
    .IN_CH        (48              ),
    .OUT_CH       (2               ),
    .IN_H         (16              ),
    .IN_W         (16              ),
    .WEIGHT_BASE  (SIZ_WEIGHT_BASE ),
    .BIAS_BASE    (SIZ_BIAS_BASE   ),
    .DATA_W       (DATA_W          ),
    .FRAC_W       (8               ),
    .ACC_W        (32              ),
    .X_AW         (X_AW            ),
    .W_AW         (8               ),
    .B_AW         (4               ),
    .FEAT_AW      (6               ),
    .OC_AW        (2               ),
    .HW_AW        (5               ),
    .ROUND_Y      (1               ),
    .ROUND_H_INC  (0               )
) u_siz (
    .clk     (clk           ),
    .rst_n   (rst_n         ),
    .start   (siz_start_r   ),
    .busy    (siz_busy      ),
    .done    (siz_done      ),
    .x_addr  (siz_x_addr    ),
    .x_i     (x_i           ),
    .w_addr  (siz_w_addr    ),
    .w_i     (w_i           ),
    .b_addr  (siz_b_addr    ),
    .b_i     (b_i           ),
    .y_valid (siz_yv        ),
    .y_data  (siz_yd        ),
    .y_oc    (siz_yc        ),
    .y_oh    (siz_yh        ),
    .y_ow    (siz_yw        ),
    .mac_phase_o   (siz_mac_phase   ),
    .mac_active_o  (siz_mac_active  ),
    .x_addr_mac_rd (siz_x_addr_mac  )
);

sigmoid_lut #(.DATA_W(DATA_W)) u_sig_ctr (
    .clk      (clk      ),
    .rst_n    (rst_n    ),
    .in_valid (ctr_yv   ),
    .in_q88   (ctr_yd   ),
    .out_valid(ctr_sig_v),
    .out_q88  (ctr_sig_d)
);

sigmoid_lut #(.DATA_W(DATA_W)) u_sig_siz (
    .clk      (clk      ),
    .rst_n    (rst_n    ),
    .in_valid (siz_yv   ),
    .in_q88   (siz_yd   ),
    .out_valid(siz_sig_v),
    .out_q88  (siz_sig_d)
);

rom_box_head_tail_ctr_offset_size_weight u_rom_w (
    .A(rom_w_a), .AM(8'b0), .CEBM(), .BIST(1'b0),
    .CEB(rom_ceb_w), .CLK(clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(rom_w_q));

rom_box_head_tail_ctr_offset_size_bias u_rom_b (
    .A(rom_b_a), .AM(7'b0), .CEBM(), .BIST(1'b0),
    .CEB(rom_ceb_b), .CLK(clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(rom_b_q));

endmodule
