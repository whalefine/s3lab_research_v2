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
// WPRE/MAC 各 2-phase；語意同前版，組合邏輯用 wire，時序按功能分組 always。
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
    wgt_rd_i
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

reg                      mac_phase ;
reg  [FEAT_AW-1:0]       mac_feat ;
reg                      mac_done ;

reg  signed [DATA_W-1:0] bias_r ;
reg  signed [ACC_W-1:0]  acc_r ;
reg  signed [ACC_W-1:0]  acc_sat_r ;
reg                      x_in_pad ;

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

wire [6:0]               mac_ic ;
wire [3:0]               mac_kh ;
wire [3:0]               mac_kw ;
wire signed [HW_AW:0]    ih_s ;
wire signed [HW_AW:0]    iw_s ;
wire [X_AW-1:0]          x_addr_nxt ;
wire                     pad_nxt ;
wire                     mac_feat_last ;

wire signed [DATA_W-1:0]       mac_x_op ;
wire signed [DATA_W-1:0]       mac_w_op ;
wire signed [2*DATA_W-1:0]     mac_prod ;
wire signed [ACC_W-1:0]        acc_next ;

wire signed [ACC_W-1:0]        acc_shifted ;
wire signed [ACC_W-1:0]        y_pre_sat ;
reg  signed [DATA_W-1:0]       y_sat ;
reg  signed [DATA_W-1:0]       y_relu ;

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
assign mac_phase_o   = mac_phase ;
// MAC phase0 posedge: prefetch x for phase1 MAC (parent SRAM read addr)
assign x_addr_mac_rd = (CS == S_MAC && !mac_done && (mac_phase == 1'b0))
                       ? x_addr_nxt : x_addr_r ;
assign wgt_wr_en   = (CS == S_WPRE) && (wpre_phase == 1'b1) ;
assign wgt_wr_addr = wpre_feat ;
assign wgt_wr_data = w_i ;
assign wgt_rd_req  = (CS == S_MAC) && !mac_done && (mac_phase == 1'b0) ;
assign wgt_rd_addr = mac_feat ;
assign y_valid = y_valid_r ;
assign y_data  = y_data_r ;
assign y_oc    = y_oc_r ;
assign y_oh    = y_oh_r ;
assign y_ow    = y_ow_r ;

assign ow_last       = (ow_r == OUT_W - 1) ;
assign oh_last       = (oh_r == OUT_H - 1) ;
assign oc_last       = (oc_r == OUT_CH - 1) ;
assign mac_ic        = mac_feat / KK ;
assign mac_kh        = (mac_feat % KK) / K ;
assign mac_kw        = (mac_feat % KK) % K ;
assign ih_s          = $signed({1'b0, oh_r}) + $signed({2'b00, mac_kh}) - PAD ;
assign iw_s          = $signed({1'b0, ow_r}) + $signed({2'b00, mac_kw}) - PAD ;
assign pad_nxt       = (ih_s < 0) || (ih_s >= IN_H) || (iw_s < 0) || (iw_s >= IN_W) ;
assign x_addr_nxt    = pad_nxt ? {X_AW{1'b0}} :
                       (mac_ic * (IN_H*IN_W) + ih_s[HW_AW-1:0] * IN_W + iw_s[HW_AW-1:0]) ;
assign mac_feat_last = (mac_feat == FEAT_PER_OC - 1) ;

assign mac_x_op  = x_in_pad ? {DATA_W{1'b0}} : $signed(x_i) ;
assign mac_w_op  = (CS == S_MAC && mac_phase) ? $signed(wgt_rd_i) : {DATA_W{1'b0}} ;
assign mac_prod  = mac_x_op * mac_w_op ;
assign acc_next  = acc_r + mac_prod ;

assign acc_shifted = acc_sat_r >>> FRAC_W ;
assign y_pre_sat   = acc_shifted + bias_r ;

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
    end else if (CS == S_SAT) begin
        if (ow_last && oh_last && !oc_last) begin
            oc_r <= oc_r + 1 ;
            oh_r <= 0 ;
            ow_r <= 0 ;
        end else if (ow_last && !oh_last) begin
            oh_r <= oh_r + 1 ;
            ow_r <= 0 ;
        end else if (!ow_last) begin
            ow_r <= ow_r + 1 ;
        end
    end
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
                    w_addr_r   <= oc_r * FEAT_PER_OC + wpre_feat ;
                    b_addr_r   <= oc_r ;
                    wpre_phase <= 1'b1 ;
                end else begin
                    if (wpre_feat == FEAT_PER_OC - 1) begin
                        wpre_done    <= 1'b1 ;
                        wpre_bias_ce <= 1'b1 ;
                    end else
                        wpre_feat <= wpre_feat + 1 ;
                    wpre_phase <= 1'b0 ;
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

// bias_r
always @(negedge clk) begin
    if (!rst_n)
        bias_r <= 16'sd0 ;
    else if (wpre_bias_ce)
        bias_r <= rom_c12b_q ;
end

// MAC: mac_*, x_addr_r, x_in_pad, acc_r, acc_sat_r
always @(posedge clk) begin
    if (!rst_n) begin
        mac_phase <= 1'b0 ;
        mac_feat  <= 0 ;
        mac_done  <= 1'b0 ;
        x_addr_r  <= 0 ;
        x_in_pad  <= 1'b0 ;
        acc_r     <= 0 ;
        acc_sat_r <= 0 ;
    end else begin
        case (CS)
            S_WPRE : begin
                if (wpre_done) begin
                    mac_phase <= 1'b0 ;
                    mac_feat  <= 0 ;
                    mac_done  <= 1'b0 ;
                    acc_r     <= 0 ;
                end
            end
            S_MAC : begin
                if (!mac_done) begin
                    if (mac_phase == 1'b0) begin
                        x_addr_r  <= x_addr_nxt ;
                        x_in_pad  <= pad_nxt ;
                        mac_phase <= 1'b1 ;
                    end else begin
                        acc_r <= acc_next ;
                        if (mac_feat_last) begin
                            mac_done  <= 1'b1 ;
                            acc_sat_r <= acc_next ;
                        end else begin
                            mac_feat  <= mac_feat + 1 ;
                            mac_phase <= 1'b0 ;
                        end
                    end
                end
            end
            S_SAT : begin
                mac_phase <= 1'b0 ;
                mac_feat  <= 0 ;
                mac_done  <= 1'b0 ;
                acc_r     <= 0 ;
            end
            default : ;
        endcase
    end
end

// y_relu (sat16 + optional ReLU)
always @(*) begin
    if (y_pre_sat >  32'sd32767)
        y_sat = 16'sh7fff ;
    else if (y_pre_sat < -32'sd32768)
        y_sat = 16'sh8000 ;
    else
        y_sat = y_pre_sat[DATA_W-1:0] ;
    if ((HAS_RELU != 0) && y_sat[DATA_W-1])
        y_relu = {DATA_W{1'b0}} ;
    else
        y_relu = y_sat ;
end

// y_valid_r, y_data_r, y_oc_r, y_oh_r, y_ow_r
always @(posedge clk) begin
    if (!rst_n) begin
        y_valid_r <= 1'b0 ;
        y_data_r  <= 0 ;
        y_oc_r    <= 0 ;
        y_oh_r    <= 0 ;
        y_ow_r    <= 0 ;
    end else begin
        if (CS == S_SAT) begin
            y_valid_r <= 1'b1 ;
            y_data_r  <= y_relu ;
            y_oc_r    <= oc_r ;
            y_oh_r    <= oh_r ;
            y_ow_r    <= ow_r ;
        end else
            y_valid_r <= 1'b0 ;
    end
end

endmodule
