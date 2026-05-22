// =============================================================================
// conv.v  --  Generic Conv2D (Q8.8 fixed point, FSM multi-cycle, ROM prefetch)
// -----------------------------------------------------------------------------
// 對應 numpy: run_backbone_numpy_shared_trunk.py -> conv2d() + bias + (ReLU) + fp()
//   acc(int) = sum_{ic,kh,kw}(x_int * w_int) , 32-bit signed
//   acc_q88  = acc >>> FRAC_W          (arithmetic shift; matches numpy ">> 8")
//   y_raw    = acc_q88 + bias_int       (Q8.8)
//   y_data   = HAS_RELU ? max(0, sat16(y_raw)) : sat16(y_raw)
//
// Tensor shapes:
//   Input  : [1, IN_CH , IN_H , IN_W ]   addr = ic*IN_H*IN_W + ih*IN_W + iw
//   Weight : [OUT_CH, IN_CH, K, K]       addr = ((oc*IN_CH + ic)*K + kh)*K + kw
//   Bias   : [OUT_CH]                    addr = oc
//   Output : [1, OUT_CH, OUT_H, OUT_W]   y emitted in order (oc, oh, ow)
//   OUT_H  = IN_H + 2*PAD - K + 1        (OUT_W likewise)
//
// shared_conv1 : IN_CH=32  OUT_CH=96   ROM_PROFILE=1  Golden Activation/box_head_shared_after_conv1_out_bi.txt
// shared_conv2 : IN_CH=96  OUT_CH=48   ROM_PROFILE=2  Golden Activation/box_head_shared_after_conv2_out_bi.txt
//   conv2 weight ROM split @16384/@32768 (head_top wa_c2_w[15:14]) ; bias addr 96+oc in c12b ROM
// TB 只接 x_addr/x_i ；權重/bias ROM + MAC 均在模組內。
//
// ROM read contract (verilog_rule.mdc §7.1 ; ROM macro CLK(~clk)) :
//   posedge T   : 送 w_addr / b_addr （feat/oc 由當拍 reg 決定）
//   posedge T+1 : w_i / b_i 為 T 拍位址讀出資料；本拍才允許 wgt_buf[k] / bias_r 鎖入
//
// Prefetch pattern (clear 2-tick per weight, §7.2 conservative form) :
//   stream_phase==0 (addr tick) : w_addr_r <= base+feat ; b_addr_r <= oc ; phase->1
//   stream_phase==1 (data tick) : wgt_buf[feat] <= w_i ;
//                                 if feat==0 : bias_r <= b_i (§7.3)
//                                 feat++ ; phase->0
//   Cleanly avoids §7.5 反模式 : 不雙寫 slot0 / 不在 MAC 每拍清 acc
//
// Input SRAM read contract (treat input mem like ROM, 1-cycle latency) :
//   posedge T   : 送 x_addr
//   posedge T+1 : x_i 為 T 拍位址讀出 ; 本拍才允許進 MAC
// =============================================================================

module conv (
    clk      ,
    rst_n    ,
    start    ,
    busy     ,
    done     ,
    // input feature memory (1-cycle read latency, see contract above)
    x_addr   ,
    x_i      ,
    // output stream (oc, oh, ow row-major)
    y_valid  ,
    y_data   ,
    y_oc     ,
    y_oh     ,
    y_ow
);

// ---------------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------------
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

// derived
parameter OUT_H       = IN_H + 2*PAD - K + 1 ;
parameter OUT_W       = IN_W + 2*PAD - K + 1 ;
parameter KK          = K*K ;
parameter FEAT_PER_OC = IN_CH * KK ;             // 32*9 = 288 for conv1
parameter W_DEPTH     = OUT_CH * FEAT_PER_OC ;   // 27648 for conv1

// width helpers (fixed widths big enough for current targets)
parameter X_AW    = 14 ;   // conv1: 8192 ; conv2 instance override to 15 (96*16*16)
parameter W_AW    = 16 ;   // conv2 max addr 41471
parameter B_AW    = 8  ;
parameter FEAT_AW = 10 ;   // conv2 FEAT_PER_OC=864
parameter OC_AW   = 8  ;
parameter HW_AW   = 5  ;
parameter ROM_PROFILE = 1 ;   // 1=conv1 ROM, 2=conv2 ROM

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// FSM state encoding
// ---------------------------------------------------------------------------
parameter S_IDLE = 3'd0 ;
parameter S_WPRE = 3'd1 ;   // weight/bias prefetch for current oc
parameter S_MAC  = 3'd2 ;   // walk (ic, kh, kw) for current (oc, oh, ow)
parameter S_SAT  = 3'd3 ;   // saturate, add bias, ReLU, emit y_valid
parameter S_DONE = 3'd4 ;

// ---------------------------------------------------------------------------
// Reg / wire declarations (Verilog-2005, no decl inside always)
// ---------------------------------------------------------------------------
reg  [2:0]              state, next_state ;
reg  [OC_AW-1:0]        oc_r ;
reg  [HW_AW-1:0]        oh_r, ow_r ;

// last-tick combinational flags
reg                     ow_last  ;
reg                     oh_last  ;
reg                     oc_last  ;
// weight prefetch (2-phase)
reg                     wpre_phase ;       // 0=address tick, 1=data tick
reg  [FEAT_AW-1:0]      wpre_feat ;        // 0..FEAT_PER_OC-1
reg                     wpre_done ;        // last data tick latched

// MAC
reg                     mac_phase ;        // 0=address tick, 1=data tick
reg  [FEAT_AW-1:0]      mac_feat ;
reg                     mac_done ;         // last MAC accumulation latched

// data path
reg  signed [DATA_W-1:0] wgt_buf [0:FEAT_PER_OC-1] ;
reg  signed [DATA_W-1:0] bias_r ;
reg  signed [ACC_W-1:0]  acc_r ;
reg  signed [ACC_W-1:0]  acc_sat_r ;   // MAC sum latched at last tap (held through S_SAT)
reg                    wpre_bias_ce ; // pulse: negedge after last WPRE data tick latches bias_r
reg                      x_in_pad ;        // current MAC tap is in zero-pad

// address regs
reg  [X_AW-1:0]          x_addr_r ;
reg  [W_AW-1:0]          w_addr_r ;
reg  [B_AW-1:0]          b_addr_r ;

// output stream regs
reg                      y_valid_r ;
reg  signed [DATA_W-1:0] y_data_r ;
reg  [OC_AW-1:0]         y_oc_r ;
reg  [HW_AW-1:0]         y_oh_r, y_ow_r ;

// busy/done
reg                      busy_r, done_r ;

// MAC decoder (combinational)
reg  [6:0]               mac_ic ;
reg  [3:0]               mac_kh ;
reg  [3:0]               mac_kw ;
reg  signed [HW_AW:0]    ih_s ;
reg  signed [HW_AW:0]    iw_s ;
reg  [X_AW-1:0]          x_addr_nxt ;
reg                      pad_nxt ;
reg                      mac_feat_last ;

// MAC datapath (combinational)
reg  signed [DATA_W-1:0] mac_x_op ;
reg  signed [DATA_W-1:0] mac_w_op ;
reg  signed [2*DATA_W-1:0] mac_prod ;
reg  signed [ACC_W-1:0]  acc_next ;

// saturation (combinational)
reg  signed [ACC_W-1:0]  acc_shifted ;
reg  signed [ACC_W-1:0]  y_pre_sat ;
reg  signed [DATA_W-1:0] y_sat ;
reg  signed [DATA_W-1:0] y_relu ;

// Saturation constants are inlined as 32-bit signed literals in the
// saturation always-block below (ACC_W=32 / DATA_W=16 assumed).

// ---------------------------------------------------------------------------
// §12.1 weight / bias ROM (inside conv; decode matches head_top S_CONV1 / S_CONV2)
// ---------------------------------------------------------------------------
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
wire signed [15:0]   b_i ;

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

assign b_i = rom_c12b_q ;


// ---------------------------------------------------------------------------
// Output assigns
// ---------------------------------------------------------------------------
assign busy    = busy_r ;
assign done    = done_r ;
assign x_addr  = x_addr_r ;
assign y_valid = y_valid_r ;
assign y_data  = y_data_r ;
assign y_oc    = y_oc_r ;
assign y_oh    = y_oh_r ;
assign y_ow    = y_ow_r ;

// ---------------------------------------------------------------------------
// (comb) last-tick flags
// ---------------------------------------------------------------------------
always @(*) begin
    ow_last = (ow_r == OUT_W - 1) ;
    oh_last = (oh_r == OUT_H - 1) ;
    oc_last = (oc_r == OUT_CH - 1) ;
end

// ---------------------------------------------------------------------------
// (seq) FSM state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) state <= S_IDLE ;
    else        state <= next_state ;
end

// ---------------------------------------------------------------------------
// (comb) FSM next-state
// ---------------------------------------------------------------------------
always @(*) begin
    next_state = state ;
    case (state)
        S_IDLE : if (start)     next_state = S_WPRE ;
        S_WPRE : if (wpre_done) next_state = S_MAC ;
        S_MAC  : if (mac_done)  next_state = S_SAT ;
        S_SAT  : begin
            if (ow_last && oh_last && oc_last) next_state = S_DONE ;
            else if (ow_last && oh_last)       next_state = S_WPRE ;
            else                               next_state = S_MAC ;
        end
        S_DONE :                next_state = S_IDLE ;
        default :               next_state = S_IDLE ;
    endcase
end

// ---------------------------------------------------------------------------
// (seq) busy / done
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        busy_r <= 1'b0 ;
        done_r <= 1'b0 ;
    end else begin
        busy_r <= (next_state != S_IDLE) && (next_state != S_DONE) ;
        done_r <= (next_state == S_DONE) ;
    end
end

// ---------------------------------------------------------------------------
// (seq) Output channel / spatial counters (oc, oh, ow)
//   Advance on S_SAT (oh/ow still at last pixel when wpre re-arms for next oc)
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        oc_r <= 0 ;
        oh_r <= 0 ;
        ow_r <= 0 ;
    end else if (state == S_IDLE) begin
        oc_r <= 0 ;
        oh_r <= 0 ;
        ow_r <= 0 ;
    end else if (state == S_SAT) begin
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

// ---------------------------------------------------------------------------
// (seq) Weight / bias prefetch  (2-phase per weight)
//   phase 0 : issue w_addr / b_addr ; phase <- 1
//   phase 1 : latch w_i into wgt_buf[wpre_feat] ;
//             if wpre_feat==0 : bias_r <= b_i (§7.3)
//             wpre_feat <- wpre_feat+1 (unless last) ; phase <- 0
//   wpre_done pulses when last data tick latched -> FSM goes to S_MAC.
//
//   NOTE: do NOT clear wgt_buf between oh/ow within same oc; only when entering
//         a new oc (state==S_SAT && ow_last && oh_last && !oc_last) do we
//         reset wpre_feat/phase to start a fresh prefetch.
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        wpre_phase <= 1'b0 ;
        wpre_feat  <= 0 ;
        wpre_done  <= 1'b0 ;
        w_addr_r   <= 0 ;
        b_addr_r   <= 0 ;
        bias_r       <= 0 ;
        wpre_bias_ce <= 1'b0 ;
    end else begin
        case (state)
            S_IDLE : begin
                wpre_phase   <= 1'b0 ;
                wpre_feat    <= 0 ;
                wpre_done    <= 1'b0 ;
                wpre_bias_ce <= 1'b0 ;
            end
            S_WPRE : begin
                wpre_bias_ce <= 1'b0 ;
                if (wpre_phase == 1'b0) begin
                    // address tick
                    w_addr_r   <= oc_r * FEAT_PER_OC + wpre_feat ;
                    b_addr_r   <= oc_r ;
                    wpre_phase <= 1'b1 ;
                end else begin
                    // data tick: latch wgt_buf[wpre_feat]; bias via negedge ROM sample
                    wgt_buf[wpre_feat] <= w_i ;
                    if (wpre_feat == FEAT_PER_OC - 1) begin
                        wpre_done    <= 1'b1 ;
                        wpre_bias_ce <= 1'b1 ;
                    end else begin
                        wpre_feat  <= wpre_feat + 1 ;
                    end
                    wpre_phase <= 1'b0 ;
                end
            end
            S_SAT : begin
                // entering next oc -> rearm prefetch
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

// Bias ROM read aligns with CLK(~clk): latch c12b_q on negedge after last WPRE data tick
// (same idea as head_top bias_q_lat on negedge; avoids posedge b_i / rom_c12b_q skew)
always @(negedge clk) begin
    if (!rst_n)
        bias_r <= 16'sd0 ;
    else if (wpre_bias_ce)
        bias_r <= rom_c12b_q ;
end

// ---------------------------------------------------------------------------
// (comb) MAC tap decoder from mac_feat (used in address tick)
// ---------------------------------------------------------------------------
always @(*) begin
    mac_ic = mac_feat / KK ;
    mac_kh = (mac_feat % KK) / K ;
    mac_kw = (mac_feat % KK) % K ;
    ih_s   = $signed({1'b0, oh_r}) + $signed({2'b00, mac_kh}) - PAD ;
    iw_s   = $signed({1'b0, ow_r}) + $signed({2'b00, mac_kw}) - PAD ;
    pad_nxt = (ih_s < 0) || (ih_s >= IN_H) || (iw_s < 0) || (iw_s >= IN_W) ;
    // when pad, x_addr is irrelevant (we'll mask x to 0); set 0 to avoid OOB
    if (pad_nxt)
        x_addr_nxt = {X_AW{1'b0}} ;
    else
        x_addr_nxt = mac_ic * (IN_H*IN_W)
                   + ih_s[HW_AW-1:0] * IN_W
                   + iw_s[HW_AW-1:0] ;
    mac_feat_last = (mac_feat == FEAT_PER_OC - 1) ;
end

// ---------------------------------------------------------------------------
// (seq) MAC datapath  (2-phase per tap)
//   phase 0 : x_addr_r <= x_addr_nxt ; x_in_pad <= pad_nxt ; phase <- 1
//   phase 1 : acc_r   <= acc_next      (uses x_i / wgt_buf[mac_feat])
//             if mac_feat==last : mac_done<=1 (state will move to S_SAT)
//             else              : mac_feat <- mac_feat+1 ; phase <- 0
//   acc_r cleared at WPRE->MAC edge ; never per-cycle in MAC (§7.5)
// ---------------------------------------------------------------------------
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
        case (state)
            S_WPRE : begin
                // arm MAC: clear at WPRE->MAC edge
                if (wpre_done) begin
                    mac_phase <= 1'b0 ;
                    mac_feat  <= 0 ;
                    mac_done  <= 1'b0 ;
                    acc_r     <= 0 ;
                end
            end
            S_MAC : begin
                // freeze datapath once final MAC latched; FSM has 1-cycle delay
                // before transitioning to S_SAT, so prevent re-accumulation.
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
                // rearm for next (oh,ow) on same oc (or stay clean for next oc)
                mac_phase <= 1'b0 ;
                mac_feat  <= 0 ;
                mac_done  <= 1'b0 ;
                acc_r     <= 0 ;
            end
            default : ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// (comb) MAC arithmetic
//   x_i is the data read for x_addr issued last cycle (1-cycle latency).
//   For zero-pad taps we ignore x_i and use 0.
//   wgt_buf[mac_feat] : already loaded during WPRE, indexed in this data tick.
// ---------------------------------------------------------------------------
always @(*) begin
    mac_x_op = x_in_pad ? {DATA_W{1'b0}} : $signed(x_i) ;
    mac_w_op = wgt_buf[mac_feat] ;
    mac_prod = mac_x_op * mac_w_op ;        // signed[2*DATA_W-1:0]
    acc_next = acc_r + mac_prod ;           // both signed ; auto sign-extend
end

// ---------------------------------------------------------------------------
// (comb) Saturation / bias add / ReLU  (acc_sat_r latched at last MAC data tick)
// ---------------------------------------------------------------------------
always @(*) begin
    acc_shifted = acc_sat_r >>> FRAC_W ;
    y_pre_sat   = acc_shifted + bias_r ;              // both signed
    if (y_pre_sat >  32'sd32767)
        y_sat = 16'sh7fff ;                           // +32767
    else if (y_pre_sat < -32'sd32768)
        y_sat = 16'sh8000 ;                           // -32768
    else
        y_sat = y_pre_sat[DATA_W-1:0] ;
    if ((HAS_RELU != 0) && y_sat[DATA_W-1])
        y_relu = {DATA_W{1'b0}} ;
    else
        y_relu = y_sat ;
end

// ---------------------------------------------------------------------------
// (seq) y_valid pulse + y_data register  (emit in S_SAT after acc_sat_r latched)
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        y_valid_r <= 1'b0 ;
        y_data_r  <= 0 ;
        y_oc_r    <= 0 ;
        y_oh_r    <= 0 ;
        y_ow_r    <= 0 ;
    end else begin
        if (state == S_SAT) begin
            y_valid_r <= 1'b1 ;
            y_data_r  <= y_relu ;
            y_oc_r    <= oc_r ;
            y_oh_r    <= oh_r ;
            y_ow_r    <= ow_r ;
        end else begin
            y_valid_r <= 1'b0 ;
        end
    end
end

// ---------------------------------------------------------------------------
// Debug $display (sim only, English ASCII per §10.1 ; default off)
//   Compile: +define+DUMP_CONV1_DBG  (includes light [Y] + diagnosis tags below)
//   Optional: +define+DUMP_CONV1_WGT_FULL for every [WGT_WR] line (large log)
//   Goal: find first X on bias / acc / x_i / y_relu when [Y] shows xxxx
// ---------------------------------------------------------------------------
`ifdef DUMP_CONV1_DBG

// [BIAS_NEG] bias latched on negedge after last WPRE feat (per oc)
always @(negedge clk) begin
    if (wpre_bias_ce) begin
        $display("[BIAS_NEG] oc=%0d c12b_a=%0d rom_c12b_q=%h bias_r_next=%h ceb_b=%0d",
                 oc_r, rom_c12b_a, rom_c12b_q, rom_c12b_q, rom_ceb_b) ;
    end
end

// [MAC_END] last MAC tap -> acc_sat_r (per oc,oh,ow)
always @(posedge clk) begin
    if (state == S_MAC && mac_phase == 1'b1 && mac_feat_last) begin
        $display("[MAC_END] oc=%0d oh=%0d ow=%0d acc_next=%h acc_sat_r_old=%h mac_done_next=1",
                 oc_r, oh_r, ow_r, acc_next, acc_sat_r) ;
    end
end

// [SAT] emit cycle: all terms into y_relu (flag if any X)
always @(posedge clk) begin
    if (state == S_SAT) begin
        $display("[SAT] oc=%0d oh=%0d ow=%0d acc_sat_r=%h bias_r=%h acc_sh=%h y_pre=%h y_relu=%h y_data_r=%h x_unknown=%0d",
                 oc_r, oh_r, ow_r, acc_sat_r, bias_r, acc_shifted, y_pre_sat, y_relu, y_data_r,
                 ((^{acc_sat_r,bias_r,y_relu,y_data_r}) === 1'bx) ? 1 : 0) ;
    end
end

// [MAC0] first MAC tap only (x_i / pad / first weight)
always @(posedge clk) begin
    if (state == S_MAC && mac_phase == 1'b1 && mac_feat == 0) begin
        $display("[MAC0] oc=%0d oh=%0d ow=%0d pad=%0d x_addr=%0d x_i=%h wgt0=%h",
                 oc_r, oh_r, ow_r, x_in_pad, x_addr_r, x_i, wgt_buf[0]) ;
    end
end

// [Y] every output (disable by not defining DUMP_CONV1_DBG if too verbose)
always @(posedge clk) begin
    if (y_valid_r) begin
        $display("[Y] oc=%0d oh=%0d ow=%0d y=%h", y_oc_r, y_oh_r, y_ow_r, y_data_r) ;
    end
end

`endif

`ifdef DUMP_CONV1_WGT_FULL
always @(posedge clk) begin
    if (state == S_WPRE && wpre_phase == 1'b1) begin
        $display("[WGT_WR] oc=%0d feat=%0d w_i=%h", oc_r, wpre_feat, w_i) ;
    end
end
`endif

`ifdef DUMP_CONV1_CHK
// legacy alias: WGT_CHK + short [Y] on unknown only
always @(posedge clk) begin
    if (state == S_WPRE && wpre_done) begin
        $display("[WGT_CHK] oc=%0d buf[0..3]=%h %h %h %h",
                 oc_r, wgt_buf[0], wgt_buf[1], wgt_buf[2], wgt_buf[3]) ;
    end
end
always @(posedge clk) begin
    if (y_valid_r && ((^{y_data_r}) === 1'bx)) begin
        $display("[Y_X] oc=%0d oh=%0d ow=%0d y=%h", y_oc_r, y_oh_r, y_ow_r, y_data_r) ;
    end
end
`endif

endmodule
