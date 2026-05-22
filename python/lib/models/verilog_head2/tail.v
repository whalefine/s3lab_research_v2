// =============================================================================
// tail.v -- head tail (1x1 conv) ctr / offset / size + sigmoid_clamped on ctr/size
// -----------------------------------------------------------------------------
// 對應 numpy : run_backbone_numpy_shared_trunk.py L597-624 (head_shared_trunk tail)
//
//   raw_ctr  = fp(conv2d(x2, w_ctr , b_ctr , padding=0))   ; [1,1,16,16]
//   raw_size = fp(conv2d(x2, w_size, b_size, padding=0))   ; [1,2,16,16]
//   raw_off  = fp(conv2d(x2, w_off , b_off , padding=0))   ; [1,2,16,16]
//   score_map_ctr   = sigmoid_clamped(raw_ctr)
//   score_map_size  = sigmoid_clamped(raw_size)
//   score_map_off   = raw_off                              ; offset has no sigmoid
//
// Input  x2     : [1, 48, 16, 16]   addr = ic*256 + oh*16 + ow   (row-major)
// Output stream : 5 channels (per tail top) -- see ports section
//
// §12.1 shared ROM (one instance each across all three sub-paths) :
//   rom_box_head_tail_ctr_offset_size_weight   (240 entries)
//     addr   0..47  : ctr_weight    (48)
//     addr  48..143 : offset_weight (96 = 2 * 48)
//     addr 144..239 : size_weight   (96 = 2 * 48)
//   rom_box_head_tail_ctr_offset_size_bias     (5 entries)
//     addr 0      : ctr_bias
//     addr 1..2   : offset_bias
//     addr 3..4   : size_bias
//
// ROM read contract (verilog_rule.mdc §7.1 ; CLK(~clk), 1-cycle latency) :
//   posedge T   : sub-module issues w_addr / b_addr (absolute, includes base)
//   posedge T+1 : w_i / b_i is the data for the previous tick's address ;
//                 sub-module's data-tick (wpre_phase==1) latches wgt_buf / bias_r.
//
// Sub-module style (kept identical to verilog_head2/conv.v) :
//   - K=1, PAD=0 specialization removes kh/kw inner loop
//   - 2-phase WPRE / MAC, negedge bias_r latch, acc_sat_r intermediate
//   - HAS_RELU = 0 (raw conv outputs ; ReLU never applied at tail per numpy)
//
// Tail top sequencer :
//   T_IDLE -> T_CTR (run u_ctr, route ROM addr/data, raw stream + sigmoid_lut)
//          -> T_OFF (run u_off, raw stream only ; offset has no sigmoid)
//          -> T_SIZE (run u_size, raw stream + sigmoid_lut)
//          -> T_DRAIN_SIG (wait one cycle so last sigmoid_lut output flushes)
//          -> T_DONE
//
// Goldens (verilog_head2/TEST_head.v compares against) :
//   Activation/box_head_tail_ctr_after_conv_out_bi.txt     (256)
//   Activation/box_head_tail_ctr_after_sigmoid_out_bi.txt  (256)
//   Activation/box_head_tail_offset_after_conv_out_bi.txt  (512)
//   Activation/box_head_tail_offset_final_out_bi.txt       (= raw_off, 512)
//   Activation/box_head_tail_size_after_conv_out_bi.txt    (512)
//   Activation/box_head_tail_size_after_sigmoid_out_bi.txt (512)
//
// Golden-Weight :
//   Weight/box_head_tail_ctr_weight_bi.txt     (48)
//   Weight/box_head_tail_offset_weight_bi.txt  (96)
//   Weight/box_head_tail_size_weight_bi.txt    (96)
//   Weight/box_head_tail_ctr_bias_bi.txt       (1)
//   Weight/box_head_tail_offset_bias_bi.txt    (2)
//   Weight/box_head_tail_size_bias_bi.txt      (2)
// =============================================================================


// =============================================================================
// tail_unit : single 1x1 conv pass (parametrized by ROM base + OUT_CH)
//   Used 3 times inside `tail` top : ctr / offset / size.
// =============================================================================
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
    y_ow
);

// ---------------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------------
parameter IN_CH       = 48 ;
parameter OUT_CH      = 1  ;
parameter IN_H        = 16 ;
parameter IN_W        = 16 ;
parameter WEIGHT_BASE = 0  ;
parameter BIAS_BASE   = 0  ;
parameter DATA_W      = 16 ;
parameter FRAC_W      = 8  ;
parameter ACC_W       = 32 ;

parameter X_AW        = 15 ;   // covers up to 48*16*16 = 12288
parameter W_AW        = 8  ;   // covers 240 entries (8-bit suffices)
parameter B_AW        = 4  ;   // covers 5 entries
parameter FEAT_AW     = 6  ;   // covers IN_CH=48
parameter OC_AW       = 2  ;   // covers OUT_CH<=2 (use 2 bits for safety)
parameter HW_AW       = 5  ;   // covers IN_H/IN_W<=16

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// FSM state encoding
// ---------------------------------------------------------------------------
parameter S_IDLE = 3'd0 ;
parameter S_WPRE = 3'd1 ;
parameter S_MAC  = 3'd2 ;
parameter S_SAT  = 3'd3 ;
parameter S_DONE = 3'd4 ;

// ---------------------------------------------------------------------------
// Reg / wire declarations
// ---------------------------------------------------------------------------
reg  [2:0]                  state, next_state ;
reg  [OC_AW-1:0]            oc_r ;
reg  [HW_AW-1:0]            oh_r, ow_r ;

reg                         ow_last ;
reg                         oh_last ;
reg                         oc_last ;

reg                         wpre_phase ;
reg  [FEAT_AW-1:0]          wpre_feat ;
reg                         wpre_done ;
reg                         wpre_bias_ce ;

reg                         mac_phase ;
reg  [FEAT_AW-1:0]          mac_feat ;
reg                         mac_done ;

reg  signed [DATA_W-1:0]    wgt_buf [0:IN_CH-1] ;
reg  signed [DATA_W-1:0]    bias_r ;
reg  signed [ACC_W-1:0]     acc_r ;
reg  signed [ACC_W-1:0]     acc_sat_r ;
reg                         x_in_pad ;

reg  [X_AW-1:0]             x_addr_r ;
reg  [W_AW-1:0]             w_addr_r ;
reg  [B_AW-1:0]             b_addr_r ;

reg                         y_valid_r ;
reg  signed [DATA_W-1:0]    y_data_r ;
reg  [OC_AW-1:0]            y_oc_r ;
reg  [HW_AW-1:0]            y_oh_r, y_ow_r ;

reg                         busy_r, done_r ;

reg  [FEAT_AW-1:0]          mac_ic ;
reg  [HW_AW-1:0]            ih_r ;
reg  [HW_AW-1:0]            iw_r ;
reg  [X_AW-1:0]             x_addr_nxt ;
reg                         mac_feat_last ;

reg  signed [DATA_W-1:0]    mac_x_op ;
reg  signed [DATA_W-1:0]    mac_w_op ;
reg  signed [2*DATA_W-1:0]  mac_prod ;
reg  signed [ACC_W-1:0]     acc_next ;

reg  signed [ACC_W-1:0]     acc_shifted ;
reg  signed [ACC_W-1:0]     y_pre_sat ;
reg  signed [DATA_W-1:0]    y_sat ;

// ---------------------------------------------------------------------------
// Output assigns
// ---------------------------------------------------------------------------
assign busy    = busy_r ;
assign done    = done_r ;
assign x_addr  = x_addr_r ;
assign w_addr  = w_addr_r ;
assign b_addr  = b_addr_r ;
assign y_valid = y_valid_r ;
assign y_data  = y_data_r ;
assign y_oc    = y_oc_r ;
assign y_oh    = y_oh_r ;
assign y_ow    = y_ow_r ;

// ---------------------------------------------------------------------------
// (comb) last-tick flags
// ---------------------------------------------------------------------------
always @(*) begin
    ow_last = (ow_r == IN_W - 1) ;
    oh_last = (oh_r == IN_H - 1) ;
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
// (seq) Weight / bias prefetch (2-phase per weight)
//   phase 0 : issue w_addr <= WEIGHT_BASE + oc*IN_CH + wpre_feat ; b_addr <= BIAS_BASE + oc
//   phase 1 : wgt_buf[wpre_feat] <= w_i ; wpre_feat++ unless last ; bias latched on negedge
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        wpre_phase   <= 1'b0 ;
        wpre_feat    <= 0 ;
        wpre_done    <= 1'b0 ;
        w_addr_r     <= 0 ;
        b_addr_r     <= 0 ;
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
                    // address tick (absolute ROM addresses include base offsets)
                    w_addr_r   <= WEIGHT_BASE + oc_r * IN_CH + wpre_feat ;
                    b_addr_r   <= BIAS_BASE + oc_r ;
                    wpre_phase <= 1'b1 ;
                end else begin
                    // data tick : latch weight ; bias via negedge sample below
                    wgt_buf[wpre_feat] <= w_i ;
                    if (wpre_feat == IN_CH - 1) begin
                        wpre_done    <= 1'b1 ;
                        wpre_bias_ce <= 1'b1 ;
                    end else begin
                        wpre_feat <= wpre_feat + 1 ;
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

// Bias ROM read aligns with CLK(~clk) : latch on negedge after last WPRE data tick
always @(negedge clk) begin
    if (!rst_n)
        bias_r <= 16'sd0 ;
    else if (wpre_bias_ce)
        bias_r <= b_i ;
end

// ---------------------------------------------------------------------------
// (comb) MAC tap decoder
//   K=1, PAD=0 : mac_ic = mac_feat ; ih=oh ; iw=ow ; never pad
// ---------------------------------------------------------------------------
always @(*) begin
    mac_ic        = mac_feat ;
    ih_r          = oh_r ;
    iw_r          = ow_r ;
    x_addr_nxt    = mac_ic * (IN_H*IN_W) + ih_r * IN_W + iw_r ;
    mac_feat_last = (mac_feat == IN_CH - 1) ;
end

// ---------------------------------------------------------------------------
// (seq) MAC datapath (2-phase per tap)
//   phase 0 : x_addr_r <= x_addr_nxt ; phase <- 1
//   phase 1 : acc_r <= acc_next ; mac_feat++ unless last
//   acc_r cleared at WPRE->MAC edge ; latched into acc_sat_r at last data tick.
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
                        x_in_pad  <= 1'b0 ;             // 1x1 PAD=0 -> never pad
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

// ---------------------------------------------------------------------------
// (comb) MAC arithmetic (always non-pad here)
// ---------------------------------------------------------------------------
always @(*) begin
    mac_x_op = $signed(x_i) ;
    mac_w_op = wgt_buf[mac_feat] ;
    mac_prod = mac_x_op * mac_w_op ;
    acc_next = acc_r + mac_prod ;
end

// ---------------------------------------------------------------------------
// (comb) Saturation / bias add (no ReLU at tail)
// ---------------------------------------------------------------------------
always @(*) begin
    acc_shifted = acc_sat_r >>> FRAC_W ;
    y_pre_sat   = acc_shifted + bias_r ;
    if (y_pre_sat >  32'sd32767)
        y_sat = 16'sh7fff ;
    else if (y_pre_sat < -32'sd32768)
        y_sat = 16'sh8000 ;
    else
        y_sat = y_pre_sat[DATA_W-1:0] ;
end

// ---------------------------------------------------------------------------
// (seq) y_valid pulse + y_data register
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
            y_data_r  <= y_sat ;
            y_oc_r    <= oc_r ;
            y_oh_r    <= oh_r ;
            y_ow_r    <= ow_r ;
        end else begin
            y_valid_r <= 1'b0 ;
        end
    end
end

endmodule


// =============================================================================
// tail : top-level wrapper -- 3 tail_unit instances share one weight ROM
//        and one bias ROM (§12.1 single macro per kind). Sequencer runs
//        ctr -> offset -> size. ctr / size streams are also routed through
//        sigmoid_lut for the *_after_sigmoid goldens.
// =============================================================================
module tail (
    clk           ,
    rst_n         ,
    start         ,
    busy          ,
    done          ,
    // input feature mem (conv2 output, [1,48,16,16])
    x_addr        ,
    x_i           ,
    // ctr raw stream (before sigmoid)  -- aligns box_head_tail_ctr_after_conv_out_bi.txt
    ctr_raw_y_valid ,
    ctr_raw_y_data  ,
    ctr_raw_y_oh    ,
    ctr_raw_y_ow    ,
    // ctr sigmoid stream               -- aligns box_head_tail_ctr_after_sigmoid_out_bi.txt
    ctr_y_valid     ,
    ctr_y_data      ,
    // offset stream (raw == final)     -- aligns box_head_tail_offset_after_conv_out_bi.txt
    off_y_valid     ,
    off_y_data      ,
    off_y_sub       ,            // 0=offset_x, 1=offset_y  (numpy oc index)
    off_y_oh        ,
    off_y_ow        ,
    // size raw stream                  -- aligns box_head_tail_size_after_conv_out_bi.txt
    size_raw_y_valid,
    size_raw_y_data ,
    size_raw_y_sub  ,
    size_raw_y_oh   ,
    size_raw_y_ow   ,
    // size sigmoid stream              -- aligns box_head_tail_size_after_sigmoid_out_bi.txt
    size_y_valid    ,
    size_y_data
);

// ---------------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------------
parameter DATA_W = 16 ;
parameter X_AW   = 15 ;     // 48*16*16=12288 fits in 14 bits ; use 15 for safety

// ROM bases per §12.1
parameter CTR_WEIGHT_BASE = 0   ;
parameter OFF_WEIGHT_BASE = 48  ;
parameter SIZ_WEIGHT_BASE = 144 ;
parameter CTR_BIAS_BASE   = 0   ;
parameter OFF_BIAS_BASE   = 1   ;
parameter SIZ_BIAS_BASE   = 3   ;

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------
input                       clk             ;
input                       rst_n           ;
input                       start           ;
output                      busy            ;
output                      done            ;
output [X_AW-1:0]           x_addr          ;
input  signed [DATA_W-1:0]  x_i             ;

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

// ---------------------------------------------------------------------------
// FSM state encoding (top sequencer)
// ---------------------------------------------------------------------------
parameter T_IDLE      = 3'd0 ;
parameter T_CTR       = 3'd1 ;
parameter T_OFF       = 3'd2 ;
parameter T_SIZE      = 3'd3 ;
parameter T_DRAIN_SIG = 3'd4 ;  // wait 1 cycle for sigmoid_lut last sample
parameter T_DONE      = 3'd5 ;

// ---------------------------------------------------------------------------
// Reg / wire declarations
// ---------------------------------------------------------------------------
reg  [2:0]                  tstate, tstate_n ;
reg                         busy_r, done_r ;
reg                         ctr_start_r ;
reg                         off_start_r ;
reg                         siz_start_r ;
reg                         drain_cnt ;     // 1-bit drain counter

// ROM share: addresses from active sub, then mux
wire [7:0]                  ctr_w_addr ;
wire [7:0]                  off_w_addr ;
wire [7:0]                  siz_w_addr ;
wire [3:0]                  ctr_b_addr ;
wire [3:0]                  off_b_addr ;
wire [3:0]                  siz_b_addr ;
wire signed [DATA_W-1:0]    w_i ;
wire signed [DATA_W-1:0]    b_i ;
// ROM macro addr width (memory2/rom_box_head_tail_*): weight A[7:0], bias A[6:0]
reg  [7:0]                  rom_w_a ;
reg  [6:0]                  rom_b_a ;
wire signed [DATA_W-1:0]    rom_w_q ;
wire signed [DATA_W-1:0]    rom_b_q ;
wire                        rom_ceb_w ;
wire                        rom_ceb_b ;

// x_addr mux
wire [X_AW-1:0]             ctr_x_addr ;
wire [X_AW-1:0]             off_x_addr ;
wire [X_AW-1:0]             siz_x_addr ;
reg  [X_AW-1:0]             x_addr_r ;

// sub-module status
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

// sigmoid_lut outputs
wire                        ctr_sig_v ;
wire signed [DATA_W-1:0]    ctr_sig_d ;
wire                        siz_sig_v ;
wire signed [DATA_W-1:0]    siz_sig_d ;

// ---------------------------------------------------------------------------
// Output assigns
// ---------------------------------------------------------------------------
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

assign size_y_valid     = siz_sig_v ;
assign size_y_data      = siz_sig_d ;

// ---------------------------------------------------------------------------
// (comb) ROM address mux : pick active sub-module's bus
//   Sub-modules drive their full absolute address (base already added).
//   Inactive subs are idle ; their addresses are don't-care.
// ---------------------------------------------------------------------------
always @(*) begin
    case (tstate)
        T_CTR : begin
            rom_w_a  = ctr_w_addr ;
            rom_b_a  = {3'b000, ctr_b_addr} ;
            x_addr_r = ctr_x_addr ;
        end
        T_OFF : begin
            rom_w_a  = off_w_addr ;
            rom_b_a  = {3'b000, off_b_addr} ;
            x_addr_r = off_x_addr ;
        end
        T_SIZE : begin
            rom_w_a  = siz_w_addr ;
            rom_b_a  = {3'b000, siz_b_addr} ;
            x_addr_r = siz_x_addr ;
        end
        default : begin
            rom_w_a  = 8'd0 ;
            rom_b_a  = 7'd0 ;
            x_addr_r = {X_AW{1'b0}} ;
        end
    endcase
end

// CEB low when any sub-module is busy. (Macro CEB active-low.)
assign rom_ceb_w = !(ctr_busy || off_busy || siz_busy) ;
assign rom_ceb_b = !(ctr_busy || off_busy || siz_busy) ;

assign w_i = rom_w_q ;
assign b_i = rom_b_q ;

// ---------------------------------------------------------------------------
// (seq) Top FSM state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) tstate <= T_IDLE ;
    else        tstate <= tstate_n ;
end

// ---------------------------------------------------------------------------
// (comb) Top FSM next-state
// ---------------------------------------------------------------------------
always @(*) begin
    tstate_n = tstate ;
    case (tstate)
        T_IDLE      : if (start)        tstate_n = T_CTR ;
        T_CTR       : if (ctr_done)     tstate_n = T_OFF ;
        T_OFF       : if (off_done)     tstate_n = T_SIZE ;
        T_SIZE      : if (siz_done)     tstate_n = T_DRAIN_SIG ;
        T_DRAIN_SIG : if (drain_cnt)    tstate_n = T_DONE ;
        T_DONE      :                   tstate_n = T_IDLE ;
        default     :                   tstate_n = T_IDLE ;
    endcase
end

// ---------------------------------------------------------------------------
// (seq) drain counter (1 cycle to flush sigmoid_lut after siz_done)
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) drain_cnt <= 1'b0 ;
    else if (tstate == T_SIZE && siz_done) drain_cnt <= 1'b0 ;
    else if (tstate == T_DRAIN_SIG)        drain_cnt <= 1'b1 ;
end

// ---------------------------------------------------------------------------
// (seq) busy / done / per-sub start pulses
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        busy_r      <= 1'b0 ;
        done_r      <= 1'b0 ;
        ctr_start_r <= 1'b0 ;
        off_start_r <= 1'b0 ;
        siz_start_r <= 1'b0 ;
    end else begin
        busy_r <= (tstate_n != T_IDLE) && (tstate_n != T_DONE) ;
        done_r <= (tstate_n == T_DONE) ;
        // 1-cycle start pulse on entering each phase
        ctr_start_r <= (tstate == T_IDLE) && start ;
        off_start_r <= (tstate == T_CTR)  && ctr_done ;
        siz_start_r <= (tstate == T_OFF)  && off_done ;
    end
end

// ---------------------------------------------------------------------------
// Sub-module instances (one weight ROM + one bias ROM shared, see mux above)
// ---------------------------------------------------------------------------
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
    .y_ow    (ctr_yw        )
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
    .y_ow    (off_yw        )
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
    .HW_AW        (5               )
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
    .y_ow    (siz_yw        )
);

// ---------------------------------------------------------------------------
// sigmoid_lut instances : ctr / size
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Shared §12.1 ROM macros (TSMC pin set, CLK(~clk), 1-cycle latency)
// ---------------------------------------------------------------------------
rom_box_head_tail_ctr_offset_size_weight u_rom_w (
    .A(rom_w_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(rom_ceb_w), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(rom_w_q));

rom_box_head_tail_ctr_offset_size_bias u_rom_b (
    .A(rom_b_a), .AM(), .CEBM(), .BIST(1'b0),
    .CEB(rom_ceb_b), .CLK(~clk),
    .SD(1'b0), .PUDELAY(),
    .RTSEL(2'b01), .PTSEL(2'b01), .TRB(2'b01), .TM(1'b0),
    .Q(rom_b_q));

endmodule
