// =============================================================================
// layer_norm.v
// -----------------------------------------------------------------------------
// Q8.8 Layer Normalization，對齊 run_backbone_numpy.layer_norm()。
//
// 每次處理 ONE TOKEN（FEAT_DIM 維）：
//   mean  = sum(x) × rcp_c     (rcp_c = 1/FEAT_DIM，以 RCP_NUM/2^RCP_SHIFT 近似)
//   var   = sum(centered²) × rcp_c
//   y[i]  = w[i] × (centered[i] × inv_std) + b[i]
//
// 介面說明：
//   - start 後外部 controller 連續送 FEAT_DIM cycle 的 x_i（x_valid=1）
//   - feat_addr_o 指出目前需要的 weight/bias 索引（0..FEAT_DIM-1）
//   - y_valid=1 的 FEAT_DIM cycle 輸出 y_o（對齊 feat_addr_o）
//
// inv_sqrt_nr 使用 4-cycle 子模組（inv_sqrt_nr.v）。
// Parent capture: out_beat_o posedge latch y_sat_o（y_o reg 晚一拍）。
// Saturation / rounding via wire only (no function).
// =============================================================================

module layer_norm #(
    parameter FEAT_DIM  = 768,
    parameter RCP_SHIFT = 16,
    parameter RCP_NUM   = 85
) (
    clk,
    reset,
    start,
    x_i,
    x_valid,
    w_i,
    b_i,
    feat_addr_o,
    busy,
    done,
    y_o,
    y_valid,
    y_sat_o,
    out_beat_o
);

input                       clk ;
input                       reset ;
input                       start ;
input  signed [15:0]        x_i ;
input                       x_valid ;
input  signed [15:0]        w_i ;
input  signed [15:0]        b_i ;
output [9:0]                feat_addr_o ;
output                      busy ;
output reg                  done ;
output reg signed [15:0]   y_o ;
output reg                  y_valid ;
output wire signed [15:0]  y_sat_o ;
output wire                 out_beat_o ;

parameter S_IDLE   = 3'd0 ;
parameter S_LOAD   = 3'd1 ;
parameter S_MEAN   = 3'd2 ;
parameter S_CENTER = 3'd3 ;
parameter S_VAR    = 3'd4 ;
parameter S_INV    = 3'd5 ;
parameter S_NORM   = 3'd6 ;
parameter S_DONE   = 3'd7 ;

reg [2:0]                   state, next_state ;
reg [9:0]                   addr ;

reg signed [15:0]           feat_buf [0:FEAT_DIM-1] ;

reg signed [31:0]           sum_acc ;
reg [47:0]                  sum_sq_acc ;
reg signed [15:0]           mean_q88 ;
reg signed [15:0]           var_q88 ;

reg                         inv_start ;
wire                        inv_busy, inv_done ;
wire signed [15:0]          inv_std ;

wire signed [16:0]          centered17 ;
wire signed [15:0]          centered ;
wire [63:0]                 var_full ;
wire [63:0]                 var_rnd ;
wire signed [15:0]          var_q88_comb ;
wire signed [15:0]          var_eps ;
wire [31:0]                 ci_std_raw ;
wire signed [15:0]          ci_std ;
wire [31:0]                 wci_raw ;
wire signed [15:0]          wci ;
wire signed [31:0]          y_full ;
wire signed [15:0]          y_sat ;
wire [31:0]                 csq ;

wire signed [31:0]          mean_prod ;
wire signed [31:0]          mean_rnd_t ;
wire signed [31:0]          mean_shr ;
wire signed [15:0]          mean_q88_comb ;

wire signed [31:0]          ci_std_t ;
wire signed [31:0]          wci_t ;

assign centered17 = $signed({feat_buf[addr][15], feat_buf[addr]}) -
                      $signed({mean_q88[15], mean_q88}) ;
assign centered = (centered17[16] ^ centered17[15]) ?
    (centered17[16] ? 16'sh8000 : 16'sh7FFF) : centered17[15:0] ;

assign var_full     = sum_sq_acc * RCP_NUM ;
assign var_rnd      = var_full + 64'd8388608 ;
assign var_q88_comb = (|var_rnd[63:39]) ? 16'sh7FFF : var_rnd[39:24] ;
assign var_eps      = (var_q88_comb <= 16'sd0) ? 16'sd1 : var_q88_comb ;

assign mean_prod    = sum_acc * RCP_NUM ;
assign mean_rnd_t   = mean_prod + 32'sd32768 ;
assign mean_shr     = mean_rnd_t >>> 16 ;
assign mean_q88_comb =
    (mean_shr > 32'sh7FFF) ? 16'sh7FFF :
    (mean_shr < -32'sh8000) ? -16'sh8000 : mean_shr[15:0] ;

assign ci_std_raw = $signed(feat_buf[addr]) * $signed(inv_std) ;
assign ci_std_t   = ci_std_raw + 32'sd128 ;
assign ci_std     =
    (ci_std_t > 32'sh7FFF_FFFF) ? 16'sh7FFF :
    (ci_std_t < -32'sh8000_0000) ? -16'sh8000 : ci_std_t[23:8] ;
assign wci_raw    = $signed(w_i) * ci_std ;
assign wci_t      = wci_raw + 32'sd128 ;
assign wci        =
    (wci_t > 32'sh7FFF_FFFF) ? 16'sh7FFF :
    (wci_t < -32'sh8000_0000) ? -16'sh8000 : wci_t[23:8] ;
assign y_full     = $signed(wci) + $signed(b_i) ;
assign y_sat      =
    (y_full > 32'sh7FFF) ? 16'sh7FFF :
    (y_full < -32'sh8000) ? -16'sh8000 : y_full[15:0] ;

assign y_sat_o    = y_sat ;
assign out_beat_o = (state == S_NORM) ;
assign csq        = $signed(centered) * $signed(centered) ;
assign feat_addr_o = addr ;
assign busy       = (state != S_IDLE) ;

inv_sqrt_nr u_inv_sqrt (
    .clk   (clk),
    .reset (reset),
    .start (inv_start),
    .v_i   (var_eps),
    .busy  (inv_busy),
    .done  (inv_done),
    .y_o   (inv_std)
);

// FSM state
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE ;
    else
        state <= next_state ;
end

// FSM next_state
always @(*) begin
    next_state = state ;
    case (state)
        S_IDLE:   next_state = start ? S_LOAD : S_IDLE ;
        // Last sample: addr==FEAT_DIM-1 & x_valid, or catch-up addr==FEAT_DIM (must not stay at FEAT_DIM for combo read)
        S_LOAD:   next_state = ((addr == FEAT_DIM-1 && x_valid) || (addr == FEAT_DIM))
                  ? S_MEAN : S_LOAD ;
        S_MEAN:   next_state = S_CENTER ;
        S_CENTER: next_state = (addr == FEAT_DIM-1) ? S_VAR : S_CENTER ;
        S_VAR:    next_state = S_INV ;
        S_INV:    next_state = inv_done ? S_NORM : S_INV ;
        S_NORM:   next_state = (addr == FEAT_DIM-1) ? S_DONE : S_NORM ;
        S_DONE:   next_state = S_IDLE ;
        default:  next_state = S_IDLE ;
    endcase
end

// done, y_valid, datapath
always @(posedge clk) begin
    done      <= 1'b0 ;
    y_valid   <= 1'b0 ;
    inv_start <= 1'b0 ;
    if (reset) begin
        addr       <= 10'd0 ;
        sum_acc    <= 32'sd0 ;
        sum_sq_acc <= 48'd0 ;
        mean_q88   <= 16'sd0 ;
        var_q88    <= 16'sd0 ;
        y_o        <= 16'sd0 ;
    end else begin
        case (state)
            S_IDLE: begin
                addr       <= 10'd0 ;
                sum_acc    <= 32'sd0 ;
                sum_sq_acc <= 48'd0 ;
            end

            S_LOAD: begin
                if (x_valid && (addr < FEAT_DIM)) begin
                    feat_buf[addr] <= x_i ;
                    sum_acc        <= sum_acc + $signed(x_i) ;
                    addr           <= (addr == FEAT_DIM - 1) ? 10'd0 : (addr + 10'd1) ;
                end
            end

            S_MEAN: begin
                mean_q88   <= mean_q88_comb ;
                addr       <= 10'd0 ;
                sum_sq_acc <= 48'd0 ;
            end

            S_CENTER: begin
                feat_buf[addr] <= centered ;
                sum_sq_acc     <= sum_sq_acc + {16'd0, csq} ;
                addr           <= (addr == FEAT_DIM - 1) ? 10'd0 : (addr + 10'd1) ;
            end

            S_VAR: begin
                var_q88   <= var_eps ;
                inv_start <= 1'b1 ;
                addr      <= 10'd0 ;
            end

            S_INV: begin
            end

            S_NORM: begin
                y_o     <= y_sat ;
                y_valid <= 1'b1 ;
                addr    <= (addr == FEAT_DIM - 1) ? 10'd0 : (addr + 10'd1) ;
            end

            S_DONE: begin
                done <= 1'b1 ;
                addr <= 10'd0 ;
            end

            default: ;
        endcase
    end
end

endmodule
