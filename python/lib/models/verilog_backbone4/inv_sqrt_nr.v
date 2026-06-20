// =============================================================================
// inv_sqrt_nr.v   (verilog_backbone4, Q7.7-native)
//
// Q7.7 inverse square root via Newton-Raphson（3 次迭代，round-to-nearest）。
//
// 輸入  v_i [15:0]  Q7.7 signed（LayerNorm variance + eps，> 0）
// 輸出  y_o [15:0]  Q7.7 signed，≈ 1/sqrt(v_float)
//
// 演算法（全 Q7.7 整數域，每次截位都加 0.5 LSB rounding）：
//   y = y × (1.5 − 0.5 × v × y²)
//   1.5 in Q7.7 = 16'd192
//   y² in Q7.7        : (y×y + 64) >> 7
//   0.5 × v × y² Q7.7 : (v×y² + 128) >> 8
//   y_new Q7.7        : (y×coeff + 64) >> 7
//
// 與 Q8.8 版差異：scale 256->128 -> 1.5 常數 384->192、各 round 常數與 shift
//   皆少一位（128->64, 256->128, >>8->>>7, >>9->>>8），bit-slice 對應下移一位。
//   省去 layer_norm 端 var<<1 / inv_std>>1 的 Q8.8 轉換。
//
// 電路：
//   - inv_sqrt_lut_seed 提供初始估值 y0（Q7.7）
//   - FSM：IDLE → ITER1 → ITER2 → ITER3 → DONE
//   - 16-entry LUT + 3 NR（seed 每 bin ~25% 誤差，3 NR 收斂）
//   - 乘法使用 32-bit 中間暫存器（v 與 y 皆為 Q7.7，積為 Q14.14）
//
// 延遲：5 clock cycles（start → done）。
// =============================================================================

module inv_sqrt_nr (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire signed [15:0] v_i,     // Q7.7 variance (positive)
    output wire        busy,
    output reg         done,
    output reg  signed [15:0] y_o      // Q7.7 result
);

// FSM state encoding (3-bit: 5 states)
parameter S_IDLE  = 3'd0;
parameter S_ITER1 = 3'd1;
parameter S_ITER2 = 3'd2;
parameter S_ITER3 = 3'd3;
parameter S_DONE  = 3'd4;

// State register
reg [2:0] state;
reg [2:0] next_state;

// Datapath registers
reg signed [15:0] v_reg;
reg signed [15:0] y_reg;

// Seed module wires
wire [15:0] seed_y0;

// NR iteration combinational wires (round-to-nearest at each shift).
// y² in Q7.7: round((y*y) / 128) = (y*y + 64) >> 7
wire signed [31:0] y_sq_raw   = $signed(y_reg) * $signed(y_reg);
wire signed [31:0] y_sq_rnd   = y_sq_raw + 32'sd64;
wire signed [15:0] y_sq       = y_sq_rnd[22:7];

// 0.5 × v × y² in Q7.7: round((v*y_sq) / 256) = (v*y_sq + 128) >> 8
wire signed [31:0] term_raw   = $signed(v_reg) * $signed(y_sq);
wire signed [31:0] term_rnd   = term_raw + 32'sd128;
wire signed [15:0] term       = term_rnd[23:8];

// 1.5 - 0.5×v×y² in Q7.7 (1.5 = 192)
wire signed [15:0] coeff = 16'sd192 - term;

// y_new = round(y * coeff / 128) = (y*coeff + 64) >> 7
wire signed [31:0] y_new_raw  = $signed(y_reg) * coeff;
wire signed [31:0] y_new_rnd  = y_new_raw + 32'sd64;
wire signed [15:0] y_new      = y_new_rnd[22:7];

// Seed instantiation (v_i treated as unsigned for LUT address)
inv_sqrt_lut_seed u_seed (
    .v_i  (v_i),
    .y0_o (seed_y0)
);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:  next_state = start   ? S_ITER1 : S_IDLE;
        S_ITER1: next_state = S_ITER2;
        S_ITER2: next_state = S_ITER3;
        S_ITER3: next_state = S_DONE;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3: datapath registers (v_reg / y_reg / y_o).
// Single if(reset); case branches use ternary (no nested if) per style rule.
always @(posedge clk) begin
    if (reset) begin
        v_reg <= 16'sd0;
        y_reg <= 16'sd0;
        y_o   <= 16'sd0;
    end else begin
        case (state)
            // load v and LUT seed only when start (ternary = hold otherwise)
            S_IDLE: begin
                v_reg <= start ? v_i : v_reg;
                y_reg <= start ? $signed(seed_y0) : y_reg;
            end
            // NR iters: y = y × (1.5 - 0.5×v×y²); 3 iters to drop LUT-seed error
            S_ITER1: y_reg <= y_new;
            S_ITER2: y_reg <= y_new;
            S_ITER3: y_reg <= y_new;
            S_DONE:  y_o   <= y_reg;
            default: ;
        endcase
    end
end

// done pulse: sole driver of done (single if + else if + else)
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE)
        done <= 1'b1;
    else
        done <= 1'b0;
end

assign busy = (state != S_IDLE);

endmodule
