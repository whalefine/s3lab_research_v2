// =============================================================================
// inv_sqrt_nr.v
//
// Q8.8 inverse square root via Newton-Raphson（3 次迭代，round-to-nearest）。
//
// 輸入  v_i [15:0]  Q8.8 signed（LayerNorm variance + eps，> 0）
// 輸出  y_o [15:0]  Q8.8 signed，≈ 1/sqrt(v_float)
//
// 演算法（全 Q8.8 整數域，每次截位都加 0.5 LSB rounding）：
//   y = y × (1.5 − 0.5 × v × y²)
//   1.5 in Q8.8 = 16'd384
//   0.5 × v × y² = (v × (y × y + 128 >> 8) + 256) >> 9
//
// 電路：
//   - inv_sqrt_lut_seed 提供初始估值 y0
//   - FSM：IDLE → ITER1 → ITER2 → ITER3 → DONE
//   - 16-entry LUT + 3 NR：max err ≈ 40 LSB（vs 2 NR 的 ~306 LSB）
//   - 乘法使用 32-bit 中間暫存器（v 與 y 皆為 Q8.8，積為 Q16.16）
//
// 延遲：5 clock cycles（start → done）。
// =============================================================================

module inv_sqrt_nr (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire signed [15:0] v_i,     // Q8.8 variance (positive)
    output wire        busy,
    output reg         done,
    output reg  signed [15:0] y_o      // Q8.8 result
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
// y² in Q8.8: round((y*y) / 256) = (y*y + 128) >> 8
wire signed [31:0] y_sq_raw   = $signed(y_reg) * $signed(y_reg);
wire signed [31:0] y_sq_rnd   = y_sq_raw + 32'sd128;
wire signed [15:0] y_sq       = y_sq_rnd[23:8];

// 0.5 × v × y² in Q8.8: round((v*y_sq) / 512) = (v*y_sq + 256) >> 9
wire signed [31:0] term_raw   = $signed(v_reg) * $signed(y_sq);
wire signed [31:0] term_rnd   = term_raw + 32'sd256;
wire signed [15:0] term       = term_rnd[24:9];

// 1.5 - 0.5×v×y² in Q8.8 (1.5 = 384)
wire signed [15:0] coeff = 16'sd384 - term;

// y_new = round(y * coeff / 256) = (y*coeff + 128) >> 8
wire signed [31:0] y_new_raw  = $signed(y_reg) * coeff;
wire signed [31:0] y_new_rnd  = y_new_raw + 32'sd128;
wire signed [15:0] y_new      = y_new_rnd[23:8];  // matches Python _inv_sqrt_nr_q88_fixed

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

// FSM segment 3: output logic and datapath registers
always @(posedge clk) begin
    done <= 1'b0;
    if (reset) begin
        v_reg <= 16'sd0;
        y_reg <= 16'sd0;
        y_o   <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                if (start) begin
                    v_reg <= v_i;
                    y_reg <= $signed(seed_y0);
                end
            end
            S_ITER1: begin
                // 1st NR: y = y × (1.5 - 0.5×v×y²)
                y_reg <= y_new;
            end
            S_ITER2: begin
                // 2nd NR
                y_reg <= y_new;
            end
            S_ITER3: begin
                // 3rd NR — required because LUT seed has ~25% error per bin;
                // 2 NR leaves up to ~306 LSB error, 3 NR drops to ~40 LSB.
                y_reg <= y_new;
            end
            S_DONE: begin
                y_o  <= y_reg;
                done <= 1'b1;
            end
            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
