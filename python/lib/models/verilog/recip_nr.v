// =============================================================================
// recip_nr.v
//
// Q8.8 reciprocal via Newton-Raphson（2 次迭代）。
//
// 輸入  x_i [15:0]  Q8.8 unsigned（CARE attention qk_mean_eps，≥ 1/256 = 1 int）
// 輸出  y_o [15:0]  Q8.8 signed，≈ 1/x_float
//
// 演算法（全 Q8.8 整數域）：
//   y = y × (2 − x × y)
//   2.0 in Q8.8 = 16'd512
//   x × y in Q8.8 = (x_int × y_int) >> 8
//
// 電路：
//   - recip_lut_seed 提供初始估值 y0
//   - FSM：IDLE → ITER1 → ITER2 → DONE
//   - 乘法使用 32-bit 中間暫存器
//
// 延遲：4 clock cycles（start → done）。
// =============================================================================

module recip_nr (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire [15:0] x_i,       // Q8.8 unsigned positive (≥ 1)
    output wire        busy,
    output reg         done,
    output reg  signed [15:0] y_o // Q8.8 result
);

// FSM state encoding
parameter S_IDLE  = 2'd0;
parameter S_ITER1 = 2'd1;
parameter S_ITER2 = 2'd2;
parameter S_DONE  = 2'd3;

// State register
reg [1:0] state;
reg [1:0] next_state;

// Datapath registers
reg [15:0] x_reg;
reg signed [15:0] y_reg;

// Seed module wires
wire [15:0] seed_y0;

// NR iteration combinational wires
// x × y in Q8.8: (x × y) >> 8 — keep Q16.16 intermediate, take [23:8]
wire [31:0] xy_raw   = x_reg * $signed(y_reg);
wire signed [15:0] xy = xy_raw[23:8];

// 2 - x×y in Q8.8 (2.0 = 512)
wire signed [15:0] coeff = 16'sd512 - xy;

// y_new = y × (2 - x×y) in Q8.8
wire [31:0] y_new_raw  = $signed(y_reg) * coeff;
wire signed [15:0] y_new = y_new_raw[23:8];

// Seed instantiation
recip_lut_seed u_seed (
    .x_i  (x_i),
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
        S_ITER2: next_state = S_DONE;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic and datapath registers
always @(posedge clk) begin
    done <= 1'b0;
    if (reset) begin
        x_reg <= 16'd0;
        y_reg <= 16'sd0;
        y_o   <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                if (start) begin
                    x_reg <= x_i;
                    y_reg <= $signed(seed_y0);
                end
            end
            S_ITER1: begin
                y_reg <= y_new;
            end
            S_ITER2: begin
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
