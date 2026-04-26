// =============================================================================
// linear_q88.v
//
// Q8.8 定點矩陣乘法的單一輸出神經元（Serial MAC）。
//
// 每次執行計算一個輸出 neuron：
//   y = clamp( (Σ a[i] × w[i]) >> 8  +  bias, Q8.8 range )
//
// 介面：
//   外部 controller 每 cycle 送一對 (a_i, w_i)，持續 CIN 個 cycle。
//   start 拉高後第一對 (a_i, w_i) 即有效（start 期間 a_valid 也需為 1）。
//   done 拉高時 y_o 輸出有效；bias 在 done 那一 cycle 由外部提供。
//
// 累加器精度：
//   每個乘積 a_int × w_int 為 16×16 → 32 bit Q16.16。
//   累加 CIN 項後最多需要 ceil(log2(CIN)) + 30 bit ≈ 40 bit（CIN ≤ 4096）。
//   使用 48-bit 有號累加器確保不溢出。
//
// 輸出飽和：
//   (acc >> 8) + bias 超出 Q8.8 signed 範圍時飽和到 ±32767。
//
// =============================================================================

module linear_q88 #(
    parameter CIN = 768    // input dimension; default backbone hidden dim
) (
    input  wire        clk,
    input  wire        reset,
    // Control
    input  wire        start,    // begin new neuron; first (a_i,w_i) valid same cycle
    input  wire        a_valid,  // (a_i, w_i) pair valid this cycle
    // Data
    input  wire signed [15:0] a_i,  // Q8.8 activation
    input  wire signed [15:0] w_i,  // Q8.8 weight
    input  wire signed [15:0] b_i,  // Q8.8 bias (sampled when done)
    // Status
    output wire        busy,
    output reg         done,
    // Result
    output reg  signed [15:0] y_o   // Q8.8 output
);

// CIN counter width
localparam CNT_W = 12;  // supports CIN up to 4096

// FSM state encoding
parameter S_IDLE = 2'd0;
parameter S_ACC  = 2'd1;
parameter S_BIAS = 2'd2;
parameter S_DONE = 2'd3;

// State register
reg [1:0] state;
reg [1:0] next_state;

// Datapath registers
reg signed [47:0] acc;              // 48-bit signed accumulator (Q16.16 sum)
reg [CNT_W-1:0]   cnt;              // input pair counter

// Multiply current input pair
wire signed [31:0] prod = $signed(a_i) * $signed(w_i);  // Q16.16

// Shift accumulator >> 8 to get Q8.8 integer; then add bias and saturate
wire signed [47:0] shifted = $signed(acc) >>> 8;
wire signed [47:0] biased  = shifted + {{32{b_i[15]}}, b_i};
wire signed [15:0] y_sat   = (biased > 48'sh7FFF) ? 16'sh7FFF :
                              (biased < -48'sh8000) ? -16'sh8000 :
                              biased[15:0];

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
        S_IDLE: next_state = start            ? S_ACC  : S_IDLE;
        S_ACC:  next_state = (cnt == CIN - 1) ? S_BIAS : S_ACC;
        S_BIAS: next_state = S_DONE;
        S_DONE: next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic and datapath
always @(posedge clk) begin
    done <= 1'b0;
    if (reset) begin
        acc  <= 48'sd0;
        cnt  <= {CNT_W{1'b0}};
        y_o  <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                acc <= 48'sd0;
                cnt <= {CNT_W{1'b0}};
                if (start && a_valid)
                    acc <= {{16{prod[31]}}, prod};  // sign-extend first product
            end
            S_ACC: begin
                if (a_valid) begin
                    acc <= acc + {{16{prod[31]}}, prod};
                    cnt <= cnt + 1'b1;
                end
            end
            S_BIAS: begin
                // shift + bias + saturate already combinationally computed
            end
            S_DONE: begin
                y_o  <= y_sat;
                done <= 1'b1;
            end
            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
