// =============================================================================
// conv2d_q88.v
//
// Q8.8 定點 2D convolution 的單一輸出 pixel（Serial MAC）。
//
// 每次執行計算一個輸出 pixel，accumulate CIN × KH × KW 個乘積：
//   y = clamp( (Σ x_patch × w_kernel) >> 8 + bias, Q8.8 range )
//
// 介面：
//   外部 controller 每 cycle 送一對 (a_i, w_i)（feature map patch + weight）。
//   共需 CIN × KH × KW 個 cycle 完成一個 output pixel。
//   start 拉高後第一對即有效。
//   done 拉高時 y_o 有效；bias 在 done 那一 cycle 由外部提供。
//
// Parameters：
//   CIN    - 輸入通道數（預設 768）
//   KH,KW  - kernel 大小（3×3 或 1×1，預設 3）
//   RELU   - 1 = 在輸出後套用 ReLU（max(y, 0)）
//
// 累加器精度：
//   同 linear_q88.v：48-bit 有號累加器，最後 >>8 + bias + 飽和。
//
// =============================================================================

module conv2d_q88 #(
    parameter CIN  = 768,  // input channels
    parameter KH   = 3,    // kernel height (3 or 1)
    parameter KW   = 3,    // kernel width  (3 or 1)
    parameter RELU = 0     // 1 = apply ReLU to output
) (
    input  wire        clk,
    input  wire        reset,
    // Control
    input  wire        start,    // begin new pixel; first pair valid same cycle
    input  wire        a_valid,  // (a_i, w_i) valid this cycle
    // Data
    input  wire signed [15:0] a_i,  // Q8.8 feature map value
    input  wire signed [15:0] w_i,  // Q8.8 kernel weight
    input  wire signed [15:0] b_i,  // Q8.8 bias (sampled when done)
    // Status
    output wire        busy,
    output reg         done,
    // Result
    output reg  signed [15:0] y_o   // Q8.8 output pixel
);

// Total accumulations per output pixel
localparam TOTAL = CIN * KH * KW;
localparam CNT_W = 21;  // ceil(log2(768*9)) = ceil(log2(6912)) = 13; use 21 for safety

// FSM state encoding
parameter S_IDLE = 2'd0;
parameter S_ACC  = 2'd1;
parameter S_BIAS = 2'd2;
parameter S_DONE = 2'd3;

// State register
reg [1:0] state;
reg [1:0] next_state;

// Datapath registers
reg signed [47:0] acc;
reg [CNT_W-1:0]   cnt;

// Multiply current input pair
wire signed [31:0] prod = $signed(a_i) * $signed(w_i);  // Q16.16

// Shift + bias + saturate + optional ReLU
wire signed [47:0] shifted = $signed(acc) >>> 8;
wire signed [47:0] biased  = shifted + {{32{b_i[15]}}, b_i};
wire signed [15:0] y_sat   = (biased > 48'sh7FFF) ? 16'sh7FFF :
                              (biased < -48'sh8000) ? -16'sh8000 :
                              biased[15:0];
wire signed [15:0] y_relu  = (y_sat[15]) ? 16'sd0 : y_sat;  // max(y, 0)
wire signed [15:0] y_final = (RELU != 0) ? y_relu : y_sat;

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
        S_IDLE: next_state = start              ? S_ACC  : S_IDLE;
        S_ACC:  next_state = (cnt == TOTAL - 1) ? S_BIAS : S_ACC;
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
                    acc <= {{16{prod[31]}}, prod};
            end
            S_ACC: begin
                if (a_valid) begin
                    acc <= acc + {{16{prod[31]}}, prod};
                    cnt <= cnt + 1'b1;
                end
            end
            S_BIAS: ;  // combinational path computes y_final
            S_DONE: begin
                y_o  <= y_final;
                done <= 1'b1;
            end
            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
