// =============================================================================
// mlp_block.v
//
// Transformer MLP Block：fc1（ReLU）→ fc2，對齊 run_backbone_numpy.block_forward MLP。
//
// 架構：
//   h = fp(relu(linear(x, fc1_w, fc1_b)))   [EMBED_DIM → MLP_DIM]
//   out = fp(linear(h, fc2_w, fc2_b))        [MLP_DIM → EMBED_DIM]
//
// 每個 linear 使用 linear_q88.v；每次計算一個輸出 neuron，需 CIN 個 cycle。
// 外部 controller 負責：
//   - 依序為每個 output neuron 觸發 start + 提供 (a_i, w_i) 串流
//   - 收集 y_o 後寫回 h 或最終輸出 buffer
//
// 此模組只包裝一個 linear_q88 instance（fc1/fc2 共用），由外部 controller
// 切換 weight 來源：fc1 weights → relu 結果寫 h buffer → fc2 weights。
//
// 因此 mlp_block.v 是一個「帶 relu 選項」的 linear_q88 包裝。
// 完整 MLP 的 token × output neuron 迭代由 transformer_block.v controller 驅動。
// =============================================================================

module mlp_block #(
    parameter CIN  = 768,     // input channels (EMBED_DIM for fc1; MLP_DIM for fc2)
    parameter RELU = 0        // 1 = apply ReLU to output (use 1 for fc1, 0 for fc2)
) (
    input  wire        clk,
    input  wire        reset,
    // Control (from transformer_block controller)
    input  wire        start,    // begin new neuron computation
    input  wire        a_valid,
    // Data
    input  wire signed [15:0] a_i,  // Q8.8 activation
    input  wire signed [15:0] w_i,  // Q8.8 weight
    input  wire signed [15:0] b_i,  // Q8.8 bias
    // Status
    output wire        busy,
    output wire        done,
    // Result
    output wire signed [15:0] y_o   // Q8.8 output (post-relu if RELU=1)
);

// Linear core
wire signed [15:0] lin_y;
wire lin_done, lin_busy;

linear_q88 #(.CIN(CIN)) u_lin (
    .clk    (clk),
    .reset  (reset),
    .start  (start),
    .a_valid(a_valid),
    .a_i    (a_i),
    .w_i    (w_i),
    .b_i    (b_i),
    .busy   (lin_busy),
    .done   (lin_done),
    .y_o    (lin_y)
);

// Optional ReLU (combinational on lin_y)
wire signed [15:0] relu_y = lin_y[15] ? 16'sd0 : lin_y;

assign y_o = (RELU != 0) ? relu_y : lin_y;
assign done = lin_done;
assign busy = lin_busy;

endmodule
