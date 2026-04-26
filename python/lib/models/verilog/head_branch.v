// =============================================================================
// head_branch.v
//
// CenterPredictor 單一 branch（ctr / size / offset），對齊 head_branch() in numpy。
//
// 架構（以 ctr 為例）：
//   conv1_out = relu(conv2d(x,        w1, b1, pad=1))  [1, 256, 16, 16]
//   conv2_out = relu(conv2d(conv1_out, w2, b2, pad=1))  [1, 128, 16, 16]
//   conv3_out = relu(conv2d(conv2_out, w3, b3, pad=1))  [1,  64, 16, 16]
//   conv4_out = relu(conv2d(conv3_out, w4, b4, pad=1))  [1,  32, 16, 16]
//   conv5_out = conv2d(conv4_out, w5, b5, pad=0)        [1, OCH, 16, 16]
//   if (DO_SIGMOID): out = sigmoid_clamped(conv5_out)   Q8.8, [1/256, 255/256]
//   else:            out = conv5_out                    (offset branch)
//
// Channel 序列由 channel 參數控制：768→256→128→64→32→OCH。
//
// conv1~4 各使用 conv2d_q88（3×3, pad=1, RELU=1）。
// conv5 使用 conv2d_q88（1×1, pad=0, RELU=0）。
// sigmoid 使用 sigmoid_lut（Q8.8 → Q0.8），之後做 fp() 回 Q8.8。
//
// 外部 controller 依序為每個 output pixel 提供 (a_i, w_i) 串流。
// 每個 conv 層處理完後，結果存入 feature_buf（16×16×channel）。
//
// =============================================================================

module head_branch #(
    parameter IN_CH    = 768,    // input channels (backbone output)
    parameter C1       = 256,    // conv1 output channels
    parameter C2       = 128,
    parameter C3       = 64,
    parameter C4       = 32,
    parameter OUT_CH   = 1,      // conv5 output channels (1 or 2)
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter DO_SIGMOID = 1     // 1 for ctr/size, 0 for offset
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // Serial (a_i, w_i) input from external controller
    input  wire signed [15:0] a_i,
    input  wire signed [15:0] w_i,
    input  wire signed [15:0] b_i,
    input  wire        a_valid,

    // Status
    output wire        busy,
    output reg         done,

    // Output feature map stream (OUT_CH × FEAT_H × FEAT_W values)
    output reg  signed [15:0] y_o,
    output reg         y_valid
);

localparam FEAT_SZ = FEAT_H * FEAT_W;  // 256

// Feature buffers (4 intermediate layers + output)
// Maximum size: max(C1..C4) × FEAT_SZ = 256 × 256 = 65536 entries
reg signed [15:0] feat_buf [0:C1*FEAT_SZ-1];  // reused for all layers
reg signed [15:0] out_buf  [0:OUT_CH*FEAT_SZ-1];

// FSM state encoding
parameter S_IDLE  = 3'd0;
parameter S_CONV1 = 3'd1;
parameter S_CONV2 = 3'd2;
parameter S_CONV3 = 3'd3;
parameter S_CONV4 = 3'd4;
parameter S_CONV5 = 3'd5;
parameter S_OUT   = 3'd6;
parameter S_DONE  = 3'd7;

reg [2:0] state, next_state;
reg [16:0] pix_cnt;   // output pixel counter within current conv layer

// conv2d_q88 instance (shared; multiplexed by state)
wire conv_busy, conv_done;
wire signed [15:0] conv_y;
reg  conv_start, conv_valid;
reg  signed [15:0] a_mux, w_mux, b_mux;

conv2d_q88 #(
    .CIN  (IN_CH),   // overridden via parameters at higher level; simplified here
    .KH   (3),
    .KW   (3),
    .RELU (1)
) u_conv (
    .clk    (clk),
    .reset  (reset),
    .start  (conv_start),
    .a_valid(conv_valid),
    .a_i    (a_mux),
    .w_i    (w_mux),
    .b_i    (b_mux),
    .busy   (conv_busy),
    .done   (conv_done),
    .y_o    (conv_y)
);

// sigmoid_lut (combinational)
wire [7:0] sig_out;
sigmoid_lut u_sig (
    .x_i(conv_y),
    .y_o(sig_out)
);

// Clamped sigmoid output in Q8.8
wire signed [15:0] sig_q88 = {8'd0, sig_out};  // Q0.8 → Q8.8 (upper byte = 0)

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:  next_state = start      ? S_CONV1 : S_IDLE;
        S_CONV1: next_state = (conv_done && pix_cnt == C1*FEAT_SZ-1) ? S_CONV2 : S_CONV1;
        S_CONV2: next_state = (conv_done && pix_cnt == C2*FEAT_SZ-1) ? S_CONV3 : S_CONV2;
        S_CONV3: next_state = (conv_done && pix_cnt == C3*FEAT_SZ-1) ? S_CONV4 : S_CONV3;
        S_CONV4: next_state = (conv_done && pix_cnt == C4*FEAT_SZ-1) ? S_CONV5 : S_CONV4;
        S_CONV5: next_state = (conv_done && pix_cnt == OUT_CH*FEAT_SZ-1) ? S_OUT : S_CONV5;
        S_OUT:   next_state = (pix_cnt == OUT_CH*FEAT_SZ-1) ? S_DONE : S_OUT;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done       <= 1'b0;
    y_valid    <= 1'b0;
    conv_start <= 1'b0;
    conv_valid <= a_valid;
    a_mux      <= a_i;
    w_mux      <= w_i;
    b_mux      <= b_i;

    if (reset) begin
        pix_cnt <= 17'd0;
    end else begin
        case (state)
            S_IDLE: begin
                pix_cnt <= 17'd0;
            end

            S_CONV1, S_CONV2, S_CONV3, S_CONV4: begin
                // Launch conv_q88 for each output pixel; store result in feat_buf
                if (!conv_busy) begin
                    conv_start <= 1'b1;
                end
                if (conv_done) begin
                    feat_buf[pix_cnt] <= conv_y;
                    pix_cnt <= pix_cnt + 1'b1;
                end
            end

            S_CONV5: begin
                // conv5: 1×1 conv; no relu (RELU=0 would need different instance)
                // Using the same instance with relu; sigmoid applied in S_OUT
                if (!conv_busy) begin
                    conv_start <= 1'b1;
                end
                if (conv_done) begin
                    // Apply sigmoid if DO_SIGMOID, else store raw
                    out_buf[pix_cnt] <= (DO_SIGMOID != 0) ? sig_q88 : conv_y;
                    pix_cnt <= pix_cnt + 1'b1;
                end
            end

            S_OUT: begin
                y_o     <= out_buf[pix_cnt];
                y_valid <= 1'b1;
                pix_cnt <= pix_cnt + 1'b1;
            end

            S_DONE: begin
                done    <= 1'b1;
                pix_cnt <= 17'd0;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
