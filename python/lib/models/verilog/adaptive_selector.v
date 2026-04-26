// =============================================================================
// adaptive_selector.v
//
// Adaptive Block Selector MLP，對齊 run_backbone_numpy.main() Step 4。
//
// 輸入：Block 5 輸出的第一個 token（token index 0）之 EMBED_DIM 維特徵。
// 輸出：3-bit selected_block_o（0~5 對應 block 6~11）。
//
// 演算法：
//   mlp_in = x[:, 0, :]                           [1, EMBED_DIM]
//   h = relu(linear(mlp_in, fc1_w, fc1_b))        [1, MLP_HIDDEN]
//   pro = sigmoid(linear(h, fc2_w, fc2_b))        [1, OUT_DIM=6]
//   selected = argmax(pro) + 5 + 1  (block index)
//
// Sigmoid 使用 sigmoid_lut.v。
// 輸出 selected_block_o = argmax index（0~5）。
// backbone_top 加上 START_LAYER+1 = 6 得到實際 block idx。
//
// 介面：
//   - x_i 串流輸入 EMBED_DIM 個 Q8.8 值（start 後第一個值即有效）
//   - fc1/fc2 weights 由外部 ROM 提供（wgt_addr_o 定址）
//   - done 拉高時 selected_block_o 有效
//
// =============================================================================

module adaptive_selector #(
    parameter EMBED_DIM  = 768,
    parameter MLP_HIDDEN = 384,  // typical hidden size; adjust per model
    parameter OUT_DIM    = 6
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // First token feature stream (EMBED_DIM values)
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // fc1 / fc2 weights from external ROM
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output reg  [13:0] wgt_addr_o,

    // Status
    output wire        busy,
    output reg         done,

    // Result
    output reg  [2:0]  selected_block_o  // 0~5, meaning block 6~11
);

// FSM state encoding
parameter S_IDLE    = 3'd0;
parameter S_FC1     = 3'd1;   // fc1 neuron loop (MLP_HIDDEN outputs per token)
parameter S_FC2     = 3'd2;   // fc2 neuron loop (OUT_DIM outputs)
parameter S_SIGMOID = 3'd3;   // apply sigmoid to each of OUT_DIM values
parameter S_ARGMAX  = 3'd4;   // find argmax of 6 sigmoid outputs
parameter S_DONE    = 3'd5;

reg [2:0] state, next_state;
reg [9:0] neu_cnt;  // current output neuron counter

// h buffer: MLP_HIDDEN × 16 bit
reg signed [15:0] h_buf   [0:MLP_HIDDEN-1];
// pro buffer: OUT_DIM × 16 bit (sigmoid outputs)
reg signed [15:0] pro_buf [0:OUT_DIM-1];

// mlp_block fc1 (EMBED_DIM → MLP_HIDDEN, ReLU)
wire fc1_busy, fc1_done;
wire signed [15:0] fc1_y;
reg  fc1_start;

mlp_block #(.CIN(EMBED_DIM), .RELU(1)) u_fc1 (
    .clk(clk), .reset(reset), .start(fc1_start),
    .a_valid(x_valid), .a_i(x_i), .w_i(wgt_i), .b_i(bias_i),
    .busy(fc1_busy), .done(fc1_done), .y_o(fc1_y)
);

// mlp_block fc2 (MLP_HIDDEN → OUT_DIM, no ReLU)
wire fc2_busy, fc2_done;
wire signed [15:0] fc2_y;
reg  fc2_start, fc2_valid;
reg  signed [15:0] h_mux;

mlp_block #(.CIN(MLP_HIDDEN), .RELU(0)) u_fc2 (
    .clk(clk), .reset(reset), .start(fc2_start),
    .a_valid(fc2_valid), .a_i(h_mux), .w_i(wgt_i), .b_i(bias_i),
    .busy(fc2_busy), .done(fc2_done), .y_o(fc2_y)
);

// sigmoid_lut for each pro value
wire [7:0] sig_out;
reg  signed [15:0] sig_in;

sigmoid_lut u_sig (
    .x_i(sig_in),
    .y_o(sig_out)
);

// Argmax logic
reg [7:0] max_val;
reg [2:0] max_idx;
integer k;

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:    next_state = start   ? S_FC1     : S_IDLE;
        S_FC1:     next_state = (fc1_done && neu_cnt == MLP_HIDDEN-1)
                                         ? S_FC2     : S_FC1;
        S_FC2:     next_state = (fc2_done && neu_cnt == OUT_DIM-1)
                                         ? S_SIGMOID : S_FC2;
        S_SIGMOID: next_state = (neu_cnt == OUT_DIM-1) ? S_ARGMAX : S_SIGMOID;
        S_ARGMAX:  next_state = S_DONE;
        S_DONE:    next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done      <= 1'b0;
    fc1_start <= 1'b0;
    fc2_start <= 1'b0;
    fc2_valid <= 1'b0;

    if (reset) begin
        neu_cnt <= 10'd0;
        max_val <= 8'd0;
        max_idx <= 3'd0;
        selected_block_o <= 3'd0;
    end else begin
        case (state)
            S_IDLE: begin
                neu_cnt  <= 10'd0;
                wgt_addr_o <= 14'd0;
            end

            S_FC1: begin
                // Launch one fc1 neuron per iteration
                if (!fc1_busy) begin
                    fc1_start  <= 1'b1;
                    wgt_addr_o <= neu_cnt[13:0];
                end
                if (fc1_done) begin
                    h_buf[neu_cnt] <= fc1_y;
                    neu_cnt        <= neu_cnt + 1'b1;
                end
            end

            S_FC2: begin
                // Stream h_buf into fc2 for each output neuron
                if (!fc2_busy) begin
                    fc2_start  <= 1'b1;
                    wgt_addr_o <= (MLP_HIDDEN + neu_cnt)[13:0];
                    h_mux      <= h_buf[0];
                    fc2_valid  <= 1'b1;
                end
                if (fc2_done) begin
                    pro_buf[neu_cnt] <= fc2_y;
                    neu_cnt          <= neu_cnt + 1'b1;
                end
            end

            S_SIGMOID: begin
                // Apply sigmoid_lut (combinational) to each pro value
                sig_in    <= pro_buf[neu_cnt];
                // sig_out is combinational; convert Q0.8 to Q8.8 (shift left 8)
                pro_buf[neu_cnt] <= {8'd0, sig_out};
                neu_cnt   <= neu_cnt + 1'b1;
            end

            S_ARGMAX: begin
                // Find argmax over OUT_DIM values
                max_val <= 8'd0;
                max_idx <= 3'd0;
                begin : argmax_loop
                    integer j;
                    for (j = 0; j < OUT_DIM; j = j + 1) begin
                        if (pro_buf[j][7:0] > max_val) begin
                            max_val <= pro_buf[j][7:0];
                            max_idx <= j[2:0];
                        end
                    end
                end
                selected_block_o <= max_idx;
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
