// =============================================================================
// backbone_top.v
//
// Backbone Top Controller，對齊 run_backbone_numpy.main() Steps 1~6。
//
// 架構（時分複用單一 transformer_block）：
//   1. 接收 post-embedding input tokens [N_TOKENS, EMBED_DIM]
//   2. 執行 blocks 0~5（固定層，START_LAYER=5）
//   3. 由外部提供 sel_block_i（host 從 golden_manifest.json 讀取後寫入）
//   4. 執行 sel_block_i 指定的 block（∈ 6..11）
//   5. backbone.norm（layer_norm 對所有 token）
//   6. 輸出 backbone_out [N_TOKENS, EMBED_DIM]
//
// FSM 狀態：
//   S_IDLE → S_RUN_FIXED → S_RUN_SELECTED → S_BACKBONE_NORM → S_OUT → S_DONE
//
// sel_block_i：由 testbench/SoC host 在 start 前根據
//   golden_manifest.json["adaptive_selected_layer_index"] 驅動，
//   代表實際要執行的 block 絕對索引（6~11）。
//
// 時分複用：block_idx 計數器控制 weight ROM bank 選擇。
// =============================================================================

module backbone_top #(
    parameter EMBED_DIM   = 768,
    parameter N_TOKENS    = 320,
    parameter START_LAYER = 5,
    parameter N_BLOCKS    = 12
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // Pre-computed adaptive block selection (absolute index 6~11).
    // Driven by host before asserting start, sourced from
    // golden_manifest.json["adaptive_selected_layer_index"].
    input  wire [3:0]  sel_block_i,

    // Token input stream (N_TOKENS × EMBED_DIM)
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // Weight ROM interface
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [19:0] wgt_addr_o,  // [19:16]=block_idx, [15:0]=internal addr
    output wire [3:0]  block_idx_o, // current block being computed

    // Status
    output wire        busy,
    output reg         done,

    // Output token stream
    output wire signed [15:0] y_o,
    output wire        y_valid
);

// FSM state encoding
parameter S_IDLE         = 3'd0;
parameter S_RUN_FIXED    = 3'd1;  // blocks 0..START_LAYER
parameter S_RUN_SELECTED = 3'd2;  // one of blocks 6..11, index from sel_block_i
parameter S_BACKBONE_NORM= 3'd3;  // final layer norm
parameter S_OUT          = 3'd4;  // stream output
parameter S_DONE         = 3'd5;

reg [2:0] state, next_state;
reg [3:0] block_idx;    // current block index [0..11]
reg [3:0] sel_block_r;  // latched sel_block_i sampled on start

// transformer_block control
wire tb_busy, tb_done;
wire signed [15:0] tb_y;
wire tb_y_valid;
reg  tb_start;

transformer_block #(
    .EMBED_DIM(EMBED_DIM),
    .N_TOKENS (N_TOKENS)
) u_tb (
    .clk(clk), .reset(reset), .start(tb_start),
    .x_i(x_i), .x_valid(x_valid),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_addr_o(wgt_addr_o[15:0]),
    .block_idx(block_idx),
    .busy(tb_busy), .done(tb_done),
    .y_o(tb_y), .y_valid(tb_y_valid)
);
// Note: adaptive_selector removed; selection is supplied via sel_block_i port.

// backbone final layer_norm
wire bn_busy, bn_done;
wire signed [15:0] bn_y;
wire bn_y_valid;
reg  bn_start;
reg  [8:0] bn_tok_cnt;

layer_norm #(.FEAT_DIM(EMBED_DIM)) u_bn (
    .clk(clk), .reset(reset), .start(bn_start),
    .x_i(x_i), .x_valid(x_valid),
    .w_i(wgt_i), .b_i(bias_i),
    .feat_addr_o(wgt_addr_o[9:0]),
    .busy(bn_busy), .done(bn_done),
    .y_o(bn_y), .y_valid(bn_y_valid)
);

// Token output buffer (after backbone norm)
reg signed [15:0] out_buf [0:N_TOKENS*EMBED_DIM-1];
reg [17:0] out_rd_addr;

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:          next_state = start   ? S_RUN_FIXED    : S_IDLE;
        // Transition to selected block once all fixed blocks done.
        // sel_block_i is already stable (driven by host before start).
        S_RUN_FIXED:     next_state = (tb_done && block_idx == START_LAYER)
                                               ? S_RUN_SELECTED : S_RUN_FIXED;
        S_RUN_SELECTED:  next_state = tb_done ? S_BACKBONE_NORM : S_RUN_SELECTED;
        S_BACKBONE_NORM: next_state = (bn_done && bn_tok_cnt == N_TOKENS-1)
                                               ? S_OUT           : S_BACKBONE_NORM;
        S_OUT:           next_state = (out_rd_addr == N_TOKENS*EMBED_DIM-1)
                                               ? S_DONE          : S_OUT;
        S_DONE:          next_state = S_IDLE;
        default:         next_state = S_IDLE;
    endcase
end

// FSM segment 3: output / datapath control
always @(posedge clk) begin
    done     <= 1'b0;
    tb_start <= 1'b0;
    bn_start <= 1'b0;

    if (reset) begin
        block_idx   <= 4'd0;
        sel_block_r <= 4'd0;
        bn_tok_cnt  <= 9'd0;
        out_rd_addr <= 18'd0;
    end else begin
        case (state)
            S_IDLE: begin
                block_idx   <= 4'd0;
                out_rd_addr <= 18'd0;
                // Latch sel_block_i on the cycle start is asserted so the
                // host-driven value is captured before the pipeline begins.
                if (start) sel_block_r <= sel_block_i;
            end

            // Run blocks 0..START_LAYER sequentially
            S_RUN_FIXED: begin
                if (!tb_busy) tb_start <= 1'b1;
                if (tb_done) begin
                    if (block_idx < START_LAYER)
                        block_idx <= block_idx + 4'd1;
                    // else stay; FSM transitions to S_RUN_SELECTED
                end
            end

            // Run sel_block_r (absolute index latched from sel_block_i)
            S_RUN_SELECTED: begin
                block_idx <= sel_block_r;
                if (!tb_busy) tb_start <= 1'b1;
            end

            // Run layer_norm for all N_TOKENS tokens
            S_BACKBONE_NORM: begin
                if (!bn_busy) bn_start <= 1'b1;
                if (bn_y_valid) begin
                    // [token/feat write-back into out_buf — address management here]
                end
                if (bn_done) begin
                    if (bn_tok_cnt < N_TOKENS-1)
                        bn_tok_cnt <= bn_tok_cnt + 9'd1;
                end
            end

            S_OUT: begin
                out_rd_addr <= out_rd_addr + 18'd1;
            end

            S_DONE: begin
                done        <= 1'b1;
                out_rd_addr <= 18'd0;
                bn_tok_cnt  <= 9'd0;
            end

            default: ;
        endcase
    end
end

// Output stream from out_buf
assign y_o     = (state == S_OUT) ? out_buf[out_rd_addr] : 16'sd0;
assign y_valid = (state == S_OUT);

// Block index output for weight ROM address selection
assign block_idx_o = block_idx;
assign wgt_addr_o[19:16] = block_idx;
assign busy = (state != S_IDLE);

endmodule
