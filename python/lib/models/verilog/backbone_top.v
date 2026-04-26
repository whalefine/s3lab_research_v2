// =============================================================================
// backbone_top.v
//
// Backbone Top Controller，對齊 run_backbone_numpy.main() Steps 1~6。
//
// 架構（時分複用單一 transformer_block）：
//   1. 接收 post-embedding input tokens [N_TOKENS, EMBED_DIM]
//   2. 執行 blocks 0~5（固定層，START_LAYER=5）
//   3. 執行 adaptive_selector MLP → 選出 block_sel ∈ {6..11}
//   4. 執行 selected block
//   5. backbone.norm（layer_norm 對所有 token）
//   6. 輸出 backbone_out [N_TOKENS, EMBED_DIM]
//
// FSM 狀態：
//   S_IDLE → S_RUN_FIXED → S_RUN_SEL_MLP → S_RUN_SELECTED → S_BACKBONE_NORM → S_OUT → S_DONE
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
parameter S_RUN_SEL_MLP  = 3'd2;  // adaptive_selector
parameter S_RUN_SELECTED = 3'd3;  // one of blocks 6..11
parameter S_BACKBONE_NORM= 3'd4;  // final layer norm
parameter S_OUT          = 3'd5;  // stream output
parameter S_DONE         = 3'd6;

reg [2:0] state, next_state;
reg [3:0] block_idx;   // current block index [0..11]
reg [2:0] sel_block;   // selected block from adaptive_selector (0~5)

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

// adaptive_selector control
wire sel_busy, sel_done;
reg  sel_start;
wire [2:0] sel_out;

adaptive_selector #(
    .EMBED_DIM(EMBED_DIM)
) u_sel (
    .clk(clk), .reset(reset), .start(sel_start),
    .x_i(x_i), .x_valid(x_valid),
    .wgt_i(wgt_i), .bias_i(bias_i),
    .wgt_addr_o(wgt_addr_o[13:0]),
    .busy(sel_busy), .done(sel_done),
    .selected_block_o(sel_out)
);

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
        S_IDLE:          next_state = start    ? S_RUN_FIXED    : S_IDLE;
        S_RUN_FIXED:     next_state = (tb_done && block_idx == START_LAYER)
                                                 ? S_RUN_SEL_MLP : S_RUN_FIXED;
        S_RUN_SEL_MLP:   next_state = sel_done  ? S_RUN_SELECTED : S_RUN_SEL_MLP;
        S_RUN_SELECTED:  next_state = tb_done   ? S_BACKBONE_NORM: S_RUN_SELECTED;
        S_BACKBONE_NORM: next_state = (bn_done && bn_tok_cnt == N_TOKENS-1)
                                                 ? S_OUT          : S_BACKBONE_NORM;
        S_OUT:           next_state = (out_rd_addr == N_TOKENS*EMBED_DIM-1)
                                                 ? S_DONE         : S_OUT;
        S_DONE:          next_state = S_IDLE;
        default:         next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done      <= 1'b0;
    tb_start  <= 1'b0;
    sel_start <= 1'b0;
    bn_start  <= 1'b0;

    if (reset) begin
        block_idx   <= 4'd0;
        sel_block   <= 3'd0;
        bn_tok_cnt  <= 9'd0;
        out_rd_addr <= 18'd0;
    end else begin
        case (state)
            S_IDLE: begin
                block_idx   <= 4'd0;
                out_rd_addr <= 18'd0;
            end

            S_RUN_FIXED: begin
                // Run blocks 0..START_LAYER sequentially
                if (!tb_busy) begin
                    tb_start <= 1'b1;
                end
                if (tb_done) begin
                    if (block_idx < START_LAYER)
                        block_idx <= block_idx + 4'd1;
                    // else stay; next state transition handles it
                end
            end

            S_RUN_SEL_MLP: begin
                // Run adaptive selector (uses block 5 output first token)
                if (!sel_busy) sel_start <= 1'b1;
                if (sel_done) sel_block <= sel_out;
            end

            S_RUN_SELECTED: begin
                // Run the selected block (block_idx = sel_block + 6)
                block_idx <= {1'b0, sel_block} + 4'd6;
                if (!tb_busy) tb_start <= 1'b1;
            end

            S_BACKBONE_NORM: begin
                // Run layer_norm for all N_TOKENS tokens
                if (!bn_busy) begin
                    bn_start <= 1'b1;
                end
                if (bn_y_valid) begin
                    // Store into out_buf (token by token)
                    // [simplified: actual token/feat address management needed]
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
