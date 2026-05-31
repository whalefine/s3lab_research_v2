// =============================================================================
// mlp.v
//
// Q8.8 MLP block: fc1 (EMBED_DIM -> MLP_DIM) -> ReLU -> fc2 (MLP_DIM -> EMBED_DIM).
// Bit-accurate mirror of numpy block_forward in
// run_backbone_numpy_shared_trunk.py (line 556).
//
// norm2 input: no local x_in_buf. Parent tmp-on-q (Sram_q macro) via
// norm_rd_en + norm_rd_flat (ADDR) and norm_x (USE, 1-cycle macro latency).
// Flatten: tok * EMBED_DIM + feat (row-major, same as layer_norm norm2 capture).
//
// Per-token sequencing:
//   S_FC1 : u_fc1 with 2-phase norm2 read per input feature
//   S_FC2 : u_fc2 from fc1_buf[0..MLP_DIM-1] (256 B reg scratch)
//
// Golden-Weight: box_head_* via backbone_top ROM decode (fc1/fc2 wtype).
// =============================================================================

module mlp #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // 2-phase read of parent norm2 buffer (transformer_block tmp-on-q / Sram_q)
    output reg         norm_rd_en,
    output reg [13:0] norm_rd_flat,
    input  wire signed [15:0] norm_x,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [15:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    output reg  signed [15:0] y_o,
    output reg         y_valid
);

// 4-bit FSM (no S_LOAD_X)
parameter S_IDLE    = 4'd0;
parameter S_FC1     = 4'd1;
parameter S_FC2     = 4'd2;
parameter S_DONE_ST = 4'd3;

reg [3:0] state, next_state;

reg signed [15:0] fc1_buf [0:MLP_DIM-1];

`ifndef SYNTHESIS
integer mlp_ii;
initial begin
    for (mlp_ii = 0; mlp_ii < MLP_DIM; mlp_ii = mlp_ii + 1)
        fc1_buf[mlp_ii] = 16'sd0;
end
`endif

reg                fc1_start;
reg                fc1_x_phase;
reg  signed [15:0] fc1_x;
reg                fc1_xv;
wire signed [15:0] fc1_y;
wire               fc1_yv;
wire [6:0]         fc1_neu;
wire [12:0]        fc1_addr;
wire               fc1_busy, fc1_done;

linear #(.IN_DIM(EMBED_DIM), .OUT_DIM(MLP_DIM)) u_fc1 (
    .clk     (clk),
    .reset   (reset),
    .start   (fc1_start),
    .x_i     (fc1_x),
    .x_valid (fc1_xv),
    .w_i     (wgt_i),
    .b_i     (bias_i),
    .w_addr_o(fc1_addr),
    .busy    (fc1_busy),
    .done    (fc1_done),
    .y_o     (fc1_y),
    .y_valid (fc1_yv),
    .y_neu_o (fc1_neu)
);

reg                fc2_start;
reg  signed [15:0] fc2_x;
reg                fc2_xv;
wire signed [15:0] fc2_y;
wire               fc2_yv;
wire [4:0]         fc2_neu;
wire [12:0]        fc2_addr;
wire               fc2_busy, fc2_done;

linear_wide #(.IN_DIM(MLP_DIM), .OUT_DIM(EMBED_DIM)) u_fc2 (
    .clk     (clk),
    .reset   (reset),
    .start   (fc2_start),
    .x_i     (fc2_x),
    .x_valid (fc2_xv),
    .w_i     (wgt_i),
    .b_i     (bias_i),
    .w_addr_o(fc2_addr),
    .busy    (fc2_busy),
    .done    (fc2_done),
    .y_o     (fc2_y),
    .y_valid (fc2_yv),
    .y_neu_o (fc2_neu)
);

assign wgt_addr_o = (state == S_FC1) ? {3'b100, fc1_addr} :
                    (state == S_FC2) ? {3'b101, fc2_addr} :
                                        16'b0;

reg [8:0]  tok_cnt;
reg [6:0]  fc1_stream_cnt;
reg [7:0]  fc2_stream_cnt;

always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

always @(*) begin
    case (state)
        S_IDLE:    next_state = start ? S_FC1 : S_IDLE;
        S_FC1:     next_state = fc1_done ? S_FC2 : S_FC1;
        S_FC2:     next_state = (fc2_done && tok_cnt == N_TOKENS[8:0] - 9'd1) ? S_DONE_ST :
                                fc2_done                                       ? S_FC1     :
                                                                                  S_FC2;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

always @(posedge clk) begin
    done        <= 1'b0;
    y_valid     <= 1'b0;
    fc1_start   <= 1'b0;
    fc2_start   <= 1'b0;
    fc1_xv      <= 1'b0;
    fc2_xv      <= 1'b0;
    norm_rd_en  <= 1'b0;
    norm_rd_flat<= 14'd0;

    if (reset) begin
        tok_cnt        <= 9'd0;
        fc1_stream_cnt <= 7'd0;
        fc2_stream_cnt <= 8'd0;
        fc1_x_phase    <= 1'b0;
        fc1_x          <= 16'sd0;
        fc2_x          <= 16'sd0;
        y_o            <= 16'sd0;
    end else begin
        if (state == S_FC1 && fc1_yv)
            fc1_buf[fc1_neu] <= fc1_y[15] ? 16'sd0 : fc1_y;

        if (state == S_FC2 && fc2_yv) begin
            y_o     <= fc2_y;
            y_valid <= 1'b1;
        end

        case (state)
            S_IDLE: begin
                tok_cnt        <= 9'd0;
                fc1_stream_cnt <= 7'd0;
                fc2_stream_cnt <= 8'd0;
                fc1_x_phase    <= 1'b0;
            end

            S_FC1: begin
                if (fc1_stream_cnt == 7'd0) begin
                    fc1_start      <= 1'b1;
                    fc1_stream_cnt <= 7'd1;
                    fc1_x_phase    <= 1'b0;
                end else if (fc1_stream_cnt <= EMBED_DIM[6:0]) begin
                    if (fc1_x_phase == 1'b0) begin
                        norm_rd_en   <= 1'b1;
                        norm_rd_flat <= {5'd0, tok_cnt} * EMBED_DIM +
                                        {7'd0, fc1_stream_cnt - 7'd1};
                        fc1_x_phase  <= 1'b1;
                    end else begin
                        fc1_x          <= norm_x;
                        fc1_xv         <= 1'b1;
                        fc1_stream_cnt <= fc1_stream_cnt + 7'd1;
                        fc1_x_phase    <= 1'b0;
                    end
                end

                if (fc1_done) begin
                    fc1_stream_cnt <= 7'd0;
                    fc2_stream_cnt <= 8'd0;
                    fc1_x_phase    <= 1'b0;
                end
            end

            S_FC2: begin
                if (fc2_stream_cnt == 8'd0) begin
                    fc2_start      <= 1'b1;
                    fc2_stream_cnt <= 8'd1;
                end else if (fc2_stream_cnt <= MLP_DIM[7:0]) begin
                    fc2_x          <= fc1_buf[fc2_stream_cnt - 8'd1];
                    fc2_xv         <= 1'b1;
                    fc2_stream_cnt <= fc2_stream_cnt + 8'd1;
                end

                if (fc2_done) begin
                    fc2_stream_cnt <= 8'd0;
                    if (tok_cnt == N_TOKENS[8:0] - 9'd1)
                        tok_cnt <= 9'd0;
                    else begin
                        tok_cnt        <= tok_cnt + 9'd1;
                        fc1_stream_cnt <= 7'd0;
                    end
                end
            end

            S_DONE_ST: done <= 1'b1;

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
