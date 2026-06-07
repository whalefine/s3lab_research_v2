// =============================================================================
// mlp_vec8.v
//
// Q8.8 MLP block: fc1 (EMBED_DIM -> MLP_DIM) -> ReLU -> fc2 (MLP_DIM -> EMBED_DIM).
// Bit-accurate mirror of numpy block_forward in
// run_backbone_numpy_shared_trunk.py (line 556) and verilog_backbone2/mlp.v.
//
// norm2 input: no local x_in_buf. Parent tmp-on-q (Sram_q macro) via
// norm_rd_en + norm_rd_flat (ADDR) and norm_x (USE, 1-cycle macro latency).
// Flatten: tok * EMBED_DIM + feat (row-major, same as layer_norm norm2 capture).
//
// Per-token sequencing:
//   S_FC1 : u_fc1 (linear_vec8 32->128) with 2-phase norm2 read per input feature
//   S_FC2 : u_fc2 (linear_vec8 128->32) from fc1_buf[0..MLP_DIM-1]
//
// wgt_addr_o (backbone_top ROM decode):
//   3'b100 = fc1 : {3'b100, fc1_addr}
//   3'b101 = fc2 : {3'b101, fc2_addr}
//
// Golden: Activation/backbone_blocks_<b>_mlp_after_mlp_out_bi.txt
// Golden-Weight: Weight/backbone_blocks_<b>_mlp_fc{1,2}_{weight,bias}_bi.txt
// =============================================================================

module mlp_vec8 #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output reg         norm_rd_en,
    output reg [13:0]  norm_rd_flat,
    input  wire signed [15:0] norm_x,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [15:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    output reg  signed [15:0] y_o,
    output reg         y_valid
);

parameter S_IDLE    = 4'd0;
parameter S_FC1     = 4'd1;
parameter S_FC2     = 4'd2;
parameter S_DONE_ST = 4'd3;

reg [3:0] state;
reg [3:0] next_state;

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
wire               fc1_busy;
wire               fc1_done;

linear_vec8 #(
    .IN_DIM    (EMBED_DIM),
    .OUT_DIM   (MLP_DIM),
    .IN_DIM_AW (5),
    .NEU_AW    (8)
) u_fc1 (
    .clk      (clk),
    .reset    (reset),
    .start    (fc1_start),
    .x_i      (fc1_x),
    .x_valid  (fc1_xv),
    .w_i      (wgt_i),
    .b_i      (bias_i),
    .w_addr_o (fc1_addr),
    .busy     (fc1_busy),
    .done     (fc1_done),
    .y_o      (fc1_y),
    .y_valid  (fc1_yv),
    .y_neu_o  (fc1_neu)
);

reg                fc2_start;
reg  signed [15:0] fc2_x;
reg                fc2_xv;
wire signed [15:0] fc2_y;
wire               fc2_yv;
wire [6:0]         fc2_neu;
wire [12:0]        fc2_addr;
wire               fc2_busy;
wire               fc2_done;

linear_vec8 #(
    .IN_DIM    (MLP_DIM),
    .OUT_DIM   (EMBED_DIM),
    .IN_DIM_AW (7),
    .NEU_AW    (8)
) u_fc2 (
    .clk      (clk),
    .reset    (reset),
    .start    (fc2_start),
    .x_i      (fc2_x),
    .x_valid  (fc2_xv),
    .w_i      (wgt_i),
    .b_i      (bias_i),
    .w_addr_o (fc2_addr),
    .busy     (fc2_busy),
    .done     (fc2_done),
    .y_o      (fc2_y),
    .y_valid  (fc2_yv),
    .y_neu_o  (fc2_neu)
);

assign wgt_addr_o = (state == S_FC1) ? {3'b100, fc1_addr} :
                    (state == S_FC2) ? {3'b101, fc2_addr} :
                                        16'b0;

reg [8:0]  tok_cnt;
reg [6:0]  fc1_stream_cnt;
reg [7:0]  fc2_stream_cnt;

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
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE_ST)
        done <= 1'b1;
    else
        done <= 1'b0;
end

always @(posedge clk) begin
    if (reset) begin
        y_o     <= 16'sd0;
        y_valid <= 1'b0;
    end else begin
        y_valid <= 1'b0;
        if (state == S_FC2 && fc2_yv) begin
            y_o     <= fc2_y;
            y_valid <= 1'b1;
        end
    end
end

always @(posedge clk) begin
    if (!reset && state == S_FC1 && fc1_yv)
        fc1_buf[fc1_neu] <= fc1_y[15] ? 16'sd0 : fc1_y;
end

always @(posedge clk) begin
    if (reset)
        fc1_start <= 1'b0;
    else begin
        fc1_start <= 1'b0;
        if (state == S_FC1 && (fc1_stream_cnt == 7'd0))
            fc1_start <= 1'b1;
    end
end

always @(posedge clk) begin
    if (reset)
        fc2_start <= 1'b0;
    else begin
        fc2_start <= 1'b0;
        if (state == S_FC2 && (fc2_stream_cnt == 8'd0))
            fc2_start <= 1'b1;
    end
end

always @(posedge clk) begin
    if (reset)
        fc1_xv <= 1'b0;
    else begin
        fc1_xv <= 1'b0;
        if (state == S_FC1 && (fc1_stream_cnt != 7'd0) &&
            (fc1_stream_cnt <= EMBED_DIM[6:0]) && fc1_x_phase)
            fc1_xv <= 1'b1;
    end
end

always @(posedge clk) begin
    if (reset)
        fc2_xv <= 1'b0;
    else begin
        fc2_xv <= 1'b0;
        if (state == S_FC2 && (fc2_stream_cnt != 8'd0) &&
            (fc2_stream_cnt <= MLP_DIM[7:0]))
            fc2_xv <= 1'b1;
    end
end

always @(posedge clk) begin
    if (reset) begin
        norm_rd_en   <= 1'b0;
        norm_rd_flat <= 14'd0;
    end else begin
        norm_rd_en   <= 1'b0;
        norm_rd_flat <= 14'd0;
        if (state == S_FC1 && (fc1_stream_cnt != 7'd0) &&
            (fc1_stream_cnt <= EMBED_DIM[6:0])) begin
            norm_rd_en   <= 1'b1;
            norm_rd_flat <= {5'd0, tok_cnt} * EMBED_DIM +
                            {7'd0, fc1_stream_cnt - 7'd1};
        end
    end
end

always @(posedge clk) begin
    if (reset) begin
        fc1_stream_cnt <= 7'd0;
        fc1_x_phase    <= 1'b0;
        fc1_x          <= 16'sd0;
    end else if (state == S_IDLE) begin
        fc1_stream_cnt <= 7'd0;
        fc1_x_phase    <= 1'b0;
    end else if (state == S_FC1) begin
        if (fc1_done) begin
            fc1_stream_cnt <= 7'd0;
            fc1_x_phase    <= 1'b0;
        end else if (fc1_stream_cnt == 7'd0) begin
            fc1_stream_cnt <= 7'd1;
            fc1_x_phase    <= 1'b0;
        end else if (fc1_stream_cnt <= EMBED_DIM[6:0]) begin
            if (fc1_x_phase == 1'b0)
                fc1_x_phase <= 1'b1;
            else begin
                fc1_x          <= norm_x;
                fc1_stream_cnt <= fc1_stream_cnt + 7'd1;
                fc1_x_phase    <= 1'b0;
            end
        end
    end else if (state == S_FC2 && fc2_done &&
                 (tok_cnt != N_TOKENS[8:0] - 9'd1))
        fc1_stream_cnt <= 7'd0;
end

always @(posedge clk) begin
    if (reset)
        fc2_stream_cnt <= 8'd0;
    else if (state == S_IDLE)
        fc2_stream_cnt <= 8'd0;
    else if (state == S_FC1 && fc1_done)
        fc2_stream_cnt <= 8'd0;
    else if (state == S_FC2) begin
        if (fc2_stream_cnt == 8'd0)
            fc2_stream_cnt <= 8'd1;
        else if (fc2_stream_cnt <= MLP_DIM[7:0])
            fc2_stream_cnt <= fc2_stream_cnt + 8'd1;
        else if (fc2_done)
            fc2_stream_cnt <= 8'd0;
    end
end

always @(posedge clk) begin
    if (reset)
        fc2_x <= 16'sd0;
    else if (state == S_FC2 && (fc2_stream_cnt != 8'd0) &&
             (fc2_stream_cnt <= MLP_DIM[7:0]))
        fc2_x <= fc1_buf[fc2_stream_cnt - 8'd1];
end

always @(posedge clk) begin
    if (reset)
        tok_cnt <= 9'd0;
    else if (state == S_IDLE)
        tok_cnt <= 9'd0;
    else if (state == S_FC2 && fc2_done) begin
        if (tok_cnt == N_TOKENS[8:0] - 9'd1)
            tok_cnt <= 9'd0;
        else
            tok_cnt <= tok_cnt + 9'd1;
    end
end

assign busy = (state != S_IDLE);

endmodule
