// =============================================================================
// mlp.v
//
// Q8.8 MLP block: fc1 (EMBED_DIM → MLP_DIM) → ReLU → fc2 (MLP_DIM → EMBED_DIM).
// Bit-accurate mirror of numpy block_forward in
// run_backbone_numpy_shared_trunk.py (line 556):
//   mlp_out = fp(linear(fp(relu(fp(linear(x_norm2, fc1_w, fc1_b)))), fc2_w, fc2_b))
//
// All fp() calls collapse to identity here because both linears (linear.v /
// linear_wide.v) already return saturated Q8.8 values. ReLU = max(0, x).
//
// Per-token sequencing:
//   1. S_LOAD_X : capture N_TOKENS * EMBED_DIM x values (norm2 output stream).
//   2. S_FC1    : per token, run u_fc1 (IN=EMBED_DIM, OUT=MLP_DIM).
//                 Capture u_fc1.y_o through ReLU into fc1_buf[0..MLP_DIM-1].
//   3. S_FC2    : per token, run u_fc2 (IN=MLP_DIM, OUT=EMBED_DIM).
//                 Forward u_fc2.y_o to module y_o (one beat per neuron).
//   Loop S_FC1/S_FC2 across all N_TOKENS tokens.
//
// ROM addressing: wgt_addr_o[15:13] = wtype (100 = fc1, 101 = fc2). Lower 13
// bits come from the active linear submodule. backbone_top.v decodes:
//   fc1 weight : addr = block*4096 + local[12:0] (blocks 0-3 / 4-6 ROMs)
//   fc1 bias   : addr = block*128  + local[11:5]
//   fc2 weight : addr = block*4096 + local[12:0]
//   fc2 bias   : addr = block*32   + local[11:7]
// linear.v packs {1'b0, neu[6:0], feat[4:0]} (suits fc1); linear_wide.v packs
// {1'b0, neu[4:0], feat[6:0]} (suits fc2). Both match the bias decoders above.
//
// Buffer sizes (sim-only reg arrays; APR should replace with SRAM):
//   x_in_buf : N_TOKENS * EMBED_DIM = 10240 × 16-bit
//   fc1_buf  : MLP_DIM                = 128   × 16-bit (per-token scratchpad)
// =============================================================================

module mlp #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // norm2 → MLP stream input (Q8.8, N_TOKENS * EMBED_DIM beats)
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // ROM weight / bias (1-cycle latency from wgt_addr_o)
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [15:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    // MLP output stream (Q8.8, N_TOKENS * EMBED_DIM beats)
    output reg  signed [15:0] y_o,
    output reg         y_valid
);

parameter X_ELEMS = N_TOKENS * EMBED_DIM;   // 10240

// 4-bit FSM
parameter S_IDLE   = 4'd0;
parameter S_LOAD_X = 4'd1;
parameter S_FC1    = 4'd2;
parameter S_FC2    = 4'd3;
parameter S_DONE_ST= 4'd4;

reg [3:0] state, next_state;

reg signed [15:0] x_in_buf [0:X_ELEMS-1];
reg signed [15:0] fc1_buf  [0:MLP_DIM-1];

`ifndef SYNTHESIS
integer mlp_ii;
initial begin
    for (mlp_ii = 0; mlp_ii < X_ELEMS;  mlp_ii = mlp_ii + 1) x_in_buf[mlp_ii] = 16'sd0;
    for (mlp_ii = 0; mlp_ii < MLP_DIM;  mlp_ii = mlp_ii + 1) fc1_buf [mlp_ii] = 16'sd0;
end
`endif

// ---------------------------------------------------------------------------
// u_fc1 : linear (IN=EMBED_DIM=32, OUT=MLP_DIM=128).
//   linear.v's address packing {1'b0, neu[6:0], feat[4:0]} matches the fc1
//   bias decoder (local[11:5] = neu) in backbone_top.v.
// ---------------------------------------------------------------------------
reg                fc1_start;
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

// ---------------------------------------------------------------------------
// u_fc2 : linear_wide (IN=MLP_DIM=128, OUT=EMBED_DIM=32).
//   Address packing {1'b0, neu[4:0], feat[6:0]} matches the fc2 bias decoder
//   (local[11:7] = neu) in backbone_top.v.
// ---------------------------------------------------------------------------
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

// Route ROM weight address with the right wtype prefix
assign wgt_addr_o = (state == S_FC1) ? {3'b100, fc1_addr} :
                    (state == S_FC2) ? {3'b101, fc2_addr} :
                                        16'b0;

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------
reg [13:0] load_ptr;          // S_LOAD_X 0..X_ELEMS-1
reg [8:0]  tok_cnt;           // current token index (0..N_TOKENS-1)
reg [6:0]  fc1_stream_cnt;    // 0..EMBED_DIM (32) for fc1 streaming (need 1 extra value)
reg [7:0]  fc2_stream_cnt;    // 0..MLP_DIM   (128) for fc2 streaming (need 1 extra value)

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:    next_state = start ? S_LOAD_X : S_IDLE;
        S_LOAD_X:  next_state = (load_ptr == X_ELEMS[13:0] - 14'd1 && x_valid) ? S_FC1 : S_LOAD_X;
        S_FC1:     next_state = fc1_done ? S_FC2 : S_FC1;
        S_FC2:     next_state = (fc2_done && tok_cnt == N_TOKENS[8:0] - 9'd1) ? S_DONE_ST :
                                fc2_done                                       ? S_FC1     :
                                                                                  S_FC2;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: datapath
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    done       <= 1'b0;
    y_valid    <= 1'b0;
    fc1_start  <= 1'b0;
    fc2_start  <= 1'b0;
    fc1_xv     <= 1'b0;
    fc2_xv     <= 1'b0;

    if (reset) begin
        load_ptr        <= 14'd0;
        tok_cnt         <= 9'd0;
        fc1_stream_cnt  <= 7'd0;
        fc2_stream_cnt  <= 8'd0;
        fc1_x           <= 16'sd0;
        fc2_x           <= 16'sd0;
        y_o             <= 16'sd0;
    end else begin
        // ------------------------------------------------------------
        // Capture fc1 outputs into fc1_buf with ReLU. Active across all
        // S_FC1 cycles regardless of streaming sub-phase.
        // ------------------------------------------------------------
        if (state == S_FC1 && fc1_yv) begin
            // ReLU: clamp negative to 0 (Q8.8 sign bit = bit 15)
            fc1_buf[fc1_neu] <= fc1_y[15] ? 16'sd0 : fc1_y;
        end

        // ------------------------------------------------------------
        // Forward fc2 outputs to module y_o (one beat per fc2 neuron).
        // ------------------------------------------------------------
        if (state == S_FC2 && fc2_yv) begin
            y_o     <= fc2_y;
            y_valid <= 1'b1;
        end

        case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                load_ptr        <= 14'd0;
                tok_cnt         <= 9'd0;
                fc1_stream_cnt  <= 7'd0;
                fc2_stream_cnt  <= 8'd0;
            end

            // -----------------------------------------------------------
            // S_LOAD_X: capture incoming norm2 stream into x_in_buf
            // -----------------------------------------------------------
            S_LOAD_X: begin
                if (x_valid && load_ptr < X_ELEMS[13:0]) begin
                    x_in_buf[load_ptr] <= x_i;
                    load_ptr <= load_ptr + 14'd1;
                end
                if (next_state == S_FC1) begin
                    tok_cnt        <= 9'd0;
                    fc1_stream_cnt <= 7'd0;
                end
            end

            // -----------------------------------------------------------
            // S_FC1: drive u_fc1 for current token
            //   phase 0     : pulse fc1_start
            //   phase 1..32 : stream x_in_buf[tok*32 + (phase-1)] with xv=1
            //   phase  >=33 : wait for fc1_done (idle on x stream)
            // -----------------------------------------------------------
            S_FC1: begin
                if (fc1_stream_cnt == 7'd0) begin
                    fc1_start      <= 1'b1;
                    fc1_stream_cnt <= 7'd1;
                end else if (fc1_stream_cnt <= EMBED_DIM[6:0]) begin
                    fc1_x          <= x_in_buf[{5'd0, tok_cnt} * EMBED_DIM +
                                               {7'd0, fc1_stream_cnt - 7'd1}];
                    fc1_xv         <= 1'b1;
                    fc1_stream_cnt <= fc1_stream_cnt + 7'd1;
                end

                if (fc1_done) begin
                    // fc1 finished for this token; prepare for fc2.
                    fc1_stream_cnt <= 7'd0;
                    fc2_stream_cnt <= 8'd0;
                end
            end

            // -----------------------------------------------------------
            // S_FC2: drive u_fc2 for current token (using fc1_buf contents)
            //   phase 0       : pulse fc2_start
            //   phase 1..128  : stream fc1_buf[phase-1] with xv=1
            //   phase >=129   : wait for fc2_done
            // -----------------------------------------------------------
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
                    if (tok_cnt == N_TOKENS[8:0] - 9'd1) begin
                        tok_cnt <= 9'd0;
                    end else begin
                        tok_cnt        <= tok_cnt + 9'd1;
                        fc1_stream_cnt <= 7'd0;
                    end
                end
            end

            // -----------------------------------------------------------
            S_DONE_ST: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
