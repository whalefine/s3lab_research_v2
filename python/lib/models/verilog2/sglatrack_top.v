// =============================================================================
// sglatrack_top.v -- verilog2 end-to-end (backbone_top + head_top)
//
// Flow: post-embed token stream -> backbone_top -> head_top -> bbox (Q8.8)
// Weights: ROM macros in backbone_top / conv / tail (compile with memory/*.v)
// Activation TXT_File/*_bi.txt: golden only (loaded by TEST.v testbench)
// =============================================================================

module sglatrack_top #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320,
    parameter FEAT_H    = 16,
    parameter FEAT_W    = 16
) (
    input  wire        clk,
    input  wire        reset,

    input  wire        start,
    output wire        busy,
    output reg         done,

    input  wire [3:0]  sel_block_i,

    input  wire signed [15:0] data_in,
    input  wire        data_valid,

    output wire [15:0] cx_o,
    output wire [15:0] cy_o,
    output wire [15:0] w_o,
    output wire [15:0] h_o
);

parameter S_IDLE     = 2'd0;
parameter S_BACKBONE = 2'd1;
parameter S_HEAD     = 2'd2;
parameter S_DONE     = 2'd3;

reg [1:0] state, next_state;

wire bb_busy, bb_done;
wire signed [15:0] bb_y;
wire bb_y_valid;
reg  bb_start;

backbone_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_backbone (
    .clk        (clk),
    .reset      (reset),
    .start      (bb_start),
    .sel_block_i(sel_block_i),
    .x_i        (data_in),
    .x_valid    (data_valid),
    .busy       (bb_busy),
    .done       (bb_done),
    .y_o        (bb_y),
    .y_valid    (bb_y_valid)
);

wire hd_busy, hd_done;
reg  hd_start;

head_top #(
    .IN_CH    (EMBED_DIM),
    .FEAT_H   (FEAT_H),
    .FEAT_W   (FEAT_W),
    .N_TOKENS (N_TOKENS)
) u_head (
    .clk      (clk),
    .reset    (reset),
    .start    (hd_start),
    .a_i      (bb_y),
    .a_valid  (bb_y_valid),
    .busy     (hd_busy),
    .done     (hd_done),
    .cx_o     (cx_o),
    .cy_o     (cy_o),
    .w_o      (w_o),
    .h_o      (h_o)
);

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
        S_IDLE:     next_state = start ? S_BACKBONE : S_IDLE;
        S_BACKBONE: next_state = bb_done ? S_HEAD : S_BACKBONE;
        S_HEAD:     next_state = hd_done ? S_DONE : S_HEAD;
        S_DONE:     next_state = S_IDLE;
        default:    next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic
always @(posedge clk) begin
    done     <= 1'b0;
    bb_start <= 1'b0;
    hd_start <= 1'b0;

    if (reset) begin
        ;
    end else begin
        case (state)
            S_IDLE: begin
                if (start) begin
                    bb_start <= 1'b1;
                    hd_start <= 1'b1;
                end
            end
            S_BACKBONE: ;
            S_HEAD:     ;
            S_DONE:     done <= 1'b1;
            default:    ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
