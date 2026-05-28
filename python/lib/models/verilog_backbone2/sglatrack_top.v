// =============================================================================
// sglatrack_top.v -- verilog_backbone2 wrapper (backbone-only)
//
// This wrapper keeps only backbone datapath for backbone-only verification.
// Input stream: template/search post-embed tokens (Q8.8) -> backbone_top output.
// =============================================================================

module sglatrack_top #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    start,
    output wire                    busy,
    output wire                    done,
    input  wire [3:0]              sel_block_i,
    input  wire signed [15:0]      data_in,
    input  wire                    data_valid,
    output wire signed [15:0]      data_o,
    output wire                    data_o_valid
);

backbone_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_backbone (
    .clk        (clk),
    .reset      (reset),
    .start      (start),
    .sel_block_i(sel_block_i),
    .x_i        (data_in),
    .x_valid    (data_valid),
    .busy       (busy),
    .done       (done),
    .y_o        (data_o),
    .y_valid    (data_o_valid)
);

endmodule
