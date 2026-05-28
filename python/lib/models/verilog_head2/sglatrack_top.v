// =============================================================================
// sglatrack_top.v -- verilog_head2 wrapper (head-only)
//
// This wrapper keeps only head datapath for head-only verification.
// Input stream: post-backbone tokens (Q8.8) -> head_top -> bbox (Q8.8)
// =============================================================================

module sglatrack_top #(
    parameter DATA_W   = 16,
    parameter IN_CH    = 32,
    parameter C_SH1    = 96,
    parameter C_SH2    = 48,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter N_TOKENS = 320,
    parameter LENS_Z   = 64
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    start,
    output wire                    busy,
    output wire                    done,
    input  wire signed [DATA_W-1:0] data_in,
    input  wire                    data_valid,
    output wire [DATA_W-1:0]       cx_o,
    output wire [DATA_W-1:0]       cy_o,
    output wire [DATA_W-1:0]       w_o,
    output wire [DATA_W-1:0]       h_o
);

head_top #(
    .IN_CH    (IN_CH   ),
    .C_SH1    (C_SH1   ),
    .C_SH2    (C_SH2   ),
    .FEAT_H   (FEAT_H  ),
    .FEAT_W   (FEAT_W  ),
    .N_TOKENS (N_TOKENS),
    .LENS_Z   (LENS_Z  ),
    .DATA_W   (DATA_W  )
) u_head (
    .clk     (clk      ),
    .reset   (reset    ),
    .start   (start    ),
    .a_i     (data_in  ),
    .a_valid (data_valid),
    .busy    (busy     ),
    .done    (done     ),
    .cx_o    (cx_o     ),
    .cy_o    (cy_o     ),
    .w_o     (w_o      ),
    .h_o     (h_o      )
);

endmodule
