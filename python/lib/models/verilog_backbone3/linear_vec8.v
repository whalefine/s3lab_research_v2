// =============================================================================
// linear_vec8.v -- verilog_backbone3 placeholder (TBD)
//
// 8-wide linear (IN_DIM x OUT_DIM). FSM: S_WPRE -> S_MAC -> S_SAT.
// Golden-Weight: via parent w_addr_o decode (backbone_top ROM mux).
// =============================================================================

module linear_vec8 #(
    parameter IN_DIM  = 32,
    parameter OUT_DIM = 96,
    parameter LANES   = 8
) (
    input wire clk
);

// TBD

endmodule
