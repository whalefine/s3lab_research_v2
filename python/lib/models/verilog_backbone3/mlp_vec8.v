// =============================================================================
// mlp_vec8.v -- verilog_backbone3 placeholder (TBD)
//
// FC1 (32->128) + ReLU + FC2 (128->32) using linear_vec8.
// Golden: backbone_blocks_<b>_mlp_after_mlp_out_bi.txt
// =============================================================================

module mlp #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input wire clk
);

// TBD

endmodule
