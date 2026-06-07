// =============================================================================
// transformer_block.v -- verilog_backbone3 placeholder (TBD)
//
// norm1 -> attn -> res1 -> norm2 -> mlp -> res2.
// Golden: backbone_blocks_<b>_after_block_out_bi.txt
// =============================================================================

module transformer_block #(
    parameter EMBED_DIM = 32,
    parameter MLP_DIM   = 128,
    parameter N_TOKENS  = 320
) (
    input wire clk
);

// TBD

endmodule
