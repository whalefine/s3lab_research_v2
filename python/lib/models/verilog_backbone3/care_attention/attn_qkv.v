// =============================================================================
// attn_qkv.v -- verilog_backbone3 placeholder (TBD)
//
// norm1 staging read + linear_vec8 QKV -> Sram_q / Sram_k / Sram_v.
// Golden: backbone_blocks_<b>_attn_after_qkv_{q,k,v}_bi.txt
// =============================================================================

module attn_qkv #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320
) (
    input wire clk
);

// TBD

endmodule
