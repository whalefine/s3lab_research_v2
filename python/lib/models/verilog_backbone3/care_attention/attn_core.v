// =============================================================================
// attn_core.v -- verilog_backbone3 placeholder (TBD)
//
// S_ATTN: q@kv*zr -> ao on Sram_v. Keep 4-phase ADDR/MAC/DOT/AO (Fmax safe).
// Golden: backbone_blocks_<b>_after_attn_attn_out_bi.txt
// =============================================================================

module attn_core #(
    parameter EMBED_DIM = 32,
    parameter NUM_HEADS = 4,
    parameter HEAD_DIM  = 8,
    parameter N_TOKENS  = 320
) (
    input wire clk
);

// TBD

endmodule
