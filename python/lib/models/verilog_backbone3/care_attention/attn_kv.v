// =============================================================================
// attn_kv.v -- verilog_backbone3 placeholder (TBD)
//
// S_KV: parallel Sram_k + Sram_v read, outer-mean -> kv_buf reg[256].
// =============================================================================

module attn_kv #(
    parameter NUM_HEADS = 4,
    parameter HEAD_DIM  = 8,
    parameter N_TOKENS  = 320
) (
    input wire clk
);

// TBD

endmodule
