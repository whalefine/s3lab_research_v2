// =============================================================================
// care_attention.v -- verilog_backbone3 placeholder (TBD)
//
// Thin FSM wiring attn_qkv / preprocess / z_recip / kv / core / proj.
// 6-SRAM plan unchanged (Sram_q/k/v/qkm + parent Sram_tok1 norm1 staging).
// =============================================================================

module care_attention #(
    parameter EMBED_DIM   = 32,
    parameter NUM_HEADS   = 4,
    parameter HEAD_DIM    = 8,
    parameter N_TOKENS    = 320,
    parameter S_Q88       = 152,
    parameter RELU6_MAX   = 1536,
    parameter RCP_N_NUM   = 205,
    parameter RCP_N_SHIFT = 16
) (
    input wire clk
);

// TBD

endmodule
