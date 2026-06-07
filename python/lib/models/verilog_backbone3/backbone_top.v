// =============================================================================
// backbone_top.v -- verilog_backbone3 placeholder (TBD)
//
// Blocks 0..START_LAYER + adaptive block + backbone norm.
// Port list matches verilog_backbone2 for sglatrack_top reuse (6 SRAM macros).
// =============================================================================

module backbone_top #(
    parameter EMBED_DIM   = 32,
    parameter N_TOKENS    = 320,
    parameter START_LAYER = 5,
    parameter N_BLOCKS    = 12
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire [3:0]  sel_block_i,
    input  wire signed [15:0] x_i,
    input  wire        x_valid,
    output wire        busy,
    output wire        x_ready,
    output reg         done,
    output wire signed [15:0] y_o,
    output wire        y_valid,
    output wire        sram_tok1_ceb_o,
    output wire        sram_tok1_web_o,
    output wire [13:0] sram_tok1_addr_o,
    output wire [15:0] sram_tok1_din_o,
    input  wire [15:0] sram_tok1_q_i,
    output wire        sram_tok2_ceb_o,
    output wire        sram_tok2_web_o,
    output wire [13:0] sram_tok2_addr_o,
    output wire [15:0] sram_tok2_din_o,
    input  wire [15:0] sram_tok2_q_i,
    output wire        sram_q_ceb_o,
    output wire        sram_q_web_o,
    output wire [13:0] sram_q_addr_o,
    output wire [15:0] sram_q_din_o,
    input  wire [15:0] sram_q_q_i,
    output wire        sram_k_ceb_o,
    output wire        sram_k_web_o,
    output wire [13:0] sram_k_addr_o,
    output wire [15:0] sram_k_din_o,
    input  wire [15:0] sram_k_q_i,
    output wire        sram_v_ceb_o,
    output wire        sram_v_web_o,
    output wire [13:0] sram_v_addr_o,
    output wire [15:0] sram_v_din_o,
    input  wire [15:0] sram_v_q_i,
    output wire        sram_qkm_ceb_o,
    output wire        sram_qkm_web_o,
    output wire [13:0] sram_qkm_addr_o,
    output wire [15:0] sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i
);

// TBD: stub tie-offs until implementation
assign busy    = 1'b0;
assign x_ready = 1'b0;
assign y_o     = 16'sd0;
assign y_valid = 1'b0;

assign sram_tok1_ceb_o  = 1'b1;
assign sram_tok1_web_o  = 1'b1;
assign sram_tok1_addr_o = 14'd0;
assign sram_tok1_din_o  = 16'd0;

assign sram_tok2_ceb_o  = 1'b1;
assign sram_tok2_web_o  = 1'b1;
assign sram_tok2_addr_o = 14'd0;
assign sram_tok2_din_o  = 16'd0;

assign sram_q_ceb_o  = 1'b1;
assign sram_q_web_o  = 1'b1;
assign sram_q_addr_o = 14'd0;
assign sram_q_din_o  = 16'd0;

assign sram_k_ceb_o  = 1'b1;
assign sram_k_web_o  = 1'b1;
assign sram_k_addr_o = 14'd0;
assign sram_k_din_o  = 16'd0;

assign sram_v_ceb_o  = 1'b1;
assign sram_v_web_o  = 1'b1;
assign sram_v_addr_o = 14'd0;
assign sram_v_din_o  = 16'd0;

assign sram_qkm_ceb_o  = 1'b1;
assign sram_qkm_web_o  = 1'b1;
assign sram_qkm_addr_o = 14'd0;
assign sram_qkm_din_o  = 16'd0;

always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else
        done <= 1'b0;
end

endmodule
