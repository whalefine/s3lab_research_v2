// =============================================================================
// residual.v
//
// Q8.8 element-wise residual add: y = sat16(a + b).
//
// Mirrors numpy fp(x + attn_out) / fp(x + mlp_out) used by block_forward in
// python/tracking/run_backbone_numpy_shared_trunk.py (lines 551, 558).
// Inputs a_i, b_i are Q8.8 16-bit signed. The 17-bit sum is saturated to Q8.8.
//
// Pipelined: a_i/b_i sampled at posedge T with v_i=1 → y_o/v_o appear at T+1.
// One add per cycle (the user explicitly asked for "獨立一拍做這件事"); a
// transformer_block-level pointer walks the buffer and feeds this unit one
// element per cycle. The same instance is reused for residual1 (x + attn) and
// residual2 (res1 + mlp) by muxing the operand sources in transformer_block.v.
//
// This module is fully synthesizable Verilog-2005. No latch (output regs cover
// all cases under sync reset).
// =============================================================================

module residual #(
    parameter WIDTH = 16
) (
    input  wire                   clk,
    input  wire                   reset,
    input  wire signed [WIDTH-1:0] a_i,
    input  wire signed [WIDTH-1:0] b_i,
    input  wire                   v_i,
    output reg  signed [WIDTH-1:0] y_o,
    output reg                    v_o
);

// 17-bit sign-extended add (covers the worst case where both inputs are 16-bit
// signed extremes). Saturate back to WIDTH-bit signed.
wire signed [WIDTH:0] sum_w =
    {a_i[WIDTH-1], a_i} + {b_i[WIDTH-1], b_i};

// Saturate (WIDTH+1)-bit signed sum to WIDTH-bit signed.
function signed [WIDTH-1:0] sat_w;
    input signed [WIDTH:0] v;
    begin
        if (v > $signed({1'b0, {(WIDTH-1){1'b1}}}))
            sat_w = {1'b0, {(WIDTH-1){1'b1}}};
        else if (v < $signed({1'b1, {(WIDTH-1){1'b0}}}))
            sat_w = {1'b1, {(WIDTH-1){1'b0}}};
        else
            sat_w = v[WIDTH-1:0];
    end
endfunction

// 1-cycle output pipeline. Sync reset only.
always @(posedge clk) begin
    if (reset) begin
        y_o <= {WIDTH{1'b0}};
        v_o <= 1'b0;
    end else begin
        y_o <= sat_w(sum_w);
        v_o <= v_i;
    end
end

endmodule
