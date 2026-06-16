// =============================================================================
// recip_lut_seed.v
//
// 16-entry Q8.8 reciprocal seed LUT for Newton-Raphson recip_nr.
//
// Mirrors numpy run_backbone_numpy_shared_trunk.py:
//   _RECIP_LUT_Y0 = [32767, 21845, 10922, 5461, 2731, 1365, 683, 341,
//                     171,    85,    43,   21,   11,    5,   3,   2]
//   k = leading-bit index (highest set bit of x, 0..15)
//
// Inputs:
//   x_i [15:0]  Q8.8 signed positive (caller clamps x >= 1)
// Outputs:
//   y0_o [15:0] Q8.8 seed estimate
//   k_o  [3:0]  MSB index (sim debug)
//
// Pure combinational, no clk. Synthesizes to priority encoder + small case.
// =============================================================================

module recip_lut_seed (
    input  wire signed [15:0] x_i,
    output reg         [15:0] y0_o,
    output reg         [3:0]  k_o
);

// Priority encoder: highest set bit of x_i (treat as unsigned 16-bit since x>=1)
always @(*) begin
    casez (x_i[15:0])
        16'b1???????????????: k_o = 4'd15;
        16'b01??????????????: k_o = 4'd14;
        16'b001?????????????: k_o = 4'd13;
        16'b0001????????????: k_o = 4'd12;
        16'b00001???????????: k_o = 4'd11;
        16'b000001??????????: k_o = 4'd10;
        16'b0000001?????????: k_o = 4'd9;
        16'b00000001????????: k_o = 4'd8;
        16'b000000001???????: k_o = 4'd7;
        16'b0000000001??????: k_o = 4'd6;
        16'b00000000001?????: k_o = 4'd5;
        16'b000000000001????: k_o = 4'd4;
        16'b0000000000001???: k_o = 4'd3;
        16'b00000000000001??: k_o = 4'd2;
        16'b000000000000001?: k_o = 4'd1;
        default:              k_o = 4'd0;  // x_i==0 (caller should clamp)
    endcase
end

// LUT: seed y0 (Q8.8) for reciprocal, matches numpy _RECIP_LUT_Y0
always @(*) begin
    case (k_o)
        4'd15: y0_o = 16'd2;
        4'd14: y0_o = 16'd3;
        4'd13: y0_o = 16'd5;
        4'd12: y0_o = 16'd11;
        4'd11: y0_o = 16'd21;
        4'd10: y0_o = 16'd43;
        4'd9 : y0_o = 16'd85;
        4'd8 : y0_o = 16'd171;
        4'd7 : y0_o = 16'd341;
        4'd6 : y0_o = 16'd683;
        4'd5 : y0_o = 16'd1365;
        4'd4 : y0_o = 16'd2731;
        4'd3 : y0_o = 16'd5461;
        4'd2 : y0_o = 16'd10922;
        4'd1 : y0_o = 16'd21845;
        4'd0 : y0_o = 16'd32767;
        default: y0_o = 16'd32767;
    endcase
end

endmodule
