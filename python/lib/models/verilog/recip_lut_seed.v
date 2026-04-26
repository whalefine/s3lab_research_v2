// =============================================================================
// recip_lut_seed.v
//
// Leading-bit LUT：Q8.8 正值的 1/x 初始估值。
//
// 輸入  x_i [15:0]  Q8.8 unsigned（CARE attention qk_mean_eps，≥ 1/256）
// 輸出  y0_o [15:0] Q8.8 初始估值，≈ 256 / x_i_int
//
// 方法：
//   找 x_i 最高有效位位置 k（priority encoder），
//   查 16-entry LUT 回傳幾何中點倒數，即
//   y0 = round( 256 / (1.5 × 2^(k−8)) )
//
// 完全組合邏輯，無 clk。
// =============================================================================

module recip_lut_seed (
    input  wire [15:0] x_i,   // Q8.8 unsigned positive (≥ 1/256)
    output reg  [15:0] y0_o   // Q8.8 initial estimate for 1/x
);

// Leading-bit position encoder
reg [3:0] k;

// Priority encoder: find highest set bit of x_i
always @(*) begin
    casez (x_i)
        16'b1???????????????: k = 4'd15;
        16'b01??????????????: k = 4'd14;
        16'b001?????????????: k = 4'd13;
        16'b0001????????????: k = 4'd12;
        16'b00001???????????: k = 4'd11;
        16'b000001??????????: k = 4'd10;
        16'b0000001?????????: k = 4'd9;
        16'b00000001????????: k = 4'd8;
        16'b000000001???????: k = 4'd7;
        16'b0000000001??????: k = 4'd6;
        16'b00000000001?????: k = 4'd5;
        16'b000000000001????: k = 4'd4;
        16'b0000000000001???: k = 4'd3;
        16'b00000000000001??: k = 4'd2;
        16'b000000000000001?: k = 4'd1;
        default:              k = 4'd0;
    endcase
end

// LUT: y0 = round(256 / (1.5 × 2^(k−8)))
// k=8  → x_float ∈ [1.0,  2.0),  y0 = round(256/1.5)    = 171
// k=9  → x_float ∈ [2.0,  4.0),  y0 = round(256/3.0)    =  85
// k=10 → x_float ∈ [4.0,  8.0),  y0 = round(256/6.0)    =  43
// large y0 for small k (reciprocal of small value = large)
// saturation at k=0,1 prevents 16-bit overflow
always @(*) begin
    case (k)
        4'd15: y0_o = 16'd2;     // x_float ∈ [128, 256),  1/x ≈ 0.0052
        4'd14: y0_o = 16'd3;     // x_float ∈ [ 64, 128),  1/x ≈ 0.0104
        4'd13: y0_o = 16'd5;     // x_float ∈ [ 32,  64),  1/x ≈ 0.0208
        4'd12: y0_o = 16'd11;    // x_float ∈ [ 16,  32),  1/x ≈ 0.0417
        4'd11: y0_o = 16'd21;    // x_float ∈ [  8,  16),  1/x ≈ 0.0833
        4'd10: y0_o = 16'd43;    // x_float ∈ [  4,   8),  1/x ≈ 0.1667
        4'd9:  y0_o = 16'd85;    // x_float ∈ [  2,   4),  1/x ≈ 0.3333
        4'd8:  y0_o = 16'd171;   // x_float ∈ [  1,   2),  1/x ≈ 0.6667
        4'd7:  y0_o = 16'd341;   // x_float ∈ [0.50, 1.0), 1/x ≈ 1.333
        4'd6:  y0_o = 16'd683;   // x_float ∈ [0.25, 0.5), 1/x ≈ 2.667
        4'd5:  y0_o = 16'd1365;  // x_float ∈ [0.125, 0.25), 1/x ≈ 5.333
        4'd4:  y0_o = 16'd2731;  // x_float ∈ [0.0625, 0.125), 1/x ≈ 10.67
        4'd3:  y0_o = 16'd5461;  // x_float ∈ [0.03125, 0.0625), 1/x ≈ 21.33
        4'd2:  y0_o = 16'd10922; // x_float ∈ [0.015625, 0.03125), 1/x ≈ 42.67
        4'd1:  y0_o = 16'd21845; // x_float ∈ [0.0078, 0.015625), 1/x ≈ 85.33
        4'd0:  y0_o = 16'd32767; // x_float < 0.0078, saturate to Q8.8 max
        default: y0_o = 16'd171;
    endcase
end

endmodule
