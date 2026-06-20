// =============================================================================
// inv_sqrt_lut_seed.v   (verilog_backbone4, Q7.7-native)
//
// Leading-bit LUT: initial estimate of 1/sqrt(v) for Q7.7 positive v.
//
// 輸入  v_i [15:0]  Q7.7 unsigned（LayerNorm variance，≥ 0）
// 輸出  y0_o [15:0] Q7.7 初始估值，≈ 1/sqrt(v_i/128)
//
// 方法：
//   找 v_i 最高有效位位置 k（priority encoder），
//   查 16-entry LUT 回傳 1/sqrt(幾何中點)，即
//   y0 = round( 128 / sqrt(1.5 × 2^(k−7)) )
//
// 與 Q8.8 版差異：scale 256->128（y0 值約半），且 bin 邊界由 2^(k-8) 改為
//   2^(k-7)（小數點位移一位）。完全組合邏輯，無 clk。
// =============================================================================

module inv_sqrt_lut_seed (
    input  wire [15:0] v_i,   // Q7.7 unsigned positive
    output reg  [15:0] y0_o   // Q7.7 initial estimate
);

// Leading-bit position encoder
reg [3:0] k;

// Priority encoder: find highest set bit of v_i
always @(*) begin
    casez (v_i)
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

// LUT: y0 = round(128 / sqrt(1.5 × 2^(k−7)))   (Q7.7)
// k=7  → v_float ∈ [1.0,  2.0),  y0 = round(128/sqrt(1.5)) = 105
// k=8  → v_float ∈ [2.0,  4.0),  y0 = round(128/sqrt(3.0)) = 74
// ...
always @(*) begin
    case (k)
        4'd15: y0_o = 16'd7;     // v_float ∈ [256, 512)
        4'd14: y0_o = 16'd9;     // v_float ∈ [128, 256)
        4'd13: y0_o = 16'd13;    // v_float ∈ [ 64, 128)
        4'd12: y0_o = 16'd18;    // v_float ∈ [ 32,  64)
        4'd11: y0_o = 16'd26;    // v_float ∈ [ 16,  32)
        4'd10: y0_o = 16'd37;    // v_float ∈ [  8,  16)
        4'd9:  y0_o = 16'd52;    // v_float ∈ [  4,   8)
        4'd8:  y0_o = 16'd74;    // v_float ∈ [  2,   4)
        4'd7:  y0_o = 16'd105;   // v_float ∈ [  1,   2)
        4'd6:  y0_o = 16'd148;   // v_float ∈ [0.50, 1.0)
        4'd5:  y0_o = 16'd209;   // v_float ∈ [0.25, 0.5)
        4'd4:  y0_o = 16'd296;   // v_float ∈ [0.125, 0.25)
        4'd3:  y0_o = 16'd418;   // v_float ∈ [0.0625, 0.125)
        4'd2:  y0_o = 16'd591;   // v_float ∈ [0.03125, 0.0625)
        4'd1:  y0_o = 16'd836;   // v_float ∈ [0.015625, 0.03125)
        4'd0:  y0_o = 16'd1182;  // v_float ∈ [0, 0.015625)
        default: y0_o = 16'd105;
    endcase
end

endmodule
