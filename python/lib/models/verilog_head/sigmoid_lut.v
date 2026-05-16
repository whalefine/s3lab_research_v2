// =============================================================================
// sigmoid_lut.v
//
// LUT + 線性插值 sigmoid，對齊 run_backbone_numpy.sigmoid() + sigmoid_clamped()
//
// 輸入  x_i [15:0]  Q8.8 有號定點（conv5 輸出，已做 fp()）
// 輸出  y_o  [7:0]  Q0.8 無號定點，值域 [1, 255] ↔ [1/256, 255/256]
//
// 硬體資源：65×8 bit ROM + 飽和器 + 加法器 + 8×6 乘法器 + 加法器 + 右移
// 完全組合邏輯；如需流水線請在 Step3→4 或 Step4→5 插入暫存器
// =============================================================================

module sigmoid_lut (
    input  wire signed [15:0] x_i,   // Q8.8 signed
    output wire        [ 7:0] y_o    // Q0.8 unsigned, [1,255]
);

// -----------------------------------------------------------------------------
// ROM function：65 端點，index 0..64，步長 0.25，值 = round(sigmoid×256)
// -----------------------------------------------------------------------------
function automatic [7:0] lut_val;
    input [6:0] addr;
    case (addr)
        7'd 0: lut_val = 8'd  0;  // x=-8.00
        7'd 1: lut_val = 8'd  0;  // x=-7.75
        7'd 2: lut_val = 8'd  0;  // x=-7.50
        7'd 3: lut_val = 8'd  0;  // x=-7.25
        7'd 4: lut_val = 8'd  0;  // x=-7.00
        7'd 5: lut_val = 8'd  0;  // x=-6.75
        7'd 6: lut_val = 8'd  0;  // x=-6.50
        7'd 7: lut_val = 8'd  0;  // x=-6.25
        7'd 8: lut_val = 8'd  1;  // x=-6.00
        7'd 9: lut_val = 8'd  1;  // x=-5.75
        7'd10: lut_val = 8'd  1;  // x=-5.50
        7'd11: lut_val = 8'd  1;  // x=-5.25
        7'd12: lut_val = 8'd  2;  // x=-5.00
        7'd13: lut_val = 8'd  2;  // x=-4.75
        7'd14: lut_val = 8'd  3;  // x=-4.50
        7'd15: lut_val = 8'd  4;  // x=-4.25
        7'd16: lut_val = 8'd  5;  // x=-4.00
        7'd17: lut_val = 8'd  6;  // x=-3.75
        7'd18: lut_val = 8'd  8;  // x=-3.50
        7'd19: lut_val = 8'd 10;  // x=-3.25
        7'd20: lut_val = 8'd 12;  // x=-3.00
        7'd21: lut_val = 8'd 15;  // x=-2.75
        7'd22: lut_val = 8'd 19;  // x=-2.50
        7'd23: lut_val = 8'd 24;  // x=-2.25
        7'd24: lut_val = 8'd 31;  // x=-2.00
        7'd25: lut_val = 8'd 38;  // x=-1.75
        7'd26: lut_val = 8'd 47;  // x=-1.50
        7'd27: lut_val = 8'd 57;  // x=-1.25
        7'd28: lut_val = 8'd 69;  // x=-1.00
        7'd29: lut_val = 8'd 82;  // x=-0.75
        7'd30: lut_val = 8'd 97;  // x=-0.50
        7'd31: lut_val = 8'd112;  // x=-0.25
        7'd32: lut_val = 8'd128;  // x=+0.00
        7'd33: lut_val = 8'd144;  // x=+0.25
        7'd34: lut_val = 8'd159;  // x=+0.50
        7'd35: lut_val = 8'd174;  // x=+0.75
        7'd36: lut_val = 8'd187;  // x=+1.00
        7'd37: lut_val = 8'd199;  // x=+1.25
        7'd38: lut_val = 8'd209;  // x=+1.50
        7'd39: lut_val = 8'd218;  // x=+1.75
        7'd40: lut_val = 8'd225;  // x=+2.00
        7'd41: lut_val = 8'd232;  // x=+2.25
        7'd42: lut_val = 8'd237;  // x=+2.50
        7'd43: lut_val = 8'd241;  // x=+2.75
        7'd44: lut_val = 8'd244;  // x=+3.00
        7'd45: lut_val = 8'd246;  // x=+3.25
        7'd46: lut_val = 8'd248;  // x=+3.50
        7'd47: lut_val = 8'd250;  // x=+3.75
        7'd48: lut_val = 8'd251;  // x=+4.00
        7'd49: lut_val = 8'd252;  // x=+4.25
        7'd50: lut_val = 8'd253;  // x=+4.50
        7'd51: lut_val = 8'd254;  // x=+4.75
        7'd52: lut_val = 8'd254;  // x=+5.00
        7'd53: lut_val = 8'd255;  // x=+5.25
        7'd54: lut_val = 8'd255;  // x=+5.50
        7'd55: lut_val = 8'd255;  // x=+5.75
        7'd56: lut_val = 8'd255;  // x=+6.00
        7'd57: lut_val = 8'd255;  // x=+6.25
        7'd58: lut_val = 8'd255;  // x=+6.50
        7'd59: lut_val = 8'd255;  // x=+6.75
        7'd60: lut_val = 8'd255;  // x=+7.00
        7'd61: lut_val = 8'd255;  // x=+7.25
        7'd62: lut_val = 8'd255;  // x=+7.50
        7'd63: lut_val = 8'd255;  // x=+7.75
        7'd64: lut_val = 8'd255;  // x=+8.00
        default: lut_val = 8'd0;
    endcase
endfunction

// Step 1: 飽和截斷到 [-8, +8]  (-8.0=16'shF800, +8.0=16'sh0800)
wire signed [15:0] x_sat = (x_i < 16'shF800) ? 16'shF800 :
                            (x_i > 16'sh0800) ? 16'sh0800 : x_i;

// Step 2: 平移 → 無號 [0, 4096]
wire [12:0] shifted = x_sat[12:0] + 13'd2048;

// Step 3: index = shifted>>6 ∈ [0,64]; frac6 = shifted&0x3F ∈ [0,63]
wire [6:0] idx_raw = shifted[12:6];
wire [6:0] idx     = (idx_raw > 7'd63) ? 7'd63 : idx_raw;
wire [6:0] hi_idx  = idx + 7'd1;          // 最大 64，合法
wire [5:0] frac6   = shifted[5:0];

// Step 4: 查表（lo / hi 各一次）
wire [7:0] lo = lut_val(idx);
wire [7:0] hi = lut_val(hi_idx);

// Step 5: 線性插值（整數運算）
//
//   result = lo + floor(delta×frac6 / 64)
//        ≡ (lo×64 + delta×frac6) >> 6
//
//   Bit width（worst case）：
//     lo×64          max=255×64=16320   → 14-bit
//     delta×frac6    max=16×63=1008     → 10-bit
//     sum            max=16319 < 2^14   → 14-bit
//     result[13:6]                      → 8-bit
//
wire [7:0]  delta  = hi - lo;                          // ≥ 0（單調）
wire [13:0] prod   = {6'b0, delta} * {8'b0, frac6};  // 最大 16065 < 2^14
wire [13:0] sum    = {lo, 6'b0} + prod;
wire [7:0]  interp = sum[13:6];

// Step 6: clamp 下界 → 1/256（對應 sigmoid_clamped 行為）
assign y_o = (interp == 8'd0) ? 8'd1 : interp;

endmodule
