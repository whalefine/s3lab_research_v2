// =============================================================================
// sigmoid_lut.v -- Q8.8 sigmoid_clamped (65-entry LUT + linear interp, 1-cycle)
// -----------------------------------------------------------------------------
// numpy: sigmoid_clamped() in run_backbone_numpy_shared_trunk.py
//   posedge T   : in_valid + in_q88
//   posedge T+1 : out_valid, out_q88
// =============================================================================

module sigmoid_lut (
    clk       ,
    rst_n     ,
    in_valid  ,
    in_q88    ,
    out_valid ,
    out_q88
);

parameter DATA_W = 16 ;

input                       clk       ;
input                       rst_n     ;
input                       in_valid  ;
input  signed [DATA_W-1:0]  in_q88    ;
output                      out_valid ;
output signed [DATA_W-1:0]  out_q88   ;

reg  signed [DATA_W-1:0]  in_clip ;
reg         [12:0]        shifted ;
reg         [5:0]         idx6 ;
reg         [5:0]         frac6 ;
reg         [9:0]         lo_u ;
reg         [9:0]         hi_u ;
reg  signed [10:0]        delta ;
reg  signed [15:0]        scaled_lo ;
reg  signed [15:0]        delta_frac ;
reg  signed [15:0]        raw_sum ;
reg  signed [15:0]        shifted_back ;
reg  signed [DATA_W-1:0]  clamped ;

reg  signed [DATA_W-1:0]  out_q88_r ;
reg                       out_valid_r ;

assign out_valid = out_valid_r ;
assign out_q88   = out_q88_r ;

// clip + LUT case + linear interp + clamp (combinational, same as original)
always @(*) begin
    if (in_q88 >  16'sd2048)
        in_clip = 16'sd2048 ;
    else if (in_q88 < -16'sd2048)
        in_clip = -16'sd2048 ;
    else
        in_clip = in_q88 ;

    shifted = in_clip[12:0] + 13'd2048 ;
    if (shifted[12])
        idx6 = 6'd63 ;
    else
        idx6 = shifted[11:6] ;
    frac6 = shifted[5:0] ;

    case (idx6)
        6'd00 : lo_u = 10'd0   ;
        6'd01 : lo_u = 10'd0   ;
        6'd02 : lo_u = 10'd0   ;
        6'd03 : lo_u = 10'd0   ;
        6'd04 : lo_u = 10'd0   ;
        6'd05 : lo_u = 10'd0   ;
        6'd06 : lo_u = 10'd0   ;
        6'd07 : lo_u = 10'd0   ;
        6'd08 : lo_u = 10'd1   ;
        6'd09 : lo_u = 10'd1   ;
        6'd10 : lo_u = 10'd1   ;
        6'd11 : lo_u = 10'd1   ;
        6'd12 : lo_u = 10'd2   ;
        6'd13 : lo_u = 10'd2   ;
        6'd14 : lo_u = 10'd3   ;
        6'd15 : lo_u = 10'd4   ;
        6'd16 : lo_u = 10'd5   ;
        6'd17 : lo_u = 10'd6   ;
        6'd18 : lo_u = 10'd8   ;
        6'd19 : lo_u = 10'd10  ;
        6'd20 : lo_u = 10'd12  ;
        6'd21 : lo_u = 10'd15  ;
        6'd22 : lo_u = 10'd19  ;
        6'd23 : lo_u = 10'd24  ;
        6'd24 : lo_u = 10'd31  ;
        6'd25 : lo_u = 10'd38  ;
        6'd26 : lo_u = 10'd47  ;
        6'd27 : lo_u = 10'd57  ;
        6'd28 : lo_u = 10'd69  ;
        6'd29 : lo_u = 10'd82  ;
        6'd30 : lo_u = 10'd97  ;
        6'd31 : lo_u = 10'd112 ;
        6'd32 : lo_u = 10'd128 ;
        6'd33 : lo_u = 10'd144 ;
        6'd34 : lo_u = 10'd159 ;
        6'd35 : lo_u = 10'd174 ;
        6'd36 : lo_u = 10'd187 ;
        6'd37 : lo_u = 10'd199 ;
        6'd38 : lo_u = 10'd209 ;
        6'd39 : lo_u = 10'd218 ;
        6'd40 : lo_u = 10'd225 ;
        6'd41 : lo_u = 10'd232 ;
        6'd42 : lo_u = 10'd237 ;
        6'd43 : lo_u = 10'd241 ;
        6'd44 : lo_u = 10'd244 ;
        6'd45 : lo_u = 10'd246 ;
        6'd46 : lo_u = 10'd248 ;
        6'd47 : lo_u = 10'd250 ;
        6'd48 : lo_u = 10'd251 ;
        6'd49 : lo_u = 10'd252 ;
        6'd50 : lo_u = 10'd253 ;
        6'd51 : lo_u = 10'd254 ;
        6'd52 : lo_u = 10'd254 ;
        6'd53 : lo_u = 10'd255 ;
        6'd54 : lo_u = 10'd255 ;
        6'd55 : lo_u = 10'd255 ;
        6'd56 : lo_u = 10'd255 ;
        6'd57 : lo_u = 10'd256 ;
        6'd58 : lo_u = 10'd256 ;
        6'd59 : lo_u = 10'd256 ;
        6'd60 : lo_u = 10'd256 ;
        6'd61 : lo_u = 10'd256 ;
        6'd62 : lo_u = 10'd256 ;
        6'd63 : lo_u = 10'd256 ;
        default: lo_u = 10'd0 ;
    endcase

    case (idx6)
        6'd00 : hi_u = 10'd0   ;
        6'd01 : hi_u = 10'd0   ;
        6'd02 : hi_u = 10'd0   ;
        6'd03 : hi_u = 10'd0   ;
        6'd04 : hi_u = 10'd0   ;
        6'd05 : hi_u = 10'd0   ;
        6'd06 : hi_u = 10'd0   ;
        6'd07 : hi_u = 10'd1   ;
        6'd08 : hi_u = 10'd1   ;
        6'd09 : hi_u = 10'd1   ;
        6'd10 : hi_u = 10'd1   ;
        6'd11 : hi_u = 10'd2   ;
        6'd12 : hi_u = 10'd2   ;
        6'd13 : hi_u = 10'd3   ;
        6'd14 : hi_u = 10'd4   ;
        6'd15 : hi_u = 10'd5   ;
        6'd16 : hi_u = 10'd6   ;
        6'd17 : hi_u = 10'd8   ;
        6'd18 : hi_u = 10'd10  ;
        6'd19 : hi_u = 10'd12  ;
        6'd20 : hi_u = 10'd15  ;
        6'd21 : hi_u = 10'd19  ;
        6'd22 : hi_u = 10'd24  ;
        6'd23 : hi_u = 10'd31  ;
        6'd24 : hi_u = 10'd38  ;
        6'd25 : hi_u = 10'd47  ;
        6'd26 : hi_u = 10'd57  ;
        6'd27 : hi_u = 10'd69  ;
        6'd28 : hi_u = 10'd82  ;
        6'd29 : hi_u = 10'd97  ;
        6'd30 : hi_u = 10'd112 ;
        6'd31 : hi_u = 10'd128 ;
        6'd32 : hi_u = 10'd144 ;
        6'd33 : hi_u = 10'd159 ;
        6'd34 : hi_u = 10'd174 ;
        6'd35 : hi_u = 10'd187 ;
        6'd36 : hi_u = 10'd199 ;
        6'd37 : hi_u = 10'd209 ;
        6'd38 : hi_u = 10'd218 ;
        6'd39 : hi_u = 10'd225 ;
        6'd40 : hi_u = 10'd232 ;
        6'd41 : hi_u = 10'd237 ;
        6'd42 : hi_u = 10'd241 ;
        6'd43 : hi_u = 10'd244 ;
        6'd44 : hi_u = 10'd246 ;
        6'd45 : hi_u = 10'd248 ;
        6'd46 : hi_u = 10'd250 ;
        6'd47 : hi_u = 10'd251 ;
        6'd48 : hi_u = 10'd252 ;
        6'd49 : hi_u = 10'd253 ;
        6'd50 : hi_u = 10'd254 ;
        6'd51 : hi_u = 10'd254 ;
        6'd52 : hi_u = 10'd255 ;
        6'd53 : hi_u = 10'd255 ;
        6'd54 : hi_u = 10'd255 ;
        6'd55 : hi_u = 10'd255 ;
        6'd56 : hi_u = 10'd256 ;
        6'd57 : hi_u = 10'd256 ;
        6'd58 : hi_u = 10'd256 ;
        6'd59 : hi_u = 10'd256 ;
        6'd60 : hi_u = 10'd256 ;
        6'd61 : hi_u = 10'd256 ;
        6'd62 : hi_u = 10'd256 ;
        6'd63 : hi_u = 10'd256 ;
        default: hi_u = 10'd0 ;
    endcase

    delta        = $signed({1'b0, hi_u}) - $signed({1'b0, lo_u}) ;
    scaled_lo    = $signed({6'b000000, lo_u}) <<< 6 ;
    delta_frac   = delta * $signed({1'b0, frac6}) ;
    raw_sum      = scaled_lo + delta_frac ;
    shifted_back = raw_sum >>> 6 ;

    if (shifted_back > 16'sd255)
        clamped = 16'sd255 ;
    else if (shifted_back < 16'sd1)
        clamped = 16'sd1 ;
    else
        clamped = shifted_back ;
end

// out_valid_r, out_q88_r
always @(posedge clk) begin
    if (!rst_n) begin
        out_valid_r <= 1'b0 ;
        out_q88_r   <= 16'sd0 ;
    end else begin
        out_valid_r <= in_valid ;
        if (in_valid)
            out_q88_r <= clamped ;
    end
end

endmodule
