// =============================================================================
// recip_nr.v
//
// Q8.8 reciprocal via Newton-Raphson (1 iteration) -- bit-accurate mirror of
// numpy _recip_nr_q88_fixed in run_backbone_numpy_shared_trunk.py:
//
//   y = _RECIP_LUT_Y0[msb(x)]            (16-entry seed)
//   repeat 1 time:
//       coeff = 512 - trunc_q88_slice32(x * y)   // truncate, NOT round
//       y     = trunc_q88_slice32(y * coeff)
//       y     = clip(y, -32768, 32767)
//
//   trunc_q88_slice32(v) = signed 16-bit interpretation of v[23:8]
//                          (sign-extended; matches Python (v>>8)&0xFFFF + 2s-comp)
//
// FSM (2 cycles after start):
//   S_IDLE -> S_ITER1 -> S_DONE -> S_IDLE
//
// Caller must clamp x_i >= 1 (numpy uses np.maximum(qkm_eps, 1)).
//
// Used by care_attention.v in S_Z_RECIP to compute z_recip = 1/qk_mean.
// =============================================================================

module recip_nr (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire signed [15:0] x_i,          // Q8.8 (>=1, caller-clamped)
    output wire        busy,
    output reg         done,
    output reg  signed [15:0] y_o           // Q8.8 reciprocal
);

// FSM states (2-bit): 1 NR iteration only (matches _RECIP_NR_ITERS=1)
parameter S_IDLE  = 2'd0;
parameter S_ITER1 = 2'd1;
parameter S_DONE  = 2'd2;

reg [1:0] state, next_state;

// Registered datapath
reg signed [15:0] x_reg;
reg signed [15:0] y_reg;

// Seed
wire signed [15:0] seed_y0;

recip_lut_seed u_seed (
    .x_i (x_i),
    .y0_o(seed_y0),
    .k_o ()
);

// NR combinational chain (matches numpy):
//   xy_raw = x * y  (signed 32-bit)
//   xy_slice = xy_raw[23:8] as signed 16-bit (truncate, no round)
//   coeff = 512 - xy_slice  (signed 17-bit; needs wider than 16 since 33280 > 32767)
//   ynew_raw = y * coeff  (signed 33-bit; widen to be safe)
//   ynew_slice = ynew_raw[23:8] as signed 16-bit (truncate)
//   y <= sat16(ynew_slice)
wire signed [31:0] xy_raw       = $signed(x_reg) * $signed(y_reg);
wire signed [15:0] xy_slice     = xy_raw[23:8];
wire signed [17:0] coeff        = $signed({{2{1'b0}}, 16'sd512}) - $signed({{2{xy_slice[15]}}, xy_slice});
wire signed [33:0] ynew_raw     = $signed({{2{y_reg[15]}}, y_reg}) * $signed(coeff);
wire signed [15:0] ynew_slice   = ynew_raw[23:8];

// sat16 (numpy: np.clip(y, -32768, 32767)); wire only (no function)
wire signed [16:0] y_next_ext = {ynew_slice[15], ynew_slice};
wire signed [15:0] y_next =
    (y_next_ext > 17'sd32767) ? 16'sh7FFF :
    (y_next_ext < -17'sd32768) ? 16'sh8000 : ynew_slice;

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:  next_state = start ? S_ITER1 : S_IDLE;
        S_ITER1: next_state = S_DONE;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3: datapath
always @(posedge clk) begin
    done <= 1'b0;
    if (reset) begin
        x_reg <= 16'sd0;
        y_reg <= 16'sd0;
        y_o   <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                if (start) begin
                    x_reg <= x_i;
                    y_reg <= seed_y0;
                end
            end
            S_ITER1: begin
                y_reg <= y_next;
            end
            S_DONE: begin
                y_o  <= y_reg;
                done <= 1'b1;
            end
            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
