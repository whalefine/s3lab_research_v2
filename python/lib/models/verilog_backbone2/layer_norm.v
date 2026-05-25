// =============================================================================
// layer_norm.v
//
// Q8.8 Layer Normalization，對齊 run_backbone_numpy.layer_norm()。
//
// 每次處理 ONE TOKEN（FEAT_DIM = 768 維）：
//   mean  = sum(x) × rcp_c     (rcp_c = 1/768，以 85/65536 近似)
//   var   = sum(centered²) × rcp_c
//   y[i]  = w[i] × (centered[i] × inv_std) + b[i]
//
// 介面說明：
//   - start 後外部 controller 連續送 FEAT_DIM cycle 的 x_i（x_valid=1）
//   - feat_addr_o 指出目前需要的 weight/bias 索引（0..FEAT_DIM-1）
//   - y_valid=1 的 FEAT_DIM cycle 輸出 y_o（對齊 feat_addr_o）
//
// 內部 buffer（FEAT_DIM × 16 bit = 1.5 KB）：
//   仿 SRAM reg array，synthesis 可對應 single-port SRAM。
//
// inv_sqrt_nr 使用 4-cycle 子模組（inv_sqrt_nr.v）。
//
// RCP_C 精度：
//   round(2^16 / FEAT_DIM) = round(65536/768) = 85
//   mean_int = (sum_int × 85) >> 16
//   var_int  = (sum_sq × 85) >> 24  (sum_sq is Q16.16 sum of Q8.8 squares)
// =============================================================================

module layer_norm #(
    parameter FEAT_DIM = 768,     // feature (channel) dimension C
    parameter RCP_SHIFT = 16,     // shift for rcp_c: mean = (sum×RCP_NUM)>>RCP_SHIFT
    parameter RCP_NUM   = 85      // numerator: round(2^RCP_SHIFT / FEAT_DIM)
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // Input feature stream
    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    // Weight / bias from external ROM (indexed by feat_addr_o)
    input  wire signed [15:0] w_i,
    input  wire signed [15:0] b_i,

    // Feature address for external ROM
    output wire [9:0]  feat_addr_o,   // 0..FEAT_DIM-1

    // Status
    output wire        busy,
    output reg         done,

    // Output feature stream
    output reg  signed [15:0] y_o,
    output reg         y_valid,
    // Combinational saturated output (valid same cycle as feat_addr_o in S_NORM).
    // Parent must capture y_sat_o on each out_beat_o posedge (y_o is 1 cycle late).
    output wire signed [15:0] y_sat_o,
    // High during S_NORM: y_sat_o and feat_addr_o are aligned (capture on posedge).
    output wire        out_beat_o
);

// FSM state encoding
parameter S_IDLE    = 3'd0;
parameter S_LOAD    = 3'd1;  // load x into feat_buf, accumulate sum
parameter S_MEAN    = 3'd2;  // compute mean (1 cycle)
parameter S_CENTER  = 3'd3;  // centered = feat_buf[i]-mean; accumulate sum_sq
parameter S_VAR     = 3'd4;  // compute var (1 cycle)
parameter S_INV     = 3'd5;  // wait for inv_sqrt_nr
parameter S_NORM    = 3'd6;  // output normalized values
parameter S_DONE    = 3'd7;

// Registers
reg [2:0] state, next_state;
reg [9:0] addr;               // feature index counter

// Feature buffer: 768 × 16 bit
reg signed [15:0] feat_buf [0:FEAT_DIM-1];

// Accumulators
reg signed [31:0] sum_acc;    // 32-bit: sum of Q8.8 integers (max 768×32767≈25M < 2^25)
reg        [47:0] sum_sq_acc; // 48-bit: sum of squares (max 768×32767²≈8e11 < 2^40)
reg signed [15:0] mean_q88;   // computed mean in Q8.8
reg signed [15:0] var_q88;    // computed variance in Q8.8

// inv_sqrt_nr signals
reg  inv_start;
wire inv_busy, inv_done;
wire signed [15:0] inv_std;

// Current centered value (Q8.8 integer domain)
wire signed [16:0] centered17 = $signed({feat_buf[addr][15], feat_buf[addr]}) -
                                  $signed({mean_q88[15], mean_q88});
wire signed [15:0] centered = (centered17[16] ^ centered17[15]) ?
    (centered17[16] ? 16'sh8000 : 16'sh7FFF) : centered17[15:0];

// ---------------------------------------------------------------------------
// Q8.8 helpers — align numpy fp() / write_bi (round + saturate, not truncate)
// ---------------------------------------------------------------------------
function signed [15:0] sat_q88;
    input signed [31:0] v;
    begin
        if (v > 32'sh7FFF)
            sat_q88 = 16'sh7FFF;
        else if (v < -32'sh8000)
            sat_q88 = -16'sh8000;
        else
            sat_q88 = v[15:0];
    end
endfunction

function signed [15:0] rnd_shr16_q88;
    input signed [31:0] v;
    reg signed [31:0] t;
    begin
        t = v + 32'sd32768;
        rnd_shr16_q88 = sat_q88(t >>> 16);
    end
endfunction

function signed [15:0] rnd_q16_to_q8;
    input signed [31:0] v;
    reg signed [31:0] t;
    begin
        t = v + 32'sd128;
        if (t > 32'sh7FFF_FFFF)
            rnd_q16_to_q8 = 16'sh7FFF;
        else if (t < -32'sh8000_0000)
            rnd_q16_to_q8 = -16'sh8000;
        else
            rnd_q16_to_q8 = t[23:8];
    end
endfunction

// Variance from sum_sq_acc: var_int = (sum_sq × RCP_NUM) >> 24
// sum_sq_acc is sum of (centered_int)², each centered_int is Q8.8 integer
// var_float = Σ(centered_float²)/FEAT_DIM
// var_int_q88 = var_float × 256 = Σ(centered_int²) × RCP_NUM >> 24
wire [63:0] var_full = sum_sq_acc * RCP_NUM;
wire [63:0] var_rnd  = var_full + 64'd8388608;
wire signed [15:0] var_q88_comb = (|var_rnd[63:39]) ? 16'sh7FFF : var_rnd[39:24];
// numpy: inv_sqrt(var + 1e-6); Q8.8 min positive step = 1 LSB only when var<=0
wire signed [15:0] var_eps = (var_q88_comb <= 16'sd0) ? 16'sd1 : var_q88_comb;

// y[i] = fp(w[i] * fp(centered[i] * inv_std) + b[i])
wire [31:0] ci_std_raw   = $signed(feat_buf[addr]) * $signed(inv_std);
wire signed [15:0] ci_std = rnd_q16_to_q8(ci_std_raw);
wire [31:0] wci_raw      = $signed(w_i) * ci_std;
wire signed [15:0] wci   = rnd_q16_to_q8(wci_raw);
wire signed [31:0] y_full = $signed(wci) + $signed(b_i);
wire signed [15:0] y_sat  = sat_q88(y_full);
assign y_sat_o   = y_sat;
assign out_beat_o = (state == S_NORM);

// Squared centered value for variance accumulation
wire [31:0] csq = $signed(centered) * $signed(centered);  // (centered_int)²

// Feature address output
assign feat_addr_o = addr;

// inv_sqrt_nr instance
inv_sqrt_nr u_inv_sqrt (
    .clk    (clk),
    .reset  (reset),
    .start  (inv_start),
    .v_i    (var_eps),
    .busy   (inv_busy),
    .done   (inv_done),
    .y_o    (inv_std)
);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:   next_state = start                         ? S_LOAD   : S_IDLE;
        // After FEAT_DIM loads, addr becomes FEAT_DIM (e.g. 32); do not require
        // x_valid that cycle — tb may drop rp_stream same posedge as last feat.
        // Exit to S_MEAN as soon as last sample is taken (addr==FEAT_DIM-1 & x_valid),
        // or legacy catch-up if addr already FEAT_DIM (must not keep addr at FEAT_DIM
        // for combo reads: feat_buf[addr] / feat_addr_o are only [0:FEAT_DIM-1]).
        S_LOAD:   next_state = ((addr == FEAT_DIM-1 && x_valid) || (addr == FEAT_DIM))
                  ? S_MEAN : S_LOAD;
        S_MEAN:   next_state = S_CENTER;
        S_CENTER: next_state = (addr == FEAT_DIM-1)          ? S_VAR    : S_CENTER;
        S_VAR:    next_state = S_INV;
        S_INV:    next_state = inv_done                      ? S_NORM   : S_INV;
        S_NORM:   next_state = (addr == FEAT_DIM-1)          ? S_DONE   : S_NORM;
        S_DONE:   next_state = S_IDLE;
        default:  next_state = S_IDLE;
    endcase
end

// FSM segment 3: output logic and datapath
always @(posedge clk) begin
    done      <= 1'b0;
    y_valid   <= 1'b0;
    inv_start <= 1'b0;
    if (reset) begin
        addr     <= 10'd0;
        sum_acc  <= 32'sd0;
        sum_sq_acc <= 48'd0;
        mean_q88 <= 16'sd0;
        var_q88  <= 16'sd0;
        y_o      <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                addr       <= 10'd0;
                sum_acc    <= 32'sd0;
                sum_sq_acc <= 48'd0;
            end

            S_LOAD: begin
                if (x_valid && (addr < FEAT_DIM)) begin
                    feat_buf[addr] <= x_i;
                    sum_acc   <= sum_acc + $signed(x_i);
                    addr      <= (addr == FEAT_DIM - 1) ? 10'd0 : (addr + 10'd1);
                end
            end

            S_MEAN: begin
                // mean = fp(sum * rcp_c); rounded >>16
                mean_q88 <= rnd_shr16_q88(sum_acc * RCP_NUM);
                addr <= 10'd0;
                sum_sq_acc <= 48'd0;
            end

            S_CENTER: begin
                // centered = sat_q88(x - mean); store for S_NORM (align numpy fp after subtract)
                feat_buf[addr]  <= centered;
                sum_sq_acc <= sum_sq_acc + {16'd0, csq};
                // Do not leave addr==FEAT_DIM: feat_buf[addr] / y_sat are combo and OOB → X
                addr       <= (addr == FEAT_DIM - 1) ? 10'd0 : (addr + 10'd1);
            end

            S_VAR: begin
                // var_q88 registered for inv_sqrt
                var_q88   <= var_eps;
                inv_start <= 1'b1;
                // Reset addr for S_NORM output phase (otherwise addr inherits
                // FEAT_DIM from S_CENTER and S_NORM wraps through 10-bit space,
                // taking ~1024 cycles per token instead of FEAT_DIM).
                addr      <= 10'd0;
            end

            S_INV: begin
                // wait for inv_sqrt_nr; inv_std wire holds result when inv_done
            end

            S_NORM: begin
                // feat_buf[addr] now holds centered value (written during S_CENTER)
                y_o     <= y_sat;
                y_valid <= 1'b1;
                addr    <= (addr == FEAT_DIM - 1) ? 10'd0 : (addr + 10'd1);
            end

            S_DONE: begin
                done <= 1'b1;
                addr <= 10'd0;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
