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
    output reg         y_valid
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

// Current centered value
wire signed [15:0] centered = $signed(feat_buf[addr]) - mean_q88;

// Variance from sum_sq_acc: var_int = (sum_sq × 85) >> 24
// sum_sq_acc is sum of (centered_int)², each centered_int is Q8.8 integer
// var_float = Σ(centered_float²)/768 = Σ(centered_int/256)² / 768 = Σ(centered_int²) × 85 / (256² × 65536)
// var_int_q88 = var_float × 256 = Σ(centered_int²) × 85 / (256 × 65536) = Σ(centered_int²) × 85 >> 24
wire [47:0] var_raw = (sum_sq_acc >> 24) * 85;  // approx: scale then multiply (order matters for precision)
// More precisely: var_q88_comb = (sum_sq_acc * 85) >> 24 — but 48*8=56 bit multiply
wire [55:0] var_full = sum_sq_acc * 8'd85;
wire signed [15:0] var_q88_comb = (|var_full[55:31]) ? 16'sh7FFF : var_full[31:16];
// Add eps=1 (1/256 ≈ 1e-3 >> actual eps 1e-6, but Q8.8 granularity limits us to 1/256)
wire signed [15:0] var_eps = var_q88_comb + 16'sd1;

// NR current neuron output normalization
// y[i] = w[i] × (centered × inv_std) + b[i]   all Q8.8
wire [31:0] ci_std_raw   = $signed(feat_buf[addr]) * $signed(inv_std);  // centered × inv_std, Q16.16
wire signed [15:0] ci_std = ci_std_raw[23:8];                       // Q8.8
wire [31:0] wci_raw       = $signed(w_i) * ci_std;                  // w × (c×inv_std), Q16.16
wire signed [15:0] wci    = wci_raw[23:8];                          // Q8.8
wire signed [31:0] y_full = $signed(wci) + $signed(b_i);
wire signed [15:0] y_sat  = (y_full > 32'sh7FFF) ? 16'sh7FFF :
                             (y_full < -32'sh8000) ? -16'sh8000 :
                             y_full[15:0];

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
        S_LOAD:   next_state = (addr == FEAT_DIM-1 && x_valid) ? S_MEAN : S_LOAD;
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
        state    <= S_IDLE;
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
                if (x_valid) begin
                    feat_buf[addr] <= x_i;
                    sum_acc   <= sum_acc + $signed(x_i);
                    addr      <= addr + 10'd1;
                end
            end

            S_MEAN: begin
                // mean_int = (sum_int × 85) >> 16
                mean_q88 <= (sum_acc * RCP_NUM) >>> RCP_SHIFT;
                addr <= 10'd0;
                sum_sq_acc <= 48'd0;
            end

            S_CENTER: begin
                // centered = feat_buf[addr] - mean; store back; accumulate squared
                feat_buf[addr]  <= centered;
                sum_sq_acc <= sum_sq_acc + {16'd0, csq};
                addr       <= addr + 10'd1;
            end

            S_VAR: begin
                // var_q88 registered for inv_sqrt
                var_q88   <= var_eps;
                inv_start <= 1'b1;
            end

            S_INV: begin
                // wait for inv_sqrt_nr; inv_std wire holds result when inv_done
            end

            S_NORM: begin
                // feat_buf[addr] now holds centered value (written during S_CENTER)
                y_o     <= y_sat;
                y_valid <= 1'b1;
                addr    <= addr + 10'd1;
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
