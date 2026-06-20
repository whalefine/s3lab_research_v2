// =============================================================================
// layer_norm_pip.v
//
// Q8.8 LayerNorm (FEAT_DIM=32 default), bit-accurate vs
// run_backbone_numpy_shared_trunk.py layer_norm() and verilog_backbone2/layer_norm.v.
//
// Memory contract: CLK(clk) posedge registered-data (addr@T -> Q@T+1).
//   The pend/wait (S_LOAD) and 3-phase (S_NORM) windows already give the +1
//   alignment, so no FSM retime is needed when the parent macro moves ~clk->clk.
//
// Per token:
//   S_LOAD       : SRAM read handshake (x_rd_en @ T -> pend -> wait -> capture x_i)
//                  sized for CLK(clk) 1-cycle Q latency (addr@T -> Q@T+1)
//   S_MEAN       : mean_q88 = sat((sum_acc * RCP + 32768) >> 16)
//   S_CENTER     : feat_buf <- sat(x - mean); sum_sq_acc += centered^2
//   S_VAR        : var_eps = max(sat((sum_sq*RCP+2^23)>>24), 1)
//   S_INV        : inv_sqrt_nr(var_eps) -> inv_std
//   S_NORM       : 3-phase per feature (breaks ROM Q -> multiplier STA path):
//                  P0: feat_addr_o=norm_idx; ci_std_r <= rnd(feat_buf[i]*inv_std)
//                  P1: feat_addr_o=norm_idx_out; w_r<=w_i; b_r<=b_i (ROM Q@P0 valid)
//                  P2: y <= sat(rnd(w_r*ci_std_r)+b_r); y_valid
//
// Golden: Activation/backbone_blocks_<b>_after_norm1_out_bi.txt
// Golden-Weight: Weight/backbone_blocks_<b>_norm1_{weight,bias}_bi.txt
//
// Saturation / rounding via wire only (no function). Verilog-2001 synthesizable.
// =============================================================================

module layer_norm_pip #(
    parameter FEAT_DIM  = 32,
    parameter FEAT_AW   = 5,
    parameter RCP_SHIFT = 16,
    parameter RCP_NUM   = 2048
) (
    clk,
    reset,
    start,
    token_base_flat,
    x_rd_en,
    x_rd_flat,
    x_i,
    w_i,
    b_i,
    feat_addr_o,
    busy,
    done,
    y_o,
    y_valid,
    x_rd_pend_o,
    x_rd_wait_o
);

input                       clk;
input                       reset;
input                       start;
input  [13:0]               token_base_flat;
output reg                  x_rd_en;
output reg [13:0]           x_rd_flat;
input  signed [15:0]        x_i;
input  signed [15:0]        w_i;
input  signed [15:0]        b_i;
output [9:0]                feat_addr_o;
output                      busy;
output reg                  done;
output reg signed [15:0]    y_o;
output reg                  y_valid;
output                      x_rd_pend_o;
output                      x_rd_wait_o;

localparam [FEAT_AW-1:0] FEAT_LAST = FEAT_DIM - 1;

parameter S_IDLE      = 4'd0;
parameter S_LOAD      = 4'd1;
parameter S_MEAN      = 4'd2;
parameter S_CENTER    = 4'd3;
parameter S_VAR       = 4'd4;
parameter S_INV       = 4'd5;
parameter S_NORM      = 4'd6;
parameter S_DONE      = 4'd7;

reg [3:0]                   state;
reg [3:0]                   next_state;
reg [FEAT_AW-1:0]           addr;
reg [FEAT_AW-1:0]           norm_idx;
reg [FEAT_AW-1:0]           norm_idx_out;
reg [1:0]                   norm_phase;

reg signed [15:0]           ci_std_r;
reg signed [15:0]           w_r;
reg signed [15:0]           b_r;

reg signed [15:0]           feat_buf [0:FEAT_DIM-1];

reg signed [31:0]           sum_acc;
reg [47:0]                  sum_sq_acc;
reg signed [15:0]           mean_q88;
reg signed [15:0]           var_q88;

reg                         x_rd_pend;
reg                         x_rd_wait;
reg                         inv_start;

assign x_rd_pend_o = x_rd_pend;
assign x_rd_wait_o = x_rd_wait;

wire                        inv_busy;
wire                        inv_done;
wire signed [15:0]          inv_std;

wire signed [16:0]          centered17;
wire signed [15:0]          centered;
wire [63:0]                 var_full;
wire [63:0]                 var_rnd;
wire signed [15:0]          var_q88_comb;
wire signed [15:0]          var_eps;
wire [31:0]                 ci_std_raw;
wire signed [15:0]          ci_std;
wire [31:0]                 csq;

wire signed [31:0]          mean_prod;
wire signed [31:0]          mean_rnd_t;
wire signed [31:0]          mean_shr;
wire signed [15:0]          mean_q88_comb;

wire signed [31:0]          ci_std_t;

wire signed [31:0]          wci_raw_reg;
wire signed [31:0]          wci_t_reg;
wire signed [15:0]          wci_reg;
wire signed [31:0]          y_full_reg;
wire signed [15:0]          y_sat_reg;

wire [FEAT_AW-1:0]          op_idx;

wire                        load_x_last_beat;
wire                        center_last_beat;
wire                        norm_last_beat;

assign op_idx = (state == S_NORM) ? norm_idx : addr;

// P0: ROM addr = norm_idx; P1/P2: norm_idx_out (aligned with latched w/b)
assign feat_addr_o = (state == S_NORM && (norm_phase == 2'd1 || norm_phase == 2'd2)) ?
    {{(10-FEAT_AW){1'b0}}, norm_idx_out} :
    {{(10-FEAT_AW){1'b0}}, op_idx};

assign load_x_last_beat  = (state == S_LOAD) && x_rd_wait && (addr == FEAT_LAST);
assign center_last_beat  = (state == S_CENTER) && (addr == FEAT_LAST);
assign norm_last_beat    = (state == S_NORM) && (norm_phase == 2'd2) &&
                           (norm_idx == FEAT_LAST);

assign centered17 = $signed({feat_buf[op_idx][15], feat_buf[op_idx]}) -
                    $signed({mean_q88[15], mean_q88});
assign centered = (centered17[16] ^ centered17[15]) ?
    (centered17[16] ? 16'sh8000 : 16'sh7FFF) : centered17[15:0];

assign var_full     = sum_sq_acc * RCP_NUM;
assign var_rnd      = var_full + 64'd8388608;
assign var_q88_comb = (|var_rnd[63:39]) ? 16'sh7FFF : var_rnd[39:24];
assign var_eps      = (var_q88_comb <= 16'sd0) ? 16'sd1 : var_q88_comb;

assign mean_prod    = sum_acc * RCP_NUM;
assign mean_rnd_t   = mean_prod + 32'sd32768;
assign mean_shr     = mean_rnd_t >>> RCP_SHIFT;
assign mean_q88_comb =
    (mean_shr > 32'sh7FFF) ? 16'sh7FFF :
    (mean_shr < -32'sh8000) ? -16'sh8000 : mean_shr[15:0];

assign ci_std_raw = $signed(feat_buf[op_idx]) * $signed(inv_std);
assign ci_std_t   = ci_std_raw + 32'sd128;
assign ci_std     =
    (ci_std_t > 32'sh7FFF_FFFF) ? 16'sh7FFF :
    (ci_std_t < -32'sh8000_0000) ? -16'sh8000 : ci_std_t[23:8];

assign wci_raw_reg = $signed(w_r) * $signed(ci_std_r);
assign wci_t_reg   = wci_raw_reg + 32'sd128;
assign wci_reg     =
    (wci_t_reg > 32'sh7FFF_FFFF) ? 16'sh7FFF :
    (wci_t_reg < -32'sh8000_0000) ? -16'sh8000 : wci_t_reg[23:8];

assign y_full_reg = $signed(wci_reg) + $signed(b_r);
assign y_sat_reg  =
    (y_full_reg > 32'sh7FFF) ? 16'sh7FFF :
    (y_full_reg < -32'sh8000) ? -16'sh8000 : y_full_reg[15:0];

assign csq = $signed(centered) * $signed(centered);

assign busy = (state != S_IDLE);

inv_sqrt_nr u_inv_sqrt (
    .clk   (clk),
    .reset (reset),
    .start (inv_start),
    .v_i   (var_eps),
    .busy  (inv_busy),
    .done  (inv_done),
    .y_o   (inv_std)
);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE: begin
            if (start)
                next_state = S_LOAD;
        end
        S_LOAD: begin
            if (load_x_last_beat)
                next_state = S_MEAN;
        end
        S_MEAN: begin
            next_state = S_CENTER;
        end
        S_CENTER: begin
            if (center_last_beat)
                next_state = S_VAR;
        end
        S_VAR: begin
            next_state = S_INV;
        end
        S_INV: begin
            if (inv_done)
                next_state = S_NORM;
        end
        S_NORM: begin
            if (norm_last_beat)
                next_state = S_DONE;
        end
        S_DONE: begin
            next_state = S_IDLE;
        end
        default: next_state = S_IDLE;
    endcase
end

// x SRAM read pipeline + datapath
always @(posedge clk) begin
    done      <= 1'b0;
    y_valid   <= 1'b0;
    inv_start <= 1'b0;
    x_rd_en   <= 1'b0;

    if (reset) begin
        addr       <= {FEAT_AW{1'b0}};
        norm_idx     <= {FEAT_AW{1'b0}};
        norm_idx_out <= {FEAT_AW{1'b0}};
        norm_phase   <= 2'd0;
        ci_std_r     <= 16'sd0;
        w_r          <= 16'sd0;
        b_r          <= 16'sd0;
        sum_acc    <= 32'sd0;
        sum_sq_acc <= 48'd0;
        mean_q88   <= 16'sd0;
        var_q88    <= 16'sd0;
        x_rd_pend  <= 1'b0;
        x_rd_wait  <= 1'b0;
        y_o        <= 16'sd0;
        x_rd_flat  <= 14'd0;
    end else begin
        case (state)
            S_IDLE: begin
                addr         <= {FEAT_AW{1'b0}};
                norm_idx     <= {FEAT_AW{1'b0}};
                norm_idx_out <= {FEAT_AW{1'b0}};
                norm_phase   <= 2'd0;
                sum_acc      <= 32'sd0;
                sum_sq_acc   <= 48'd0;
                x_rd_pend    <= 1'b0;
                x_rd_wait    <= 1'b0;
            end

            S_LOAD: begin
                if (!x_rd_pend && !x_rd_wait) begin
                    x_rd_en   <= 1'b1;
                    x_rd_flat <= token_base_flat + {{(14-FEAT_AW){1'b0}}, addr};
                    x_rd_pend <= 1'b1;
                end else if (x_rd_pend && !x_rd_wait) begin
                    x_rd_wait <= 1'b1;
                end else begin
                    feat_buf[addr] <= x_i;
                    sum_acc        <= sum_acc + $signed(x_i);
                    x_rd_wait      <= 1'b0;
                    x_rd_pend      <= 1'b0;
                    if (addr != FEAT_LAST)
                        addr <= addr + {{(FEAT_AW-1){1'b0}}, 1'b1};
                    else
                        addr <= {FEAT_AW{1'b0}};
                end
            end

            S_MEAN: begin
                mean_q88   <= mean_q88_comb;
                addr       <= {FEAT_AW{1'b0}};
                sum_sq_acc <= 48'd0;
            end

            S_CENTER: begin
                feat_buf[addr] <= centered;
                sum_sq_acc     <= sum_sq_acc + {16'd0, csq};
                if (addr != FEAT_LAST)
                    addr <= addr + {{(FEAT_AW-1){1'b0}}, 1'b1};
                else
                    addr <= {FEAT_AW{1'b0}};
            end

            S_VAR: begin
                var_q88   <= var_eps;
                inv_start <= 1'b1;
                addr      <= {FEAT_AW{1'b0}};
            end

            S_INV: begin
                if (inv_done) begin
                    norm_idx     <= {FEAT_AW{1'b0}};
                    norm_idx_out <= {FEAT_AW{1'b0}};
                    norm_phase   <= 2'd0;
                end
            end

            S_NORM: begin
                if (norm_phase == 2'd0) begin
                    ci_std_r     <= ci_std;
                    norm_idx_out <= norm_idx;
                    norm_phase   <= 2'd1;
                end else if (norm_phase == 2'd1) begin
                    w_r        <= w_i;
                    b_r        <= b_i;
                    norm_phase <= 2'd2;
                end else begin
                    y_o        <= y_sat_reg;
                    y_valid    <= 1'b1;
                    norm_phase <= 2'd0;
                    if (norm_idx != FEAT_LAST)
                        norm_idx <= norm_idx + {{(FEAT_AW-1){1'b0}}, 1'b1};
                end
            end

            S_DONE: begin
                done <= 1'b1;
                addr <= {FEAT_AW{1'b0}};
            end

            default: ;
        endcase
    end
end

endmodule
