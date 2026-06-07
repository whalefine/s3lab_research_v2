// =============================================================================
// attn_preprocess.v
//
// CARE attention preprocess: S_SPLIT + S_K_MEAN + S_QK_MEAN.
// Bit-accurate vs run_backbone_numpy_shared_trunk.py and verilog_backbone2/care_attention.v.
//
// S_SPLIT : in-place q/k relu6(rnd_shr8(x*S_Q88)) on Sram_q / Sram_k
// S_K_MEAN: mean over tokens of k -> km_buf[NUM_HEADS*HEAD_DIM]
// S_QK_MEAN: sum_d q*km -> Sram_qkm (H,N), qkm = max(sat16((acc+128)>>8), 1)
//
// SRAM read: 1-cycle latency (USE beat consumes sram_*_q_i from prior ADDR beat).
// Flat (H,N,d): ptr = h*(N_TOKENS*HEAD_DIM) + n*HEAD_DIM + d
// =============================================================================

module attn_preprocess #(
    parameter NUM_HEADS   = 4,
    parameter HEAD_DIM    = 8,
    parameter N_TOKENS    = 320,
    parameter S_Q88       = 152,
    parameter RELU6_MAX   = 1536,
    parameter RCP_N_NUM     = 205,
    parameter RCP_N_SHIFT   = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output wire        busy,
    output reg         done,

    output reg         sram_q_ceb_o,
    output reg         sram_q_web_o,
    output reg [13:0]  sram_q_addr_o,
    output reg [15:0]  sram_q_din_o,
    input  wire [15:0] sram_q_q_i,

    output reg         sram_k_ceb_o,
    output reg         sram_k_web_o,
    output reg [13:0]  sram_k_addr_o,
    output reg [15:0]  sram_k_din_o,
    input  wire [15:0] sram_k_q_i,

    output reg         sram_qkm_ceb_o,
    output reg         sram_qkm_web_o,
    output reg [13:0]  sram_qkm_addr_o,
    output reg [15:0]  sram_qkm_din_o
);

localparam HD_ELEMS  = NUM_HEADS * N_TOKENS * HEAD_DIM;
localparam KM_ELEMS  = NUM_HEADS * HEAD_DIM;
localparam QKM_ELEMS = NUM_HEADS * N_TOKENS;

parameter S_IDLE    = 4'd0;
parameter S_SPLIT   = 4'd1;
parameter S_K_MEAN  = 4'd2;
parameter S_QK_MEAN = 4'd3;
parameter S_DONE_ST = 4'd4;

reg [3:0] state;
reg [3:0] next_state;

// S_SPLIT
reg [13:0] sp_ptr;
reg        sp_phase;
reg signed [15:0] sp_q_r;
reg signed [15:0] sp_k_r;

// S_K_MEAN
reg [4:0]         km_oidx;
reg [8:0]         km_n;
reg signed [32:0] km_acc;
reg               km_phase;

reg signed [15:0] km_buf [0:KM_ELEMS-1];

`ifndef SYNTHESIS
integer km_ii;
initial begin
    for (km_ii = 0; km_ii < KM_ELEMS; km_ii = km_ii + 1)
        km_buf[km_ii] = 16'sd0;
end
`endif

// S_QK_MEAN
reg [10:0]        qk_oidx;
reg [1:0]         qk_h_reg;
reg [8:0]         qk_n_reg;
reg [2:0]         qk_d;
reg signed [32:0] qk_acc;
reg               qk_phase;

wire signed [15:0] sp_q_op;
wire signed [15:0] sp_k_op;

wire signed [31:0] sp_q_prod;
wire signed [31:0] sp_q_rnd_t;
wire signed [31:0] sp_q_shr;
wire signed [15:0] sp_q_sat;
wire signed [15:0] sp_q_relu;

wire signed [31:0] sp_k_prod;
wire signed [31:0] sp_k_rnd_t;
wire signed [31:0] sp_k_shr;
wire signed [15:0] sp_k_sat;
wire signed [15:0] sp_k_relu;

wire [1:0]        km_h;
wire [2:0]        km_d;
wire [13:0]       km_k_flat;
wire signed [15:0] km_k_op;
wire signed [32:0] km_acc_next;
wire signed [47:0] km_scaled;
wire signed [47:0] km_shr_w;
wire signed [15:0] km_wr_val;

wire [13:0]       qk_q_flat;
wire [4:0]        qk_km_flat;
wire signed [15:0] qk_q_op;
wire signed [31:0] qk_term;
wire signed [32:0] qk_acc_next;
wire signed [32:0] qk_rounded;
wire signed [32:0] qk_shr8;
wire signed [15:0] qk_sat;
wire signed [15:0] qkm_wr_val;

assign sp_q_op = (state == S_SPLIT && sp_phase == 1'b1) ?
                 $signed(sram_q_q_i) : $signed(sp_q_r);
assign sp_k_op = (state == S_SPLIT && sp_phase == 1'b1) ?
                 $signed(sram_k_q_i) : $signed(sp_k_r);

assign sp_q_prod   = sp_q_op * S_Q88;
assign sp_q_rnd_t  = sp_q_prod + 32'sd128;
assign sp_q_shr    = sp_q_rnd_t >>> 8;
assign sp_q_sat    =
    (sp_q_shr > 32'sd32767) ? 16'sh7FFF :
    (sp_q_shr < -32'sh8000) ? 16'sh8000 : sp_q_shr[15:0];
assign sp_q_relu   =
    sp_q_sat[15] ? 16'sd0 :
    ($signed(sp_q_sat) > RELU6_MAX) ? RELU6_MAX[15:0] : sp_q_sat;

assign sp_k_prod   = sp_k_op * S_Q88;
assign sp_k_rnd_t  = sp_k_prod + 32'sd128;
assign sp_k_shr    = sp_k_rnd_t >>> 8;
assign sp_k_sat    =
    (sp_k_shr > 32'sd32767) ? 16'sh7FFF :
    (sp_k_shr < -32'sh8000) ? 16'sh8000 : sp_k_shr[15:0];
assign sp_k_relu   =
    sp_k_sat[15] ? 16'sd0 :
    ($signed(sp_k_sat) > RELU6_MAX) ? RELU6_MAX[15:0] : sp_k_sat;

assign km_h      = km_oidx[4:3];
assign km_d      = km_oidx[2:0];
assign km_k_flat =
    {12'd0, km_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, km_n}) * HEAD_DIM
  + {11'd0, km_d};

assign km_k_op = $signed(sram_k_q_i);
assign km_acc_next =
    (km_n == 9'd0) ? $signed({{17{km_k_op[15]}}, km_k_op})
                   : km_acc + $signed({{17{km_k_op[15]}}, km_k_op});
assign km_scaled =
    $signed({{15{km_acc_next[32]}}, km_acc_next}) * $signed({32'd0, RCP_N_NUM[15:0]});
assign km_shr_w  = (km_scaled + 48'sd32768) >>> RCP_N_SHIFT;
assign km_wr_val =
    (km_shr_w > 48'sd32767) ? 16'sh7FFF :
    (km_shr_w < -48'sd32768) ? 16'sh8000 : km_shr_w[15:0];

assign qk_q_flat =
    {12'd0, qk_h_reg} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, qk_n_reg}) * HEAD_DIM
  + {11'd0, qk_d};
assign qk_km_flat = {qk_h_reg, qk_d};

assign qk_q_op = $signed(sram_q_q_i);
assign qk_term = $signed(qk_q_op) * $signed(km_buf[qk_km_flat]);
assign qk_acc_next =
    (qk_d == 3'd0) ? $signed({qk_term[31], qk_term})
                   : qk_acc + $signed({qk_term[31], qk_term});
assign qk_rounded = qk_acc_next + 33'sd128;
assign qk_shr8    = qk_rounded >>> 8;
assign qk_sat     =
    (qk_shr8 > 33'sd32767) ? 16'sh7FFF :
    (qk_shr8 < -33'sd32768) ? 16'sh8000 : qk_shr8[15:0];
assign qkm_wr_val = (qk_sat < 16'sd1) ? 16'sd1 : qk_sat;

assign busy = (state != S_IDLE);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:    next_state = start ? S_SPLIT : S_IDLE;
        S_SPLIT:   next_state = (sp_ptr == HD_ELEMS[13:0] - 14'd1 && sp_phase == 1'b1) ?
                                 S_K_MEAN : S_SPLIT;
        S_K_MEAN:  next_state = (km_oidx == KM_ELEMS[4:0] - 5'd1 &&
                                 km_n == N_TOKENS[8:0] - 9'd1 && km_phase == 1'b1) ?
                                 S_QK_MEAN : S_K_MEAN;
        S_QK_MEAN: next_state = (qk_oidx == QKM_ELEMS[10:0] - 11'd1 &&
                                 qk_d == HEAD_DIM[2:0] - 3'd1 && qk_phase == 1'b1) ?
                                 S_DONE_ST : S_QK_MEAN;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// done pulse
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE_ST)
        done <= 1'b1;
    else
        done <= 1'b0;
end

// S_SPLIT shadow (debug)
always @(posedge clk) begin
    if (reset) begin
        sp_q_r <= 16'sd0;
        sp_k_r <= 16'sd0;
    end else if (state == S_SPLIT && sp_phase == 1'b0) begin
        sp_q_r <= sram_q_q_i;
        sp_k_r <= sram_k_q_i;
    end
end

// S_SPLIT pointer / phase
always @(posedge clk) begin
    if (reset) begin
        sp_ptr   <= 14'd0;
        sp_phase <= 1'b0;
    end else begin
        case (state)
            S_IDLE: begin
                sp_ptr   <= 14'd0;
                sp_phase <= 1'b0;
            end
            S_SPLIT: begin
                if (sp_phase == 1'b0)
                    sp_phase <= 1'b1;
                else if (sp_ptr < HD_ELEMS[13:0] - 14'd1) begin
                    sp_ptr   <= sp_ptr + 14'd1;
                    sp_phase <= 1'b0;
                end
            end
            default: ;
        endcase
    end
end

// S_K_MEAN / S_QK_MEAN datapath
always @(posedge clk) begin
    if (reset) begin
        km_oidx  <= 5'd0;
        km_n     <= 9'd0;
        km_acc   <= 33'sd0;
        km_phase <= 1'b0;
        qk_oidx  <= 11'd0;
        qk_h_reg <= 2'd0;
        qk_n_reg <= 9'd0;
        qk_d     <= 3'd0;
        qk_acc   <= 33'sd0;
        qk_phase <= 1'b0;
    end else begin
        case (state)
            S_IDLE: begin
                km_oidx  <= 5'd0;
                km_n     <= 9'd0;
                km_acc   <= 33'sd0;
                km_phase <= 1'b0;
                qk_oidx  <= 11'd0;
                qk_h_reg <= 2'd0;
                qk_n_reg <= 9'd0;
                qk_d     <= 3'd0;
                qk_acc   <= 33'sd0;
                qk_phase <= 1'b0;
            end

            S_K_MEAN: begin
                if (km_phase == 1'b0)
                    km_phase <= 1'b1;
                else begin
                    km_acc <= km_acc_next;
                    if (km_n == N_TOKENS[8:0] - 9'd1) begin
                        km_buf[km_oidx] <= km_wr_val;
                        km_n <= 9'd0;
                        if (km_oidx == KM_ELEMS[4:0] - 5'd1)
                            km_oidx <= 5'd0;
                        else
                            km_oidx <= km_oidx + 5'd1;
                    end else begin
                        km_n <= km_n + 9'd1;
                    end
                    km_phase <= 1'b0;
                end
                if (next_state == S_QK_MEAN) begin
                    qk_oidx  <= 11'd0;
                    qk_h_reg <= 2'd0;
                    qk_n_reg <= 9'd0;
                    qk_d     <= 3'd0;
                    qk_acc   <= 33'sd0;
                    qk_phase <= 1'b0;
                end
            end

            S_QK_MEAN: begin
                if (qk_phase == 1'b0)
                    qk_phase <= 1'b1;
                else begin
                    qk_acc <= qk_acc_next;
                    if (qk_d == HEAD_DIM[2:0] - 3'd1) begin
                        qk_d <= 3'd0;
                        if (qk_oidx == QKM_ELEMS[10:0] - 11'd1) begin
                            qk_oidx  <= 11'd0;
                            qk_h_reg <= 2'd0;
                            qk_n_reg <= 9'd0;
                        end else begin
                            qk_oidx <= qk_oidx + 11'd1;
                            if (qk_n_reg == N_TOKENS[8:0] - 9'd1) begin
                                qk_n_reg <= 9'd0;
                                qk_h_reg <= qk_h_reg + 2'd1;
                            end else begin
                                qk_n_reg <= qk_n_reg + 9'd1;
                            end
                        end
                    end else begin
                        qk_d <= qk_d + 3'd1;
                    end
                    qk_phase <= 1'b0;
                end
            end

            default: ;
        endcase
    end
end

// SRAM port mux
always @(*) begin
    sram_q_ceb_o     = 1'b1;
    sram_q_web_o     = 1'b1;
    sram_q_addr_o    = 14'd0;
    sram_q_din_o     = 16'd0;
    sram_k_ceb_o     = 1'b1;
    sram_k_web_o     = 1'b1;
    sram_k_addr_o    = 14'd0;
    sram_k_din_o     = 16'd0;
    sram_qkm_ceb_o   = 1'b1;
    sram_qkm_web_o   = 1'b1;
    sram_qkm_addr_o  = 14'd0;
    sram_qkm_din_o   = 16'd0;

    case (state)
        S_SPLIT: begin
            if (sp_phase == 1'b0) begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b1;
                sram_q_addr_o = sp_ptr;
                sram_k_ceb_o  = 1'b0;
                sram_k_web_o  = 1'b1;
                sram_k_addr_o = sp_ptr;
            end else begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b0;
                sram_q_addr_o = sp_ptr;
                sram_q_din_o  = sp_q_relu;
                sram_k_ceb_o  = 1'b0;
                sram_k_web_o  = 1'b0;
                sram_k_addr_o = sp_ptr;
                sram_k_din_o  = sp_k_relu;
            end
        end

        S_K_MEAN: begin
            sram_k_ceb_o  = 1'b0;
            sram_k_web_o  = 1'b1;
            sram_k_addr_o = km_k_flat;
        end

        S_QK_MEAN: begin
            sram_q_ceb_o  = 1'b0;
            sram_q_web_o  = 1'b1;
            sram_q_addr_o = qk_q_flat;
            if ((qk_phase == 1'b1) && (qk_d == HEAD_DIM[2:0] - 3'd1)) begin
                sram_qkm_ceb_o  = 1'b0;
                sram_qkm_web_o  = 1'b0;
                sram_qkm_addr_o = {3'd0, qk_oidx[10:0]};
                sram_qkm_din_o  = qkm_wr_val;
            end
        end

        default: ;
    endcase
end

endmodule
