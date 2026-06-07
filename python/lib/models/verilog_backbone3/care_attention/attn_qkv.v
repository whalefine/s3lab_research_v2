// =============================================================================
// attn_qkv.v
//
// QKV linear (norm1 staging read + linear_vec8 32->96) and capture to Sram_q/k/v.
// Bit-accurate vs run_backbone_numpy_shared_trunk.py attention_forward() QKV stage
// and verilog_backbone2/care_attention.v S_QKV.
//
// Per token:
//   2-phase read parent norm1 buffer (norm_rd_en / norm_rd_flat / norm_x)
//   linear_vec8 streams 96 outputs; neu[6:5] selects q(00)/k(01)/v(10) SRAM write.
//   Capture flat (H,N,d) C-order:
//     flat = h*(N_TOKENS*HEAD_DIM) + n*HEAD_DIM + d
//     h = neu[4:3], d = neu[2:0], n = qx_tok
//
// wgt_addr_o: 13-bit local QKV ROM addr (0..3071); parent adds 3'b010 in backbone_top.
//
// Golden: Activation/backbone_blocks_<b>_attn_after_qkv_{q,k,v}_bi.txt
// Golden-Weight: Weight/backbone_blocks_<b>_attn_qkv_{weight,bias}_bi.txt
// =============================================================================

module attn_qkv #(
    parameter EMBED_DIM = 32,
    parameter NUM_HEADS = 4,
    parameter HEAD_DIM  = 8,
    parameter N_TOKENS  = 320,
    parameter QKV_OUT   = 96
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output reg         norm_rd_en,
    output reg [13:0]  norm_rd_flat,
    input  wire signed [15:0] norm_x,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [12:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    output reg         sram_q_ceb_o,
    output reg         sram_q_web_o,
    output reg [13:0]  sram_q_addr_o,
    output reg [15:0]  sram_q_din_o,

    output reg         sram_k_ceb_o,
    output reg         sram_k_web_o,
    output reg [13:0]  sram_k_addr_o,
    output reg [15:0]  sram_k_din_o,

    output reg         sram_v_ceb_o,
    output reg         sram_v_web_o,
    output reg [13:0]  sram_v_addr_o,
    output reg [15:0]  sram_v_din_o
);

parameter S_IDLE    = 4'd0;
parameter S_QKV     = 4'd1;
parameter S_DONE_ST = 4'd2;

reg [3:0] state;
reg [3:0] next_state;

reg [8:0]           qx_tok;
reg [5:0]           qkv_stream_cnt;
reg                 qkv_x_phase;

reg                 lin_qkv_start;
reg signed [15:0]   lin_qkv_x;
reg                 lin_qkv_xv;
wire signed [15:0]  lin_qkv_y;
wire                lin_qkv_yv;
wire [6:0]          lin_qkv_neu;
wire [12:0]         lin_qkv_addr;
wire                lin_qkv_busy;
wire                lin_qkv_done;

wire [13:0] qkv_x_flat;
wire [13:0] qkv_cap_flat_w;

assign qkv_x_flat =
    ({5'd0, qx_tok}) * EMBED_DIM + {9'd0, qkv_stream_cnt - 6'd1};

assign qkv_cap_flat_w =
    {12'd0, lin_qkv_neu[4:3]} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, qx_tok}) * HEAD_DIM
  + {11'd0, lin_qkv_neu[2:0]};

assign wgt_addr_o = (state == S_QKV) ? lin_qkv_addr : 13'd0;
assign busy       = (state != S_IDLE);

linear_vec8 #(
    .IN_DIM    (EMBED_DIM),
    .OUT_DIM   (QKV_OUT),
    .IN_DIM_AW (5),
    .NEU_AW    (8)
) u_lin_qkv (
    .clk      (clk),
    .reset    (reset),
    .start    (lin_qkv_start),
    .x_i      (lin_qkv_x),
    .x_valid  (lin_qkv_xv),
    .w_i      (wgt_i),
    .b_i      (bias_i),
    .w_addr_o (lin_qkv_addr),
    .busy     (lin_qkv_busy),
    .done     (lin_qkv_done),
    .y_o      (lin_qkv_y),
    .y_valid  (lin_qkv_yv),
    .y_neu_o  (lin_qkv_neu)
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
    case (state)
        S_IDLE:    next_state = start ? S_QKV : S_IDLE;
        S_QKV:     next_state = (lin_qkv_done && qx_tok == N_TOKENS[8:0] - 9'd1) ?
                                 S_DONE_ST : S_QKV;
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

// S_QKV datapath: norm1 2-phase read + linear_vec8 feed
always @(posedge clk) begin
    lin_qkv_start <= 1'b0;
    lin_qkv_xv    <= 1'b0;
    norm_rd_en    <= 1'b0;
    norm_rd_flat  <= 14'd0;

    if (reset) begin
        qx_tok         <= 9'd0;
        qkv_stream_cnt <= 6'd0;
        qkv_x_phase    <= 1'b0;
        lin_qkv_x      <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                qx_tok         <= 9'd0;
                qkv_stream_cnt <= 6'd0;
                qkv_x_phase    <= 1'b0;
            end

            S_QKV: begin
                if (qkv_stream_cnt == 6'd0) begin
                    lin_qkv_start  <= 1'b1;
                    qkv_stream_cnt <= 6'd1;
                    qkv_x_phase    <= 1'b0;
                end else if (qkv_stream_cnt <= EMBED_DIM[5:0]) begin
                    if (qkv_x_phase == 1'b0) begin
                        norm_rd_en   <= 1'b1;
                        norm_rd_flat <= qkv_x_flat;
                        qkv_x_phase  <= 1'b1;
                    end else begin
                        lin_qkv_x      <= norm_x;
                        lin_qkv_xv     <= 1'b1;
                        qkv_stream_cnt <= qkv_stream_cnt + 6'd1;
                        qkv_x_phase    <= 1'b0;
                    end
                end

                if (lin_qkv_done) begin
                    if (qx_tok == N_TOKENS[8:0] - 9'd1) begin
                        qx_tok         <= 9'd0;
                        qkv_stream_cnt <= 6'd0;
                    end else begin
                        qx_tok         <= qx_tok + 9'd1;
                        qkv_stream_cnt <= 6'd0;
                    end
                end
            end

            default: ;
        endcase
    end
end

// SRAM q/k/v write mux (S_QKV capture on lin_qkv_yv)
always @(*) begin
    sram_q_ceb_o  = 1'b1;
    sram_q_web_o  = 1'b1;
    sram_q_addr_o = 14'd0;
    sram_q_din_o  = 16'd0;
    sram_k_ceb_o  = 1'b1;
    sram_k_web_o  = 1'b1;
    sram_k_addr_o = 14'd0;
    sram_k_din_o  = 16'd0;
    sram_v_ceb_o  = 1'b1;
    sram_v_web_o  = 1'b1;
    sram_v_addr_o = 14'd0;
    sram_v_din_o  = 16'd0;

    if (state == S_QKV && lin_qkv_yv) begin
        case (lin_qkv_neu[6:5])
            2'b00: begin
                sram_q_ceb_o  = 1'b0;
                sram_q_web_o  = 1'b0;
                sram_q_addr_o = qkv_cap_flat_w;
                sram_q_din_o  = lin_qkv_y;
            end
            2'b01: begin
                sram_k_ceb_o  = 1'b0;
                sram_k_web_o  = 1'b0;
                sram_k_addr_o = qkv_cap_flat_w;
                sram_k_din_o  = lin_qkv_y;
            end
            2'b10: begin
                sram_v_ceb_o  = 1'b0;
                sram_v_web_o  = 1'b0;
                sram_v_addr_o = qkv_cap_flat_w;
                sram_v_din_o  = lin_qkv_y;
            end
            default: ;
        endcase
    end
end

endmodule
