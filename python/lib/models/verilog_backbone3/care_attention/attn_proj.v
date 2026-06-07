// =============================================================================
// attn_proj.v
//
// CARE attention S_PROJ: read ao from Sram_ao + linear_vec8 (32->32) -> y stream.
// Bit-accurate vs run_backbone_numpy_shared_trunk.py linear_sat_q88(ao, proj_w, proj_b)
// and verilog_backbone2/care_attention.v S_PROJ.
//
// Per token pj_sub: START(addr ao[t,0]) -> USE(feed linear) -> ADDR -> USE ... -> WAIT.
// SRAM ao read: 1-cycle latency (ADDR/START beat -> USE beat consumes sram_ao_q_i).
//
// wgt_addr_o: local proj ROM addr 0..1023 (parent backbone adds +3072 if merged).
// Golden: Activation/backbone_blocks_<b>_after_attn_attn_out_bi.txt
// Golden-Weight: Weight/backbone_blocks_<b>_attn_proj_{weight,bias}_bi.txt
// =============================================================================

module attn_proj #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [12:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    output reg signed [15:0] y_o,
    output reg               y_valid,
    output reg [6:0]         y_neu_o,
    output wire [8:0]        px_tok_o,

    output reg         sram_ao_ceb_o,
    output reg         sram_ao_web_o,
    output reg [13:0]  sram_ao_addr_o,
    input  wire [15:0] sram_ao_q_i
);

parameter S_IDLE    = 2'd0;
parameter S_PROJ    = 2'd1;
parameter S_DONE_ST = 2'd2;

parameter PJ_START = 2'd0;
parameter PJ_ADDR  = 2'd1;
parameter PJ_USE   = 2'd2;
parameter PJ_WAIT  = 2'd3;

reg [1:0] state;
reg [1:0] next_state;

reg [8:0]         px_tok;
reg [5:0]         proj_stream_cnt;
reg [1:0]         pj_sub;

reg               lin_proj_start;
reg signed [15:0] lin_proj_x;
reg               lin_proj_xv;
wire signed [15:0] lin_proj_y;
wire              lin_proj_yv;
wire [6:0]        lin_proj_neu;
wire [12:0]       lin_proj_addr;
wire              lin_proj_busy;
wire              lin_proj_done;

wire [13:0] pj_ao_addr;

assign pj_ao_addr =
    ({5'd0, px_tok}) * EMBED_DIM[13:0] + {9'd0, proj_stream_cnt};

assign wgt_addr_o = (state == S_PROJ) ? lin_proj_addr : 13'd0;
assign busy       = (state != S_IDLE);
assign px_tok_o   = px_tok;

linear_vec8 #(
    .IN_DIM    (EMBED_DIM),
    .OUT_DIM   (EMBED_DIM),
    .IN_DIM_AW (5),
    .NEU_AW    (8)
) u_lin_proj (
    .clk      (clk),
    .reset    (reset),
    .start    (lin_proj_start),
    .x_i      (lin_proj_x),
    .x_valid  (lin_proj_xv),
    .w_i      (wgt_i),
    .b_i      (bias_i),
    .w_addr_o (lin_proj_addr),
    .busy     (lin_proj_busy),
    .done     (lin_proj_done),
    .y_o      (lin_proj_y),
    .y_valid  (lin_proj_yv),
    .y_neu_o  (lin_proj_neu)
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
        S_IDLE:    next_state = start ? S_PROJ : S_IDLE;
        S_PROJ:    next_state = (lin_proj_done && px_tok == N_TOKENS[8:0] - 9'd1) ?
                                 S_DONE_ST : S_PROJ;
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

// S_PROJ datapath
always @(posedge clk) begin
    lin_proj_start <= 1'b0;
    lin_proj_xv    <= 1'b0;
    y_valid        <= 1'b0;

    if (reset) begin
        px_tok          <= 9'd0;
        proj_stream_cnt <= 6'd0;
        pj_sub          <= PJ_START;
        lin_proj_x      <= 16'sd0;
        y_o             <= 16'sd0;
        y_neu_o         <= 7'd0;
    end else begin
        case (state)
            S_IDLE: begin
                px_tok          <= 9'd0;
                proj_stream_cnt <= 6'd0;
                pj_sub          <= PJ_START;
            end

            S_PROJ: begin
                case (pj_sub)
                    PJ_START: begin
                        lin_proj_start  <= 1'b1;
                        proj_stream_cnt <= 6'd0;
                        pj_sub          <= PJ_USE;
                    end
                    PJ_ADDR: begin
                        pj_sub <= PJ_USE;
                    end
                    PJ_USE: begin
                        lin_proj_x <= $signed(sram_ao_q_i);
                        lin_proj_xv <= 1'b1;
                        if (proj_stream_cnt == EMBED_DIM[5:0] - 6'd1)
                            pj_sub <= PJ_WAIT;
                        else begin
                            proj_stream_cnt <= proj_stream_cnt + 6'd1;
                            pj_sub          <= PJ_ADDR;
                        end
                    end
                    PJ_WAIT: begin
                        // wait lin_proj_done
                    end
                    default: ;
                endcase

                if (lin_proj_yv) begin
                    y_o     <= lin_proj_y;
                    y_neu_o <= lin_proj_neu;
                    y_valid <= 1'b1;
                end

                if (lin_proj_done) begin
                    if (px_tok == N_TOKENS[8:0] - 9'd1) begin
                        px_tok          <= 9'd0;
                        proj_stream_cnt <= 6'd0;
                    end else begin
                        px_tok          <= px_tok + 9'd1;
                        proj_stream_cnt <= 6'd0;
                        pj_sub          <= PJ_START;
                    end
                end
            end

            default: ;
        endcase
    end
end

// SRAM ao read mux (ADDR/START phases)
always @(*) begin
    sram_ao_ceb_o  = 1'b1;
    sram_ao_web_o  = 1'b1;
    sram_ao_addr_o = 14'd0;

    if (state == S_PROJ) begin
        if ((pj_sub == PJ_START) || (pj_sub == PJ_ADDR)) begin
            sram_ao_ceb_o  = 1'b0;
            sram_ao_web_o  = 1'b1;
            sram_ao_addr_o = pj_ao_addr;
        end
    end
end

endmodule
