// =============================================================================
// attn_core.v
//
// CARE attention S_ATTN: ao = rnd_shr8(sat16(sum_dk q*kv[h,dk,dout]) * zr).
// C2 architecture (per h,n): cache q_buf[0:7] + zr once, then 8-wide dk MAC.
//
// Per (h,n):
//   Q_ADDR/Q_CAP x8 : 2-phase SRAM load q_buf (1-cycle macro latency)
//   ZR_ADDR/ZR_CAP  : 2-phase load at_zr_r
//   Per d_out 0..7  : MAC8 (parallel sum) -> DOT (shr8+sat) -> AO write
//
// Inputs : split q (Sram_q), zr (Sram_qkm), kv_buf[NUM_HEADS*HEAD_DIM*HEAD_DIM]
// Output : ao (Sram_ao write; same role as Sram_v ao in care_attention.v)
//
// kv_buf layout from attn_kv: index h*64+d_out*8+d_k -> kv[h,d_out,d_k]
// MAC read index: h*64+dk*8+dout -> kv[h,dk,dout] (transpose; matches numpy _care_attn_q88)
//
// Golden: numpy _care_attn_q88 (per-token ao before proj linear)
// =============================================================================

module attn_core #(
    parameter EMBED_DIM = 32,
    parameter NUM_HEADS = 4,
    parameter HEAD_DIM  = 8,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output wire        busy,
    output reg         done,

    output reg         sram_q_ceb_o,
    output reg         sram_q_web_o,
    output reg [13:0]  sram_q_addr_o,
    input  wire [15:0] sram_q_q_i,

    output reg         sram_qkm_ceb_o,
    output reg         sram_qkm_web_o,
    output reg [13:0]  sram_qkm_addr_o,
    input  wire [15:0] sram_qkm_q_i,

    output reg         sram_ao_ceb_o,
    output reg         sram_ao_web_o,
    output reg [13:0]  sram_ao_addr_o,
    output reg [15:0]  sram_ao_din_o
);

localparam HD_ELEMS = NUM_HEADS * N_TOKENS * HEAD_DIM;
localparam KV_ELEMS = NUM_HEADS * HEAD_DIM * HEAD_DIM;

reg signed [15:0] kv_buf [0:KV_ELEMS-1];
reg signed [15:0] q_buf  [0:HEAD_DIM-1];

parameter S_IDLE    = 2'd0;
parameter S_ATTN    = 2'd1;
parameter S_DONE_ST = 2'd2;

localparam AT_PH_Q_ADDR  = 3'd0;
localparam AT_PH_Q_CAP   = 3'd1;
localparam AT_PH_ZR_ADDR = 3'd2;
localparam AT_PH_ZR_CAP  = 3'd3;
localparam AT_PH_MAC8    = 3'd4;
localparam AT_PH_DOT     = 3'd5;
localparam AT_PH_AO      = 3'd6;

reg [1:0] state;
reg [1:0] next_state;

reg [2:0]         at_phase;
reg [2:0]         q_load_dk;

reg [1:0]         at_h_reg;
reg [8:0]         at_n_reg;
reg [2:0]         at_dout_reg;

reg signed [48:0] at_acc;
reg signed [15:0] at_zr_r;
reg signed [15:0] at_dot_sat_r;

reg               ao_wr_en;
reg [13:0]        ao_wr_addr;
reg signed [15:0] ao_wr_data;

wire [13:0] at_q_flat_load;
wire [10:0] at_zr_idx;
wire [13:0] at_ao_flat;

wire signed [31:0] at_mac_term_0;
wire signed [31:0] at_mac_term_1;
wire signed [31:0] at_mac_term_2;
wire signed [31:0] at_mac_term_3;
wire signed [31:0] at_mac_term_4;
wire signed [31:0] at_mac_term_5;
wire signed [31:0] at_mac_term_6;
wire signed [31:0] at_mac_term_7;

wire [7:0] at_kv_flat_0;
wire [7:0] at_kv_flat_1;
wire [7:0] at_kv_flat_2;
wire [7:0] at_kv_flat_3;
wire [7:0] at_kv_flat_4;
wire [7:0] at_kv_flat_5;
wire [7:0] at_kv_flat_6;
wire [7:0] at_kv_flat_7;

wire signed [48:0] at_acc_par;

wire signed [48:0] at_dot_shr_from_acc;
wire signed [15:0] at_dot_sat_next;

wire signed [31:0] at_zprod_raw;
wire signed [31:0] at_zprod_rnd;
wire signed [31:0] at_zprod_shr_w;
wire signed [15:0] at_ao_sat_comb;

wire hn_last;
wire dout_last;
wire attn_done_now;

assign at_q_flat_load =
    {12'd0, at_h_reg} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, at_n_reg}) * HEAD_DIM
  + {11'd0, q_load_dk};

assign at_zr_idx =
    ({2'd0, at_h_reg} * N_TOKENS[10:0]) + {2'd0, at_n_reg};

assign at_ao_flat =
    ({5'd0, at_n_reg}) * EMBED_DIM[13:0]
  + {12'd0, at_h_reg} * HEAD_DIM
  + {11'd0, at_dout_reg};

assign at_kv_flat_0 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd0} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_1 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd1} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_2 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd2} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_3 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd3} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_4 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd4} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_5 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd5} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_6 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd6} * HEAD_DIM + {5'd0, at_dout_reg};
assign at_kv_flat_7 =
    {6'd0, at_h_reg} * (HEAD_DIM * HEAD_DIM) + {5'd0, 3'd7} * HEAD_DIM + {5'd0, at_dout_reg};

assign at_mac_term_0 = $signed(q_buf[0]) * $signed(kv_buf[at_kv_flat_0]);
assign at_mac_term_1 = $signed(q_buf[1]) * $signed(kv_buf[at_kv_flat_1]);
assign at_mac_term_2 = $signed(q_buf[2]) * $signed(kv_buf[at_kv_flat_2]);
assign at_mac_term_3 = $signed(q_buf[3]) * $signed(kv_buf[at_kv_flat_3]);
assign at_mac_term_4 = $signed(q_buf[4]) * $signed(kv_buf[at_kv_flat_4]);
assign at_mac_term_5 = $signed(q_buf[5]) * $signed(kv_buf[at_kv_flat_5]);
assign at_mac_term_6 = $signed(q_buf[6]) * $signed(kv_buf[at_kv_flat_6]);
assign at_mac_term_7 = $signed(q_buf[7]) * $signed(kv_buf[at_kv_flat_7]);

assign at_acc_par =
    $signed({{17{at_mac_term_0[31]}}, at_mac_term_0}) +
    $signed({{17{at_mac_term_1[31]}}, at_mac_term_1}) +
    $signed({{17{at_mac_term_2[31]}}, at_mac_term_2}) +
    $signed({{17{at_mac_term_3[31]}}, at_mac_term_3}) +
    $signed({{17{at_mac_term_4[31]}}, at_mac_term_4}) +
    $signed({{17{at_mac_term_5[31]}}, at_mac_term_5}) +
    $signed({{17{at_mac_term_6[31]}}, at_mac_term_6}) +
    $signed({{17{at_mac_term_7[31]}}, at_mac_term_7});

assign at_dot_shr_from_acc = (at_acc + 49'sd128) >>> 8;
assign at_dot_sat_next =
    (at_dot_shr_from_acc > 49'sd32767) ? 16'sh7FFF :
    (at_dot_shr_from_acc < -49'sd32768) ? 16'sh8000 : at_dot_shr_from_acc[15:0];

assign at_zprod_raw   = $signed(at_dot_sat_r) * $signed(at_zr_r);
assign at_zprod_rnd   = at_zprod_raw + 32'sd128;
assign at_zprod_shr_w = at_zprod_rnd >>> 8;
assign at_ao_sat_comb =
    (at_zprod_shr_w > 32'sd32767) ? 16'sh7FFF :
    (at_zprod_shr_w < -32'sd32768) ? 16'sh8000 : at_zprod_shr_w[15:0];

assign dout_last = (at_dout_reg == HEAD_DIM[2:0] - 3'd1);
assign hn_last   = (at_h_reg == NUM_HEADS[1:0] - 2'd1) &&
                   (at_n_reg == N_TOKENS[8:0] - 9'd1);
assign attn_done_now = hn_last && dout_last && (at_phase == AT_PH_AO);

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
        S_IDLE:    next_state = start ? S_ATTN : S_IDLE;
        S_ATTN:    next_state = attn_done_now ? S_DONE_ST : S_ATTN;
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

// AO write capture on AT_PH_AO
always @(posedge clk) begin
    if (reset) begin
        ao_wr_en   <= 1'b0;
        ao_wr_addr <= 14'd0;
        ao_wr_data <= 16'sd0;
    end else if (state == S_ATTN && at_phase == AT_PH_AO) begin
        ao_wr_en   <= 1'b1;
        ao_wr_addr <= at_ao_flat;
        ao_wr_data <= at_ao_sat_comb;
    end else begin
        ao_wr_en <= 1'b0;
    end
end

// S_ATTN datapath
always @(posedge clk) begin
    if (reset) begin
        at_phase     <= AT_PH_Q_ADDR;
        q_load_dk    <= 3'd0;
        at_h_reg     <= 2'd0;
        at_n_reg     <= 9'd0;
        at_dout_reg  <= 3'd0;
        at_acc       <= 49'sd0;
        at_zr_r      <= 16'sd0;
        at_dot_sat_r <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                at_phase     <= AT_PH_Q_ADDR;
                q_load_dk    <= 3'd0;
                at_h_reg     <= 2'd0;
                at_n_reg     <= 9'd0;
                at_dout_reg  <= 3'd0;
                at_acc       <= 49'sd0;
                at_zr_r      <= 16'sd0;
                at_dot_sat_r <= 16'sd0;
            end

            S_ATTN: begin
                case (at_phase)
                    AT_PH_Q_ADDR: begin
                        at_phase <= AT_PH_Q_CAP;
                    end

                    AT_PH_Q_CAP: begin
                        q_buf[q_load_dk] <= sram_q_q_i;
                        if (q_load_dk == HEAD_DIM[2:0] - 3'd1) begin
                            q_load_dk <= 3'd0;
                            at_phase  <= AT_PH_ZR_ADDR;
                        end else begin
                            q_load_dk <= q_load_dk + 3'd1;
                            at_phase  <= AT_PH_Q_ADDR;
                        end
                    end

                    AT_PH_ZR_ADDR: begin
                        at_phase <= AT_PH_ZR_CAP;
                    end

                    AT_PH_ZR_CAP: begin
                        at_zr_r    <= sram_qkm_q_i;
                        at_dout_reg <= 3'd0;
                        at_phase   <= AT_PH_MAC8;
                    end

                    AT_PH_MAC8: begin
                        at_acc   <= at_acc_par;
                        at_phase <= AT_PH_DOT;
                    end

                    AT_PH_DOT: begin
                        at_dot_sat_r <= at_dot_sat_next;
                        at_phase     <= AT_PH_AO;
                    end

                    AT_PH_AO: begin
                        if (!dout_last) begin
                            at_dout_reg <= at_dout_reg + 3'd1;
                            at_phase    <= AT_PH_MAC8;
                        end else if (at_n_reg != N_TOKENS[8:0] - 9'd1) begin
                            at_n_reg    <= at_n_reg + 9'd1;
                            at_dout_reg <= 3'd0;
                            q_load_dk   <= 3'd0;
                            at_phase    <= AT_PH_Q_ADDR;
                        end else if (at_h_reg != NUM_HEADS[1:0] - 2'd1) begin
                            at_h_reg    <= at_h_reg + 2'd1;
                            at_n_reg    <= 9'd0;
                            at_dout_reg <= 3'd0;
                            q_load_dk   <= 3'd0;
                            at_phase    <= AT_PH_Q_ADDR;
                        end
                    end

                    default: at_phase <= AT_PH_Q_ADDR;
                endcase
            end

            default: ;
        endcase
    end
end

// SRAM port mux
always @(*) begin
    sram_q_ceb_o    = 1'b1;
    sram_q_web_o    = 1'b1;
    sram_q_addr_o   = 14'd0;
    sram_qkm_ceb_o  = 1'b1;
    sram_qkm_web_o  = 1'b1;
    sram_qkm_addr_o = 14'd0;
    sram_ao_ceb_o   = 1'b1;
    sram_ao_web_o   = 1'b1;
    sram_ao_addr_o  = 14'd0;
    sram_ao_din_o   = 16'd0;

    if (state == S_ATTN) begin
        if (at_phase == AT_PH_Q_ADDR) begin
            sram_q_ceb_o  = 1'b0;
            sram_q_web_o  = 1'b1;
            sram_q_addr_o = at_q_flat_load;
        end
        if (at_phase == AT_PH_ZR_ADDR) begin
            sram_qkm_ceb_o  = 1'b0;
            sram_qkm_web_o  = 1'b1;
            sram_qkm_addr_o = {3'd0, at_zr_idx[10:0]};
        end
    end

    if (ao_wr_en) begin
        sram_ao_ceb_o  = 1'b0;
        sram_ao_web_o  = 1'b0;
        sram_ao_addr_o = ao_wr_addr;
        sram_ao_din_o  = ao_wr_data;
    end
end

endmodule
