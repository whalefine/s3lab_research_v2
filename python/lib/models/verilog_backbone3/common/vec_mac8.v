// =============================================================================
// vec_mac8.v
//
// 8-lane Q8.8 MAC engine for backbone3 linear layers.
// Reference: verilog_head3/conv.v (OC_PAR=8 WPRE + 1-phase MAC pipeline).
//
// One session computes OUT_DIM lanes starting at neu_base_i:
//   S_WPRE : 2-phase ROM prefetch -> wgt_buf[0:7][0:IN_DIM-1], bias_r[0:7]
//   S_MAC  : mac_fill + IN_DIM accumulate beats (shared x_i, 8 parallel MACs)
//
// Weight flat addr (parent ROM mux): w_addr = (neu_base + lane) * IN_DIM + feat
// Bias addr: b_addr = neu_base + lane
//
// ROM CLK(~clk): posedge T drive w_addr/b_addr -> T+1 w_i/b_i valid (parent latches).
//
// Parent feeds x_i when x_consume_o=1; use mac_feat_o to index x_buf[feat].
// On mac_done (1-cycle pulse), read acc_sat_* and bias_* for SAT in linear_vec8.
//
// Saturation via wire only (no function). Verilog-2001 synthesizable.
// =============================================================================

module vec_mac8 #(
    parameter DATA_W    = 16,
    parameter ACC_W     = 32,
    parameter LANES     = 8,
    parameter IN_DIM    = 32,
    parameter IN_DIM_AW = 7,
    parameter NEU_AW    = 7,
    parameter W_ADDR_W  = 16,
    parameter B_ADDR_W  = 8
) (
    input  wire                       clk,
    input  wire                       reset,
    input  wire                       start,
    input  wire [NEU_AW-1:0]          neu_base_i,
    input  wire [NEU_AW-1:0]          out_dim_i,
    output wire                       busy,
    output reg                        mac_done,
    output wire [W_ADDR_W-1:0]        w_addr_o,
    output wire [B_ADDR_W-1:0]        b_addr_o,
    input  wire signed [DATA_W-1:0]   w_i,
    input  wire signed [DATA_W-1:0]   b_i,
    input  wire signed [DATA_W-1:0]   x_i,
    output wire                       x_consume_o,
    output wire                       mac_active_o,
    output wire [IN_DIM_AW-1:0]       mac_feat_o,
    output wire signed [ACC_W-1:0]    acc_sat_0,
    output wire signed [ACC_W-1:0]    acc_sat_1,
    output wire signed [ACC_W-1:0]    acc_sat_2,
    output wire signed [ACC_W-1:0]    acc_sat_3,
    output wire signed [ACC_W-1:0]    acc_sat_4,
    output wire signed [ACC_W-1:0]    acc_sat_5,
    output wire signed [ACC_W-1:0]    acc_sat_6,
    output wire signed [ACC_W-1:0]    acc_sat_7,
    output wire signed [DATA_W-1:0]   bias_0,
    output wire signed [DATA_W-1:0]   bias_1,
    output wire signed [DATA_W-1:0]   bias_2,
    output wire signed [DATA_W-1:0]   bias_3,
    output wire signed [DATA_W-1:0]   bias_4,
    output wire signed [DATA_W-1:0]   bias_5,
    output wire signed [DATA_W-1:0]   bias_6,
    output wire signed [DATA_W-1:0]   bias_7,
    output wire [LANES-1:0]           lane_valid_o
);

parameter S_IDLE = 2'd0;
parameter S_WPRE = 2'd1;
parameter S_MAC  = 2'd2;

localparam [IN_DIM_AW-1:0] FEAT_LAST = IN_DIM - 1;

integer i_lane;

reg [1:0]                   state;
reg [1:0]                   next_state;

reg [NEU_AW-1:0]            neu_base_r;
reg [NEU_AW-1:0]            out_dim_r;

reg                         wpre_phase;
reg [IN_DIM_AW-1:0]         wpre_feat;
reg [3:0]                   wpre_lane;
reg                         wpre_done;
reg                         bpre_phase;
reg [3:0]                   bpre_lane;
reg                         bpre_done;

reg                         mac_fill;
reg [IN_DIM_AW-1:0]         mac_feat;
reg                         mac_done_r;

reg [W_ADDR_W-1:0]          w_addr_r;
reg [B_ADDR_W-1:0]          b_addr_r;

reg signed [DATA_W-1:0]     bias_r [0:LANES-1];
reg signed [DATA_W-1:0]     wgt_buf [0:LANES-1][0:IN_DIM-1];
reg signed [ACC_W-1:0]      acc_r [0:LANES-1];
reg signed [ACC_W-1:0]      acc_sat_r [0:LANES-1];

wire                        mac_feat_last;
wire signed [DATA_W-1:0]    mac_x_op;
wire [LANES-1:0]            lane_valid_w;

reg signed [DATA_W-1:0]     mac_w_op [0:LANES-1];
reg signed [2*DATA_W-1:0]   mac_prod [0:LANES-1];
reg signed [ACC_W-1:0]      acc_next [0:LANES-1];

wire                        cs_en;
wire                        wpre_clr;
wire                        wpre_w_rom_a0;
wire                        wpre_w_rom_a1_last;
wire                        wpre_w_rom_a1_more;
wire                        wpre_w_rom_a1_lane;
wire                        bpre_rom_a0;
wire                        bpre_rom_a1_last;
wire                        bpre_rom_a1_more;
wire                        mac_wpre_arm;
wire                        mac_fill_rd;
wire                        mac_accum_last;
wire                        mac_accum_more;

wire [NEU_AW-1:0]           wpre_neu_idx;
wire [W_ADDR_W-1:0]         wpre_w_addr_calc;

assign busy          = (state != S_IDLE);
assign w_addr_o      = w_addr_r;
assign b_addr_o      = b_addr_r;
assign mac_feat_o    = mac_feat;
assign mac_active_o  = (state == S_MAC) && !mac_done_r;
assign x_consume_o   = (state == S_MAC) && !mac_done_r && !mac_fill;

assign mac_feat_last = (mac_feat == FEAT_LAST);

assign cs_en = 1'b1;

assign wpre_neu_idx    = neu_base_r + {{(NEU_AW-4){1'b0}}, wpre_lane};
assign wpre_w_addr_calc = wpre_neu_idx * IN_DIM + {{(W_ADDR_W-IN_DIM_AW){1'b0}}, wpre_feat};

generate
    genvar gi;
    for (gi = 0; gi < LANES; gi = gi + 1) begin : gen_lane_valid
        assign lane_valid_w[gi] = (neu_base_r + gi[NEU_AW-1:0] < out_dim_r);
    end
endgenerate

assign lane_valid_o = lane_valid_w;

assign acc_sat_0 = acc_sat_r[0];
assign acc_sat_1 = acc_sat_r[1];
assign acc_sat_2 = acc_sat_r[2];
assign acc_sat_3 = acc_sat_r[3];
assign acc_sat_4 = acc_sat_r[4];
assign acc_sat_5 = acc_sat_r[5];
assign acc_sat_6 = acc_sat_r[6];
assign acc_sat_7 = acc_sat_r[7];

assign bias_0 = bias_r[0];
assign bias_1 = bias_r[1];
assign bias_2 = bias_r[2];
assign bias_3 = bias_r[3];
assign bias_4 = bias_r[4];
assign bias_5 = bias_r[5];
assign bias_6 = bias_r[6];
assign bias_7 = bias_r[7];

assign mac_x_op = x_i;

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
                next_state = S_WPRE;
        end
        S_WPRE: begin
            if (wpre_done && bpre_done)
                next_state = S_MAC;
        end
        S_MAC: begin
            if (mac_done_r)
                next_state = S_IDLE;
        end
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3a: capture neu_base / out_dim on start
always @(posedge clk) begin
    if (reset) begin
        neu_base_r <= {NEU_AW{1'b0}};
        out_dim_r  <= {NEU_AW{1'b0}};
    end else if (state == S_IDLE && start) begin
        neu_base_r <= neu_base_i;
        out_dim_r  <= out_dim_i;
    end
end

// FSM segment 3b: mac_done pulse
always @(posedge clk) begin
    if (reset)
        mac_done <= 1'b0;
    else
        mac_done <= (state == S_MAC) && mac_done_r;
end

assign wpre_clr = cs_en && (state == S_IDLE);

assign wpre_w_rom_a0 = cs_en && (state == S_WPRE) && !wpre_done && (wpre_phase == 1'b0);

assign wpre_w_rom_a1_last = cs_en && (state == S_WPRE) && !wpre_done && (wpre_phase == 1'b1) &&
                            (wpre_lane == LANES - 1) &&
                            (wpre_feat == FEAT_LAST);

assign wpre_w_rom_a1_more = cs_en && (state == S_WPRE) && !wpre_done && (wpre_phase == 1'b1) &&
                            (wpre_lane == LANES - 1) &&
                            (wpre_feat != FEAT_LAST);

assign wpre_w_rom_a1_lane = cs_en && (state == S_WPRE) && !wpre_done && (wpre_phase == 1'b1) &&
                            (wpre_lane != LANES - 1);

assign bpre_rom_a0 = cs_en && (state == S_WPRE) && wpre_done && !bpre_done && (bpre_phase == 1'b0);

assign bpre_rom_a1_last = cs_en && (state == S_WPRE) && wpre_done && !bpre_done &&
                          (bpre_phase == 1'b1) && (bpre_lane == LANES - 1);

assign bpre_rom_a1_more = cs_en && (state == S_WPRE) && wpre_done && !bpre_done &&
                          (bpre_phase == 1'b1) && (bpre_lane != LANES - 1);

// WPRE: weight/bias prefetch (2-phase ROM)
always @(posedge clk) begin
    if (reset) begin
        wpre_phase <= 1'b0;
        wpre_feat  <= {IN_DIM_AW{1'b0}};
        wpre_lane  <= 4'd0;
        wpre_done  <= 1'b0;
        bpre_phase <= 1'b0;
        bpre_lane  <= 4'd0;
        bpre_done  <= 1'b0;
        w_addr_r   <= {W_ADDR_W{1'b0}};
        b_addr_r   <= {B_ADDR_W{1'b0}};
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            bias_r[i_lane] <= {DATA_W{1'b0}};
    end else if (wpre_clr) begin
        wpre_phase <= 1'b0;
        wpre_feat  <= {IN_DIM_AW{1'b0}};
        wpre_lane  <= 4'd0;
        wpre_done  <= 1'b0;
        bpre_phase <= 1'b0;
        bpre_lane  <= 4'd0;
        bpre_done  <= 1'b0;
    end else if (wpre_w_rom_a0) begin
        w_addr_r   <= wpre_w_addr_calc;
        wpre_phase <= 1'b1;
    end else if (wpre_w_rom_a1_last) begin
        wgt_buf[wpre_lane][wpre_feat] <= w_i;
        wpre_lane  <= 4'd0;
        wpre_done  <= 1'b1;
        wpre_phase <= 1'b0;
    end else if (wpre_w_rom_a1_more) begin
        wgt_buf[wpre_lane][wpre_feat] <= w_i;
        wpre_lane  <= 4'd0;
        wpre_feat  <= wpre_feat + {{(IN_DIM_AW-1){1'b0}}, 1'b1};
        wpre_phase <= 1'b0;
    end else if (wpre_w_rom_a1_lane) begin
        wgt_buf[wpre_lane][wpre_feat] <= w_i;
        wpre_lane  <= wpre_lane + 4'd1;
        wpre_phase <= 1'b0;
    end else if (bpre_rom_a0) begin
        b_addr_r   <= neu_base_r[B_ADDR_W-1:0] + {{(B_ADDR_W-4){1'b0}}, bpre_lane};
        bpre_phase <= 1'b1;
    end else if (bpre_rom_a1_last) begin
        bias_r[bpre_lane] <= b_i;
        bpre_done  <= 1'b1;
        bpre_phase <= 1'b0;
    end else if (bpre_rom_a1_more) begin
        bias_r[bpre_lane] <= b_i;
        bpre_lane  <= bpre_lane + 4'd1;
        bpre_phase <= 1'b0;
    end
end

assign mac_wpre_arm   = cs_en && (state == S_WPRE) && wpre_done && bpre_done;
assign mac_fill_rd    = cs_en && (state == S_MAC) && !mac_done_r && mac_fill;
assign mac_accum_last = cs_en && (state == S_MAC) && !mac_done_r && !mac_fill && mac_feat_last;
assign mac_accum_more = cs_en && (state == S_MAC) && !mac_done_r && !mac_fill && !mac_feat_last;

// MAC: 1-phase pipeline (mac_fill + IN_DIM accumulate beats)
always @(posedge clk) begin
    if (reset) begin
        mac_fill   <= 1'b1;
        mac_feat   <= {IN_DIM_AW{1'b0}};
        mac_done_r <= 1'b0;
    end else if (mac_wpre_arm) begin
        mac_fill   <= 1'b1;
        mac_feat   <= {IN_DIM_AW{1'b0}};
        mac_done_r <= 1'b0;
    end else if (mac_fill_rd) begin
        mac_fill <= 1'b0;
    end else if (mac_accum_last) begin
        mac_done_r <= 1'b1;
    end else if (mac_accum_more) begin
        mac_feat <= mac_feat + {{(IN_DIM_AW-1){1'b0}}, 1'b1};
    end else if (state == S_IDLE) begin
        mac_fill   <= 1'b1;
        mac_feat   <= {IN_DIM_AW{1'b0}};
        mac_done_r <= 1'b0;
    end
end

// MAC: per-lane accumulator update
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
            acc_r[i_lane]     <= {ACC_W{1'b0}};
            acc_sat_r[i_lane] <= {ACC_W{1'b0}};
        end
    end else if (mac_wpre_arm) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc_r[i_lane] <= {ACC_W{1'b0}};
    end else if (mac_accum_last) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
            if (lane_valid_w[i_lane])
                acc_r[i_lane] <= acc_next[i_lane];
            if (lane_valid_w[i_lane])
                acc_sat_r[i_lane] <= acc_next[i_lane];
        end
    end else if (mac_accum_more) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
            if (lane_valid_w[i_lane])
                acc_r[i_lane] <= acc_next[i_lane];
        end
    end else if (state == S_IDLE) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
            acc_r[i_lane]     <= {ACC_W{1'b0}};
            acc_sat_r[i_lane] <= {ACC_W{1'b0}};
        end
    end
end

// MAC: 8-lane combinational multiply-accumulate
always @(*) begin
    for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
        mac_w_op[i_lane] = wgt_buf[i_lane][mac_feat];
        if ((state == S_MAC) && !mac_done_r && !mac_fill && lane_valid_w[i_lane]) begin
            mac_prod[i_lane] = mac_x_op * mac_w_op[i_lane];
            acc_next[i_lane] = acc_r[i_lane] + mac_prod[i_lane];
        end else begin
            mac_prod[i_lane] = {2*DATA_W{1'b0}};
            acc_next[i_lane] = acc_r[i_lane];
        end
    end
end

endmodule
