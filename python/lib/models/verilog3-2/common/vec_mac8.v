// =============================================================================
// vec_mac8.v
//
// 8-lane Q8.8 MAC engine for backbone3 linear layers (Option A: stream ROM).
// Reference: verilog_backbone2/linear.v MAC_PREF + MAC_RUN (no full WPRE buffer).
//
// One session computes OUT_DIM lanes starting at neu_base_i:
//   S_BIAS : 2-phase ROM prefetch -> bias_r[0:7]
//   S_RUN  : per feat, 8 lanes x 2-phase weight ROM -> acc[lane] += x_i * w_i
//            (fused fetch+MAC; no wgt_buf, no separate MAC pass over buffer)
//
// Weight flat addr (parent ROM mux): w_addr = (neu_base + lane) * IN_DIM + feat
// Bias addr: b_addr = neu_base + lane
//
// ROM CLK(clk): w_addr/b_addr are COMBINATIONAL (driven from current lane/feat
//   regs that are stable across the 2-phase issue+consume window). With a posedge
//   registered-data macro this gives addr@phase0 -> Q valid@phase1, so the existing
//   2-phase consume (S_BIAS / S_RUN) stays bit-aligned with no FSM retime. Driving a
//   registered addr would slip the data one cycle late under CLK(clk).
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
parameter S_BIAS = 2'd1;
parameter S_RUN  = 2'd2;

localparam [IN_DIM_AW-1:0] FEAT_LAST = IN_DIM - 1;

integer i_lane;

// Avoid out-of-bounds slice when parameter defaults or instantiation mismatch
// (e.g., B_ADDR_W > NEU_AW).
localparam integer B_ADDR_W_EFF = (B_ADDR_W > NEU_AW) ? NEU_AW : B_ADDR_W;

reg [1:0]                   state;
reg [1:0]                   next_state;

reg [NEU_AW-1:0]            neu_base_r;
reg [NEU_AW-1:0]            out_dim_r;

reg                         bpre_phase;
reg [3:0]                   bpre_lane;
reg                         bpre_done;

reg                         w_phase;
reg [3:0]                   w_lane;
reg [IN_DIM_AW-1:0]         mac_feat;
reg                         mac_done_r;

reg signed [DATA_W-1:0]     bias_r [0:LANES-1];
reg signed [ACC_W-1:0]      acc_r [0:LANES-1];
reg signed [ACC_W-1:0]      acc_sat_r [0:LANES-1];

reg signed [2*DATA_W-1:0]   mac_prod;
reg signed [ACC_W-1:0]      acc_next;

wire                        feat_last;
wire                        lane_last;
wire [LANES-1:0]            lane_valid_w;

wire signed [DATA_W-1:0]    mac_x_op;

wire                        cs_en;
wire                        run_arm;
wire                        bpre_rom_a0;
wire                        bpre_rom_a1_last;
wire                        bpre_rom_a1_more;
wire                        w_rom_a0;
wire                        w_rom_a1;
wire                        w_rom_a1_lane_more;
wire                        w_rom_a1_feat_done;

wire [NEU_AW-1:0]           w_neu_idx;
wire [W_ADDR_W-1:0]         w_addr_calc;
wire [B_ADDR_W-1:0]         b_addr_calc;

assign busy          = (state != S_IDLE);
// CLK(clk) combinational addr: present current (lane,feat) so macro sampling at the
// issue->consume posedge returns Q during the consume phase (see header).
assign w_addr_o      = w_addr_calc;
assign b_addr_o      = b_addr_calc;
assign mac_feat_o    = mac_feat;
assign mac_active_o  = (state == S_RUN) && !mac_done_r;
assign x_consume_o   = w_rom_a1_feat_done;

assign feat_last = (mac_feat == FEAT_LAST);
assign lane_last = (w_lane == LANES - 1);

assign cs_en = 1'b1;

assign w_neu_idx   = neu_base_r + {{(NEU_AW-4){1'b0}}, w_lane};
assign w_addr_calc = w_neu_idx * IN_DIM + {{(W_ADDR_W-IN_DIM_AW){1'b0}}, mac_feat};
assign b_addr_calc = {{(B_ADDR_W-B_ADDR_W_EFF){1'b0}}, neu_base_r[B_ADDR_W_EFF-1:0]} +
                     {{(B_ADDR_W-4){1'b0}}, bpre_lane};

generate
    genvar gi;
    for (gi = 0; gi < LANES; gi = gi + 1) begin : gen_lane_valid
        assign lane_valid_w[gi] = (({1'b0, neu_base_r} + gi) < {1'b0, out_dim_r});
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
                next_state = S_BIAS;
        end
        S_BIAS: begin
            if (bpre_done)
                next_state = S_RUN;
        end
        S_RUN: begin
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
        mac_done <= (state == S_RUN) && mac_done_r;
end

assign run_arm = cs_en && (state == S_BIAS) && bpre_rom_a1_last;

assign bpre_rom_a0 = cs_en && (state == S_BIAS) && !bpre_done && (bpre_phase == 1'b0);

assign bpre_rom_a1_last = cs_en && (state == S_BIAS) && !bpre_done &&
                          (bpre_phase == 1'b1) && (bpre_lane == LANES - 1);

assign bpre_rom_a1_more = cs_en && (state == S_BIAS) && !bpre_done &&
                          (bpre_phase == 1'b1) && (bpre_lane != LANES - 1);

assign w_rom_a0 = cs_en && (state == S_RUN) && !mac_done_r && (w_phase == 1'b0);

assign w_rom_a1 = cs_en && (state == S_RUN) && !mac_done_r && (w_phase == 1'b1);

assign w_rom_a1_lane_more = w_rom_a1 && !lane_last;

assign w_rom_a1_feat_done = w_rom_a1 && lane_last && !feat_last;

// BIAS: 2-phase ROM prefetch (8 lanes)
always @(posedge clk) begin
    if (reset) begin
        bpre_phase <= 1'b0;
        bpre_lane  <= 4'd0;
        bpre_done  <= 1'b0;
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            bias_r[i_lane] <= {DATA_W{1'b0}};
    end else if (state == S_IDLE) begin
        bpre_phase <= 1'b0;
        bpre_lane  <= 4'd0;
        bpre_done  <= 1'b0;
    end else if (bpre_rom_a0) begin
        // b_addr_o combinational (= b_addr_calc); phase0 just advances to consume phase
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

// RUN: fused 2-phase weight ROM + per-lane accumulate
always @(posedge clk) begin
    if (reset) begin
        w_phase    <= 1'b0;
        w_lane     <= 4'd0;
        mac_feat   <= {IN_DIM_AW{1'b0}};
        mac_done_r <= 1'b0;
    end else if (run_arm) begin
        w_phase    <= 1'b0;
        w_lane     <= 4'd0;
        mac_feat   <= {IN_DIM_AW{1'b0}};
        mac_done_r <= 1'b0;
    end else if (w_rom_a0) begin
        // w_addr_o combinational (= w_addr_calc); phase0 just advances to consume phase
        w_phase  <= 1'b1;
    end else if (w_rom_a1_lane_more) begin
        w_phase <= 1'b0;
        w_lane  <= w_lane + 4'd1;
    end else if (w_rom_a1 && lane_last && !feat_last) begin
        w_phase  <= 1'b0;
        w_lane   <= 4'd0;
        mac_feat <= mac_feat + {{(IN_DIM_AW-1){1'b0}}, 1'b1};
    end else if (w_rom_a1 && lane_last && feat_last) begin
        mac_done_r <= 1'b1;
    end else if (state == S_IDLE) begin
        w_phase    <= 1'b0;
        w_lane     <= 4'd0;
        mac_feat   <= {IN_DIM_AW{1'b0}};
        mac_done_r <= 1'b0;
    end
end

// acc_r / acc_sat_r
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
            acc_r[i_lane]     <= {ACC_W{1'b0}};
            acc_sat_r[i_lane] <= {ACC_W{1'b0}};
        end
    end else if (run_arm) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
            acc_r[i_lane]     <= {ACC_W{1'b0}};
            acc_sat_r[i_lane] <= {ACC_W{1'b0}};
        end
    end else begin
        if (w_rom_a1 && lane_valid_w[w_lane])
            acc_r[w_lane] <= acc_next;
        if (w_rom_a1 && lane_last && feat_last) begin
            for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1) begin
                if (lane_valid_w[i_lane]) begin
                    if (i_lane == w_lane)
                        acc_sat_r[i_lane] <= acc_next;
                    else
                        acc_sat_r[i_lane] <= acc_r[i_lane];
                end
            end
        end
    end
end

// Combinational MAC for current lane (weight latched via ROM 2-phase on w_rom_a1)
always @(*) begin
    mac_prod = {2*DATA_W{1'b0}};
    acc_next = acc_r[w_lane];
    if (w_rom_a1 && lane_valid_w[w_lane]) begin
        mac_prod = mac_x_op * w_i;
        acc_next = acc_r[w_lane] + mac_prod;
    end
end

endmodule
