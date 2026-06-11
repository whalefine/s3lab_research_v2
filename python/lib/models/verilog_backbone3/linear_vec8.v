// =============================================================================
// linear_vec8.v
//
// Q8.8 linear layer with 8-wide MAC (vec_mac8). Drop-in for backbone2 linear.v /
// linear_wide.v port list (per-token: load x -> OUT_DIM neurons -> stream y).
//
// Per token:
//   S_LOAD      : capture IN_DIM features into x_buf (x_valid stream)
//   S_VMAC_ARM  : 1-cycle vec_mac8.start for current neu_base
//   S_VMAC_WAIT : vec_mac8 S_BIAS+S_RUN (stream ROM, fused 8-lane MAC);
//                 feed x_buf[mac_feat] when x_consume_o
//   S_SAT       : 8 beats serialize lane 0..7 (>>>8 + bias + sat16)
//   repeat neu_base += 8 until neu_base >= OUT_DIM
//
// w_addr_o (13b, matches backbone_top ROM local decode when IN_DIM is power-of-2):
//   IN_DIM=32  : {1'b0, neu[6:0], feat[4:0]}  == neu*32+feat
//   IN_DIM=128 : {1'b0, neu[4:0], feat[6:0]}  == neu*128+feat
//
// Golden-Weight: via parent w_addr_o decode (QKV / PROJ / FC1 / FC2).
// Saturation via wire only (no function).
// =============================================================================

module linear_vec8 #(
    parameter IN_DIM     = 32,
    parameter OUT_DIM    = 96,
    parameter LANES      = 8,
    parameter DATA_W     = 16,
    parameter ACC_W      = 32,
    parameter IN_DIM_AW  = 7,
    parameter NEU_AW     = 8,
    parameter W_ADDR_W   = 16,
    parameter B_ADDR_W   = 8
) (
    clk,
    reset,
    start,
    x_i,
    x_valid,
    w_i,
    b_i,
    w_addr_o,
    busy,
    done,
    y_o,
    y_valid,
    y_neu_o
);

input                       clk;
input                       reset;
input                       start;
input  signed [DATA_W-1:0]  x_i;
input                       x_valid;
input  signed [DATA_W-1:0]  w_i;
input  signed [DATA_W-1:0]  b_i;
output [12:0]               w_addr_o;
output                      busy;
output reg                  done;
output reg signed [DATA_W-1:0] y_o;
output reg                  y_valid;
output reg [6:0]            y_neu_o;

localparam [IN_DIM_AW-1:0] FEAT_LAST  = IN_DIM - 1;
localparam [8:0]           OUT_DIM_U = OUT_DIM;

parameter S_IDLE       = 3'd0;
parameter S_LOAD       = 3'd1;
parameter S_VMAC_ARM   = 3'd2;
parameter S_VMAC_WAIT  = 3'd3;
parameter S_SAT        = 3'd4;
parameter S_DONE       = 3'd5;

reg [2:0]                   state;
reg [2:0]                   next_state;

reg signed [DATA_W-1:0]     x_buf [0:IN_DIM-1];
reg [IN_DIM_AW-1:0]         load_cnt;
reg [NEU_AW-1:0]            neu_base_r;
reg [3:0]                   sat_lane;

reg signed [ACC_W-1:0]      acc_pick;
reg signed [DATA_W-1:0]     bias_pick;

wire                        vm_start;
wire                        vm_busy;
wire                        vm_mac_done;
wire [W_ADDR_W-1:0]         vm_w_addr;
wire [B_ADDR_W-1:0]         vm_b_addr;
wire                        vm_x_consume;
wire [IN_DIM_AW-1:0]        vm_mac_feat;
wire signed [ACC_W-1:0]     vm_acc_0;
wire signed [ACC_W-1:0]     vm_acc_1;
wire signed [ACC_W-1:0]     vm_acc_2;
wire signed [ACC_W-1:0]     vm_acc_3;
wire signed [ACC_W-1:0]     vm_acc_4;
wire signed [ACC_W-1:0]     vm_acc_5;
wire signed [ACC_W-1:0]     vm_acc_6;
wire signed [ACC_W-1:0]     vm_acc_7;
wire signed [DATA_W-1:0]    vm_bias_0;
wire signed [DATA_W-1:0]    vm_bias_1;
wire signed [DATA_W-1:0]    vm_bias_2;
wire signed [DATA_W-1:0]    vm_bias_3;
wire signed [DATA_W-1:0]    vm_bias_4;
wire signed [DATA_W-1:0]    vm_bias_5;
wire signed [DATA_W-1:0]    vm_bias_6;
wire signed [DATA_W-1:0]    vm_bias_7;
wire [LANES-1:0]            vm_lane_valid;

wire signed [DATA_W-1:0]    vm_x_feed;
wire                        load_last_beat;
wire                        sat_lane_valid;
wire                        more_neu_groups;
wire signed [ACC_W-1:0]     acc_shr8_w;
wire signed [ACC_W-1:0]     acc_plus_b_w;
wire signed [DATA_W-1:0]    y_sat_w;
wire [NEU_AW-1:0]           y_neu_idx_w;

assign load_last_beat = (load_cnt == FEAT_LAST) && x_valid;

assign vm_start   = (state == S_VMAC_ARM);
assign vm_x_feed  = x_buf[vm_mac_feat];

assign w_addr_o = vm_busy ? vm_w_addr[12:0] : 13'd0;

assign busy = (state != S_IDLE);

assign sat_lane_valid = (({1'b0, neu_base_r} + {5'b0, sat_lane}) < OUT_DIM_U);

assign more_neu_groups = (({1'b0, neu_base_r} + 9'd8) < OUT_DIM_U);

vec_mac8 #(
    .DATA_W    (DATA_W),
    .ACC_W     (ACC_W),
    .LANES     (LANES),
    .IN_DIM    (IN_DIM),
    .IN_DIM_AW (IN_DIM_AW),
    .NEU_AW    (NEU_AW),
    .W_ADDR_W  (W_ADDR_W),
    .B_ADDR_W  (B_ADDR_W)
) u_vec_mac8 (
    .clk          (clk),
    .reset        (reset),
    .start        (vm_start),
    .neu_base_i   (neu_base_r),
    .out_dim_i    (OUT_DIM[NEU_AW-1:0]),   // supports OUT_DIM up to 255
    .busy         (vm_busy),
    .mac_done     (vm_mac_done),
    .w_addr_o     (vm_w_addr),
    .b_addr_o     (vm_b_addr),
    .w_i          (w_i),
    .b_i          (b_i),
    .x_i          (vm_x_feed),
    .x_consume_o  (vm_x_consume),
    .mac_active_o (),
    .mac_feat_o   (vm_mac_feat),
    .acc_sat_0    (vm_acc_0),
    .acc_sat_1    (vm_acc_1),
    .acc_sat_2    (vm_acc_2),
    .acc_sat_3    (vm_acc_3),
    .acc_sat_4    (vm_acc_4),
    .acc_sat_5    (vm_acc_5),
    .acc_sat_6    (vm_acc_6),
    .acc_sat_7    (vm_acc_7),
    .bias_0       (vm_bias_0),
    .bias_1       (vm_bias_1),
    .bias_2       (vm_bias_2),
    .bias_3       (vm_bias_3),
    .bias_4       (vm_bias_4),
    .bias_5       (vm_bias_5),
    .bias_6       (vm_bias_6),
    .bias_7       (vm_bias_7),
    .lane_valid_o (vm_lane_valid)
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
            if (load_last_beat)
                next_state = S_VMAC_ARM;
        end
        S_VMAC_ARM: begin
            next_state = S_VMAC_WAIT;
        end
        S_VMAC_WAIT: begin
            if (vm_mac_done)
                next_state = S_SAT;
        end
        S_SAT: begin
            if (sat_lane == 4'd7) begin
                if (more_neu_groups)
                    next_state = S_VMAC_ARM;
                else
                    next_state = S_DONE;
            end
        end
        S_DONE: begin
            next_state = S_IDLE;
        end
        default: next_state = S_IDLE;
    endcase
end

// x_buf capture during S_LOAD
always @(posedge clk) begin
    if (state == S_LOAD && x_valid)
        x_buf[load_cnt] <= x_i;
end

// load_cnt
always @(posedge clk) begin
    if (reset)
        load_cnt <= {IN_DIM_AW{1'b0}};
    else if (state == S_IDLE)
        load_cnt <= {IN_DIM_AW{1'b0}};
    else if (state == S_LOAD && x_valid) begin
        if (load_cnt == FEAT_LAST)
            load_cnt <= {IN_DIM_AW{1'b0}};
        else
            load_cnt <= load_cnt + {{(IN_DIM_AW-1){1'b0}}, 1'b1};
    end
end

// neu_base_r: reset on LOAD entry; +=8 after each SAT group
always @(posedge clk) begin
    if (reset)
        neu_base_r <= {NEU_AW{1'b0}};
    else if (state == S_LOAD && load_last_beat)
        neu_base_r <= {NEU_AW{1'b0}};
    else if (state == S_SAT && sat_lane == 4'd7 && more_neu_groups)
        neu_base_r <= neu_base_r + {{(NEU_AW-4){1'b0}}, 4'd8};
end

// sat_lane
always @(posedge clk) begin
    if (reset)
        sat_lane <= 4'd0;
    else if (state == S_VMAC_WAIT && vm_mac_done)
        sat_lane <= 4'd0;
    else if (state == S_SAT && sat_lane != 4'd7)
        sat_lane <= sat_lane + 4'd1;
end

// done pulse
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE)
        done <= 1'b1;
    else
        done <= 1'b0;
end

// SAT: pick acc/bias by sat_lane (combinational)
always @(*) begin
    acc_pick  = {ACC_W{1'b0}};
    bias_pick = {DATA_W{1'b0}};
    case (sat_lane)
        4'd0: begin acc_pick = vm_acc_0; bias_pick = vm_bias_0; end
        4'd1: begin acc_pick = vm_acc_1; bias_pick = vm_bias_1; end
        4'd2: begin acc_pick = vm_acc_2; bias_pick = vm_bias_2; end
        4'd3: begin acc_pick = vm_acc_3; bias_pick = vm_bias_3; end
        4'd4: begin acc_pick = vm_acc_4; bias_pick = vm_bias_4; end
        4'd5: begin acc_pick = vm_acc_5; bias_pick = vm_bias_5; end
        4'd6: begin acc_pick = vm_acc_6; bias_pick = vm_bias_6; end
        4'd7: begin acc_pick = vm_acc_7; bias_pick = vm_bias_7; end
        default: ;
    endcase
end

assign acc_shr8_w  = acc_pick >>> 8;
assign acc_plus_b_w = acc_shr8_w + $signed({{(ACC_W-DATA_W){bias_pick[DATA_W-1]}}, bias_pick});

assign y_sat_w =
    (acc_plus_b_w > 32'sd32767) ? 16'sh7FFF :
    (acc_plus_b_w < -32'sh8000) ? 16'sh8000 :
    acc_plus_b_w[DATA_W-1:0];

assign y_neu_idx_w = neu_base_r + {{(NEU_AW-4){1'b0}}, sat_lane};

// y_neu_o port is 7b; y_neu_idx_w is NEU_AW wide. Do not slice [6:0] when NEU_AW<7
// (VCS marks bit 6 X, same class of bug as backbone2 at_n_reg[10:0]).
wire [6:0] y_neu_o_w;
generate
    if (NEU_AW >= 7) begin : g_y_neu_slice
        assign y_neu_o_w = y_neu_idx_w[6:0];
    end else begin : g_y_neu_pad
        assign y_neu_o_w = {{(7-NEU_AW){1'b0}}, y_neu_idx_w[NEU_AW-1:0]};
    end
endgenerate

// y_o / y_valid / y_neu_o during S_SAT
always @(posedge clk) begin
    if (reset) begin
        y_o     <= {DATA_W{1'b0}};
        y_valid <= 1'b0;
        y_neu_o <= 7'd0;
    end else begin
        y_valid <= 1'b0;
        if (state == S_SAT && sat_lane_valid) begin
            y_o     <= y_sat_w;
            y_neu_o <= y_neu_o_w;
            y_valid <= 1'b1;
        end
    end
end

endmodule
