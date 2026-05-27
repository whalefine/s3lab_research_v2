// =============================================================================
// head_top.v -- verilog_head2 head (conv1 + conv2 + tail + cal_bbox)
// -----------------------------------------------------------------------------
// numpy: head_shared_trunk() + cal_bbox() in run_backbone_numpy_shared_trunk.py
// Input: backbone token stream ; search -> x_buf[8192] ; Golden: bbox + activations
// Weights in conv.v / tail.v ROM (memory/ at compile)
// sh1 (conv1 out / conv2 in): Sram_q lo + Sram_k hi (12288x16 each, A[13:0] per macro)
//   bank = flat >= SH1_HALF(12288); local = flat[13:0] or flat[13:0]-12288
// Golden: Activation/box_head_shared_after_conv1_out_bi.txt
// SRAM 1P CLK(~clk): posedge T A/WEB=1 -> posedge T+1 Q; conv2 .x_i(sh1_rd_q) on MAC phase1
// Write: latch on c1_y_valid, drive WEB=0 next posedge (was negedge reg capture)
// =============================================================================

module head_top #(
    parameter IN_CH    = 32,
    parameter C_SH1    = 96,
    parameter C_SH2    = 48,
    parameter FEAT_H   = 16,
    parameter FEAT_W   = 16,
    parameter N_TOKENS = 320,
    parameter LENS_Z   = 64,
    parameter DATA_W   = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    input  wire signed [DATA_W-1:0] a_i,
    input  wire                     a_valid,

    output wire        busy,
    output reg         done,

    output wire [DATA_W-1:0] cx_o,
    output wire [DATA_W-1:0] cy_o,
    output wire [DATA_W-1:0] w_o,
    output wire [DATA_W-1:0] h_o
);

localparam FEAT_SZ     = FEAT_H * FEAT_W;
localparam SKIP_VALS   = LENS_Z * IN_CH;
localparam TOT_VALS    = N_TOKENS * IN_CH;
localparam TOT_VALS_M1 = TOT_VALS - 1;
localparam IN_LEN_HEAD = FEAT_SZ * IN_CH;
localparam C1_LEN      = C_SH1 * FEAT_SZ;
localparam C2_LEN      = C_SH2 * FEAT_SZ;
localparam SH1_HALF    = C1_LEN >> 1;

wire rst_n = ~reset;

parameter S_IDLE  = 3'd0;
parameter S_FILL  = 3'd1;
parameter S_CONV1 = 3'd2;
parameter S_CONV2 = 3'd3;
parameter S_TAIL  = 3'd4;
parameter S_BBOX  = 3'd5;
parameter S_DONE  = 3'd6;

reg [2:0] CS, NS;

reg [13:0] fill_cnt;
wire [13:0] fill_off    = fill_cnt - SKIP_VALS;
wire [7:0]  fill_n      = fill_off[12:5];
wire [4:0]  fill_c      = fill_off[4:0];
wire        fill_search = (fill_cnt >= SKIP_VALS) && (fill_cnt < TOT_VALS);
wire [13:0] fill_dst    = {fill_c, fill_n};

reg [DATA_W-1:0] x_buf   [0:IN_LEN_HEAD-1];
reg [DATA_W-1:0] sh2_buf [0:C2_LEN-1];
reg [DATA_W-1:0] bbox_reg [0:3];

reg c1_start, c2_start, t_start, b_start;
reg c1_started, c2_started, t_started, b_started;

wire              c1_busy, c1_done, c1_y_valid;
wire [DATA_W-1:0] c1_y_data;
wire [13:0]       c1_x_addr;
reg  [DATA_W-1:0] c1_x_i_q;

wire              c2_busy, c2_done, c2_y_valid;
wire [DATA_W-1:0] c2_y_data;
wire [14:0]       c2_x_addr;
wire              c2_mac_x_phase;
wire              c1_mac_x_phase_unused;

reg [14:0] c1_wr_idx;
reg        c1_wr_do;
reg [14:0] c1_wr_idx_lat;
reg [15:0] c1_wr_din_lat;
reg [31:0] c2_wr_idx;

wire              t_busy, t_done;
wire [14:0]       t_x_addr;
reg  [DATA_W-1:0] t_x_i_q;
wire              tc_sig_v, to_v, ts_sig_v;
wire [DATA_W-1:0] tc_sig_d, to_d, ts_sig_d;

wire              b_busy, b_done, b_valid;
wire [DATA_W-1:0] b_data;
wire [1:0]        b_idx;

reg        s1q_ceb, s1q_web, s1k_ceb, s1k_web;
reg [13:0] s1_addr;
reg [15:0] s1_din;
wire [15:0] s1q_q, s1k_q;
wire [15:0] sh1_rd_q;
wire        sh1_wr_en;
wire        sh1_rd_en;
wire [14:0] sh1_wr_flat;
wire [14:0] sh1_rd_flat;
wire        sh1_wr_bank;
wire        sh1_rd_bank;
wire [13:0] sh1_wr_local;
wire [13:0] sh1_rd_local;

assign sh1_wr_en    = c1_wr_do;
assign sh1_wr_flat  = c1_wr_idx_lat;
assign sh1_wr_bank  = (sh1_wr_flat >= SH1_HALF);
assign sh1_wr_local = sh1_wr_bank ? (sh1_wr_flat[13:0] - SH1_HALF[13:0]) :
                                     sh1_wr_flat[13:0];

assign sh1_rd_en    = (CS == S_CONV2) && c2_busy && !c2_mac_x_phase;
assign sh1_rd_flat  = c2_x_addr;
assign sh1_rd_bank  = (sh1_rd_flat >= SH1_HALF);
assign sh1_rd_local = sh1_rd_bank ? (sh1_rd_flat[13:0] - SH1_HALF[13:0]) :
                                     sh1_rd_flat[13:0];

assign sh1_rd_q     = sh1_rd_bank ? s1k_q : s1q_q;

Sram_q u_sh1_sram_q (
    .SLP    (1'b0),
    .DSLP   (1'b0),
    .SD     (1'b0),
    .PUDELAY(),
    .CLK    (~clk),
    .CEB    (s1q_ceb),
    .WEB    (s1q_web),
    .BIST   (1'b0),
    .CEBM   (),
    .WEBM   (),
    .A      (s1_addr),
    .D      (s1_din),
    .BWEB   (16'b0),
    .AM     (),
    .DM     (),
    .BWEBM  (16'b0),
    .RTSEL  (2'b01),
    .WTSEL  (2'b00),
    .Q      (s1q_q)
);

Sram_k u_sh1_sram_k (
    .SLP    (1'b0),
    .DSLP   (1'b0),
    .SD     (1'b0),
    .PUDELAY(),
    .CLK    (~clk),
    .CEB    (s1k_ceb),
    .WEB    (s1k_web),
    .BIST   (1'b0),
    .CEBM   (),
    .WEBM   (),
    .A      (s1_addr),
    .D      (s1_din),
    .BWEB   (16'b0),
    .AM     (),
    .DM     (),
    .BWEBM  (16'b0),
    .RTSEL  (2'b01),
    .WTSEL  (2'b00),
    .Q      (s1k_q)
);

// x_buf / sh2_buf read (negedge sample)
always @(negedge clk)
    c1_x_i_q <= x_buf[c1_x_addr];

always @(negedge clk)
    t_x_i_q <= sh2_buf[t_x_addr];

conv #(
    .IN_CH       (IN_CH   ),
    .OUT_CH      (C_SH1   ),
    .IN_H        (FEAT_H  ),
    .IN_W        (FEAT_W  ),
    .K           (3       ),
    .PAD         (1       ),
    .HAS_RELU    (1       ),
    .DATA_W      (DATA_W  ),
    .FRAC_W      (8       ),
    .ACC_W       (32      ),
    .ROM_PROFILE (1       ),
    .X_AW        (14      )
) u_conv1 (
    .clk     (clk       ),
    .rst_n   (rst_n     ),
    .start   (c1_start  ),
    .busy    (c1_busy   ),
    .done    (c1_done   ),
    .x_addr  (c1_x_addr ),
    .x_i     (c1_x_i_q  ),
    .y_valid (c1_y_valid),
    .y_data  (c1_y_data ),
    .y_oc    (           ),
    .y_oh    (           ),
    .y_ow    (           ),
    .mac_x_phase (c1_mac_x_phase_unused)
);

conv #(
    .IN_CH       (C_SH1   ),
    .OUT_CH      (C_SH2   ),
    .IN_H        (FEAT_H  ),
    .IN_W        (FEAT_W  ),
    .K           (3       ),
    .PAD         (1       ),
    .HAS_RELU    (1       ),
    .DATA_W      (DATA_W  ),
    .FRAC_W      (8       ),
    .ACC_W       (32      ),
    .ROM_PROFILE (2       ),
    .X_AW        (15      )
) u_conv2 (
    .clk     (clk       ),
    .rst_n   (rst_n     ),
    .start   (c2_start  ),
    .busy    (c2_busy   ),
    .done    (c2_done   ),
    .x_addr  (c2_x_addr ),
    .x_i     (sh1_rd_q  ),
    .y_valid (c2_y_valid),
    .y_data  (c2_y_data ),
    .y_oc    (           ),
    .y_oh    (           ),
    .y_ow    (           ),
    .mac_x_phase (c2_mac_x_phase)
);

// sh1 SRAM port mux (1P: one bank read or write per cycle)
always @(*) begin
    s1q_ceb  = 1'b1;
    s1q_web  = 1'b1;
    s1k_ceb  = 1'b1;
    s1k_web  = 1'b1;
    s1_addr  = 14'd0;
    s1_din   = 16'd0;

    if (sh1_wr_en) begin
        s1_addr = sh1_wr_local;
        s1_din  = c1_wr_din_lat;
        if (sh1_wr_bank) begin
            s1k_ceb = 1'b0;
            s1k_web = 1'b0;
        end else begin
            s1q_ceb = 1'b0;
            s1q_web = 1'b0;
        end
    end else if (sh1_rd_en) begin
        s1_addr = sh1_rd_local;
        if (sh1_rd_bank) begin
            s1k_ceb = 1'b0;
            s1k_web = 1'b1;
        end else begin
            s1q_ceb = 1'b0;
            s1q_web = 1'b1;
        end
    end
end

// sh1 SRAM write capture + index (1-cycle delayed write vs c1_y_valid)
always @(posedge clk) begin
    if (reset) begin
        c1_wr_idx     <= 15'd0;
        c1_wr_do      <= 1'b0;
        c1_wr_idx_lat <= 15'd0;
        c1_wr_din_lat <= 16'd0;
    end else begin
        c1_wr_do <= c1_y_valid && (CS == S_CONV1);
        if (c1_y_valid && (CS == S_CONV1)) begin
            c1_wr_idx_lat <= c1_wr_idx;
            c1_wr_din_lat <= c1_y_data;
            c1_wr_idx     <= c1_wr_idx + 15'd1;
        end
    end
end

// sh2_buf capture (negedge, same as conv ROM data tick)
always @(negedge clk) begin
    if (!rst_n)
        c2_wr_idx <= 32'd0;
    else if (c2_y_valid) begin
        sh2_buf[c2_wr_idx] <= c2_y_data;
        c2_wr_idx <= c2_wr_idx + 32'd1;
    end
end

tail #(
    .DATA_W (DATA_W),
    .X_AW   (15    )
) u_tail (
    .clk              (clk      ),
    .rst_n            (rst_n    ),
    .start            (t_start  ),
    .busy             (t_busy   ),
    .done             (t_done   ),
    .x_addr           (t_x_addr ),
    .x_i              (t_x_i_q  ),
    .ctr_raw_y_valid  (         ),
    .ctr_raw_y_data   (         ),
    .ctr_raw_y_oh     (         ),
    .ctr_raw_y_ow     (         ),
    .ctr_y_valid      (tc_sig_v ),
    .ctr_y_data       (tc_sig_d ),
    .off_y_valid      (to_v     ),
    .off_y_data       (to_d     ),
    .off_y_sub        (         ),
    .off_y_oh         (         ),
    .off_y_ow         (         ),
    .size_raw_y_valid (         ),
    .size_raw_y_data  (         ),
    .size_raw_y_sub   (         ),
    .size_raw_y_oh    (         ),
    .size_raw_y_ow    (         ),
    .size_y_valid     (ts_sig_v ),
    .size_y_data      (ts_sig_d )
);

cal_bbox #(.DATA_W(DATA_W)) u_bbox (
    .clk           (clk     ),
    .rst_n         (rst_n   ),
    .start         (b_start ),
    .busy          (b_busy  ),
    .done          (b_done  ),
    .ctr_in_valid  (tc_sig_v),
    .ctr_in_data   (tc_sig_d),
    .size_in_valid (ts_sig_v),
    .size_in_data  (ts_sig_d),
    .size_in_sub   (1'b0    ),
    .off_in_valid  (to_v    ),
    .off_in_data   (to_d    ),
    .off_in_sub    (1'b0    ),
    .bbox_valid    (b_valid ),
    .bbox_data     (b_data  ),
    .bbox_idx      (b_idx   )
);

// FSM CS
always @(posedge clk) begin
    if (reset)
        CS <= S_IDLE;
    else
        CS <= NS;
end

// FSM NS
always @(*) begin
    NS = CS;
    case (CS)
        S_IDLE:  NS = start ? S_FILL : S_IDLE;
        S_FILL:  NS = (a_valid && (fill_cnt == TOT_VALS_M1)) ? S_CONV1 : S_FILL;
        S_CONV1: NS = c1_done ? S_CONV2 : S_CONV1;
        S_CONV2: NS = c2_done ? S_TAIL : S_CONV2;
        S_TAIL:  NS = t_done ? S_BBOX : S_TAIL;
        S_BBOX:  NS = b_done ? S_DONE : S_BBOX;
        S_DONE:  NS = S_IDLE;
        default: NS = S_IDLE;
    endcase
end

// fill_cnt, x_buf
always @(posedge clk) begin
    if (reset)
        fill_cnt <= 14'd0;
    else if (CS == S_IDLE)
        fill_cnt <= 14'd0;
    else if (CS == S_FILL && a_valid) begin
        if (fill_search)
            x_buf[fill_dst] <= a_i[DATA_W-1:0];
        if (fill_cnt < TOT_VALS)
            fill_cnt <= fill_cnt + 14'd1;
    end
end

// done, submodule starts, started flags, bbox_reg
always @(posedge clk) begin
    done     <= 1'b0;
    c1_start <= 1'b0;
    c2_start <= 1'b0;
    t_start  <= 1'b0;
    b_start  <= 1'b0;

    if (reset) begin
        c1_started <= 1'b0;
        c2_started <= 1'b0;
        t_started  <= 1'b0;
        b_started  <= 1'b0;
        c1_wr_idx  <= 15'd0;
        c2_wr_idx  <= 32'd0;
    end else begin
        case (CS)
            S_IDLE: begin
                c1_started <= 1'b0;
                c2_started <= 1'b0;
                t_started  <= 1'b0;
                b_started  <= 1'b0;
            end

            S_CONV1: begin
                if (!c1_started) begin
                    c1_start   <= 1'b1;
                    c1_started <= 1'b1;
                    c1_wr_idx  <= 15'd0;
                end
            end

            S_CONV2: begin
                if (!c2_started && !c2_busy) begin
                    c2_start   <= 1'b1;
                    c2_started <= 1'b1;
                    c2_wr_idx  <= 32'd0;
                end
            end

            S_TAIL: begin
                if (!t_started && !t_busy) begin
                    t_start   <= 1'b1;
                    t_started <= 1'b1;
                end
            end

            S_BBOX: begin
                if (!b_started && !b_busy) begin
                    b_start   <= 1'b1;
                    b_started <= 1'b1;
                end
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase

        if (b_valid)
            bbox_reg[b_idx] <= b_data;
    end
end

assign busy = (CS != S_IDLE) && (CS != S_DONE);
assign cx_o = bbox_reg[0];
assign cy_o = bbox_reg[1];
assign w_o  = bbox_reg[2];
assign h_o  = bbox_reg[3];

endmodule
