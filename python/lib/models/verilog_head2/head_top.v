// =============================================================================
// head_top.v -- verilog_head2 head wrapper (conv1 + conv2 + tail + cal_bbox)
//
// Development copy: edit here first (e.g. SRAM buffers), then sync to verilog2/.
//
// Maps to numpy: head_shared_trunk() + cal_bbox() in run_backbone_numpy_shared_trunk.py
//
// Input: backbone norm token stream (template 2048 + search 8192, Q8.8).
//   Only search tokens are reshaped to NCHW x_buf[8192] (skip first 2048).
//   Golden fill reference: backbone_after_norm_backbone_out_bi.txt
//
// Weights: internal ROM in conv.v / tail.v (memory/ at compile). No TXT_File/Weight in RTL.
//
// Output bbox Q8.8:
//   Golden: Activation/box_head_after_cal_bbox_bbox_bi.txt
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

wire rst_n = ~reset;

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
parameter S_IDLE  = 3'd0;
parameter S_FILL  = 3'd1;
parameter S_CONV1 = 3'd2;
parameter S_CONV2 = 3'd3;
parameter S_TAIL  = 3'd4;
parameter S_BBOX  = 3'd5;
parameter S_DONE  = 3'd6;

reg [2:0] state, next_state;

reg [13:0] fill_cnt;
wire [13:0] fill_off    = fill_cnt - SKIP_VALS;
wire [7:0]  fill_n      = fill_off[12:5];
wire [4:0]  fill_c      = fill_off[4:0];
wire        fill_search = (fill_cnt >= SKIP_VALS) && (fill_cnt < TOT_VALS);
wire [13:0] fill_dst    = {fill_c, fill_n};

// Feature buffers (search NCHW + conv outputs)
reg [DATA_W-1:0] x_buf   [0:IN_LEN_HEAD-1];
reg [DATA_W-1:0] sh1_buf [0:C1_LEN-1];
reg [DATA_W-1:0] sh2_buf [0:C2_LEN-1];

reg [DATA_W-1:0] bbox_reg [0:3];

// Submodule controls
reg c1_start, c2_start, t_start, b_start;
reg c1_started, c2_started, t_started, b_started;

wire              c1_busy, c1_done, c1_y_valid;
wire [DATA_W-1:0] c1_y_data;
wire [13:0]       c1_x_addr;
reg  [DATA_W-1:0] c1_x_i_q;

always @(negedge clk)
    c1_x_i_q <= x_buf[c1_x_addr];

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
    .y_ow    (           )
);

wire              c2_busy, c2_done, c2_y_valid;
wire [DATA_W-1:0] c2_y_data;
wire [14:0]       c2_x_addr;
reg  [DATA_W-1:0] c2_x_i_q;

always @(negedge clk)
    c2_x_i_q <= sh1_buf[c2_x_addr];

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
    .x_i     (c2_x_i_q  ),
    .y_valid (c2_y_valid),
    .y_data  (c2_y_data ),
    .y_oc    (           ),
    .y_oh    (           ),
    .y_ow    (           )
);

reg [31:0] c1_wr_idx;
always @(negedge clk) begin
    if (!rst_n)
        c1_wr_idx <= 32'd0;
    else if (c1_y_valid) begin
        sh1_buf[c1_wr_idx] <= c1_y_data;
        c1_wr_idx <= c1_wr_idx + 32'd1;
    end
end

reg [31:0] c2_wr_idx;
always @(negedge clk) begin
    if (!rst_n)
        c2_wr_idx <= 32'd0;
    else if (c2_y_valid) begin
        sh2_buf[c2_wr_idx] <= c2_y_data;
        c2_wr_idx <= c2_wr_idx + 32'd1;
    end
end

wire              t_busy, t_done;
wire [14:0]       t_x_addr;
reg  [DATA_W-1:0] t_x_i_q;
wire              tc_sig_v, to_v, ts_sig_v;
wire [DATA_W-1:0] tc_sig_d, to_d, ts_sig_d;

always @(negedge clk)
    t_x_i_q <= sh2_buf[t_x_addr];

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

wire              b_busy, b_done, b_valid;
wire [DATA_W-1:0] b_data;
wire [1:0]        b_idx;

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

// ---------------------------------------------------------------------------
// FSM segment 1: state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// ---------------------------------------------------------------------------
// FSM segment 2: next-state logic
// ---------------------------------------------------------------------------
always @(*) begin
    case (state)
        S_IDLE:  next_state = start ? S_FILL : S_IDLE;
        S_FILL:  next_state = (a_valid && (fill_cnt == TOT_VALS_M1)) ? S_CONV1 : S_FILL;
        S_CONV1: next_state = c1_done ? S_CONV2 : S_CONV1;
        S_CONV2: next_state = c2_done ? S_TAIL : S_CONV2;
        S_TAIL:  next_state = t_done ? S_BBOX : S_TAIL;
        S_BBOX:  next_state = b_done ? S_DONE : S_BBOX;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// ---------------------------------------------------------------------------
// FSM segment 3: output / fill / submodule starts
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    done        <= 1'b0;
    c1_start    <= 1'b0;
    c2_start    <= 1'b0;
    t_start     <= 1'b0;
    b_start     <= 1'b0;

    if (reset) begin
        fill_cnt    <= 14'd0;
        c1_started  <= 1'b0;
        c2_started  <= 1'b0;
        t_started   <= 1'b0;
        b_started   <= 1'b0;
        c1_wr_idx   <= 32'd0;
        c2_wr_idx   <= 32'd0;
    end else begin
        case (state)
            S_IDLE: begin
                fill_cnt   <= 14'd0;
                c1_started <= 1'b0;
                c2_started <= 1'b0;
                t_started  <= 1'b0;
                b_started  <= 1'b0;
            end

            S_FILL: begin
                if (a_valid) begin
                    if (fill_search)
                        x_buf[fill_dst] <= a_i[DATA_W-1:0];
                    if (fill_cnt < TOT_VALS)
                        fill_cnt <= fill_cnt + 14'd1;
                end
            end

            S_CONV1: begin
                if (!c1_started) begin
                    c1_start   <= 1'b1;
                    c1_started <= 1'b1;
                    c1_wr_idx  <= 32'd0;
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

assign busy = (state != S_IDLE) && (state != S_DONE);
assign cx_o = bbox_reg[0];
assign cy_o = bbox_reg[1];
assign w_o  = bbox_reg[2];
assign h_o  = bbox_reg[3];

endmodule
