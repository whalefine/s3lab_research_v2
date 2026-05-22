// =============================================================================
// TEST_head.v -- verilog_head2 : conv1 + conv2 + tail + cal_bbox
//   Input  : backbone_after_norm (reshape to head NCHW)
//   Check  : box_head_after_cal_bbox_bbox_bi.txt (4 x Q8.8) only
// =============================================================================

`timescale 1ns/1ps

`ifndef GOLDEN_ACT
`define GOLDEN_ACT "./TXT_File/Activation"
`endif

module TEST_head ;

parameter DATA_W       = 16 ;
parameter IN_LEN_RAW   = 10240 ;
parameter IN_LEN_HEAD  = 8192  ;
parameter C1_LEN       = 24576 ;
parameter C2_LEN       = 12288 ;
parameter BBOX_LEN     = 4     ;
parameter SKIP_TOKENS  = 64 ;
parameter EMBED_DIM    = 32 ;
parameter FEAT_SZ      = 16 ;
parameter OC_CONV1     = 96 ;
parameter OC_CONV2     = 48 ;
parameter IN_CH_CONV1  = 32 ;
parameter IN_CH_CONV2  = 96 ;

reg clk ;
reg rst_n ;

initial begin
    clk = 1'b0 ;
end
always #5 clk = ~clk ;

reg [DATA_W-1:0] raw_in    [0:IN_LEN_RAW-1] ;
reg [DATA_W-1:0] x_buf     [0:IN_LEN_HEAD-1] ;
reg [DATA_W-1:0] sh1_buf   [0:C1_LEN-1] ;
reg [DATA_W-1:0] sh2_buf   [0:C2_LEN-1] ;
reg [DATA_W-1:0] bbox_gold [0:BBOX_LEN-1] ;
reg [DATA_W-1:0] bbox_out  [0:BBOX_LEN-1] ;

// conv1
wire [13:0]       c1_x_addr ;
reg  [DATA_W-1:0] c1_x_i_q ;
always @(negedge clk)
    c1_x_i_q <= x_buf[c1_x_addr] ;

wire              c1_busy, c1_done, c1_y_valid ;
wire [DATA_W-1:0] c1_y_data ;
reg               c1_start ;

conv #(
    .IN_CH       (IN_CH_CONV1),
    .OUT_CH      (OC_CONV1   ),
    .IN_H        (FEAT_SZ    ),
    .IN_W        (FEAT_SZ    ),
    .K           (3          ),
    .PAD         (1          ),
    .HAS_RELU    (1          ),
    .DATA_W      (DATA_W     ),
    .FRAC_W      (8          ),
    .ACC_W       (32         ),
    .ROM_PROFILE (1          ),
    .X_AW        (14         )
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

// conv2
wire [14:0]       c2_x_addr ;
reg  [DATA_W-1:0] c2_x_i_q ;
always @(negedge clk)
    c2_x_i_q <= sh1_buf[c2_x_addr] ;

wire              c2_busy, c2_done, c2_y_valid ;
wire [DATA_W-1:0] c2_y_data ;
reg               c2_start ;

conv #(
    .IN_CH       (IN_CH_CONV2),
    .OUT_CH      (OC_CONV2   ),
    .IN_H        (FEAT_SZ    ),
    .IN_W        (FEAT_SZ    ),
    .K           (3          ),
    .PAD         (1          ),
    .HAS_RELU    (1          ),
    .DATA_W      (DATA_W     ),
    .FRAC_W      (8          ),
    .ACC_W       (32         ),
    .ROM_PROFILE (2          ),
    .X_AW        (15         )
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

reg [31:0] c1_wr_idx ;
always @(negedge clk) begin
    if (!rst_n)
        c1_wr_idx <= 0 ;
    else if (c1_y_valid) begin
        sh1_buf[c1_wr_idx] <= c1_y_data ;
        c1_wr_idx <= c1_wr_idx + 1 ;
    end
end

reg [31:0] c2_wr_idx ;
always @(negedge clk) begin
    if (!rst_n)
        c2_wr_idx <= 0 ;
    else if (c2_y_valid) begin
        sh2_buf[c2_wr_idx] <= c2_y_data ;
        c2_wr_idx <= c2_wr_idx + 1 ;
    end
end

// tail
wire [14:0]       t_x_addr ;
reg  [DATA_W-1:0] t_x_i_q ;
always @(negedge clk)
    t_x_i_q <= sh2_buf[t_x_addr] ;

wire              t_busy, t_done ;
reg               t_start ;
wire              tc_raw_v, tc_sig_v, to_v, ts_raw_v, ts_sig_v ;
wire [DATA_W-1:0] tc_raw_d, tc_sig_d, to_d, ts_raw_d, ts_sig_d ;
wire [4:0]        tc_raw_oh, tc_raw_ow, to_oh, to_ow, ts_raw_oh, ts_raw_ow ;
wire              to_sub, ts_raw_sub ;

tail #(
    .DATA_W (DATA_W),
    .X_AW   (15    )
) u_tail (
    .clk              (clk       ),
    .rst_n            (rst_n     ),
    .start            (t_start   ),
    .busy             (t_busy    ),
    .done             (t_done    ),
    .x_addr           (t_x_addr  ),
    .x_i              (t_x_i_q   ),
    .ctr_raw_y_valid  (tc_raw_v  ),
    .ctr_raw_y_data   (tc_raw_d  ),
    .ctr_raw_y_oh     (tc_raw_oh ),
    .ctr_raw_y_ow     (tc_raw_ow ),
    .ctr_y_valid      (tc_sig_v  ),
    .ctr_y_data       (tc_sig_d  ),
    .off_y_valid      (to_v      ),
    .off_y_data       (to_d      ),
    .off_y_sub        (to_sub    ),
    .off_y_oh         (to_oh     ),
    .off_y_ow         (to_ow     ),
    .size_raw_y_valid (ts_raw_v  ),
    .size_raw_y_data  (ts_raw_d  ),
    .size_raw_y_sub   (ts_raw_sub),
    .size_raw_y_oh    (ts_raw_oh ),
    .size_raw_y_ow    (ts_raw_ow ),
    .size_y_valid     (ts_sig_v  ),
    .size_y_data      (ts_sig_d  )
);

// cal_bbox
reg               b_start ;
wire              b_busy, b_done, b_valid ;
wire [DATA_W-1:0] b_data ;
wire [1:0]        b_idx ;

cal_bbox #(.DATA_W(DATA_W)) u_bbox (
    .clk           (clk      ),
    .rst_n         (rst_n    ),
    .start         (b_start  ),
    .busy          (b_busy   ),
    .done          (b_done   ),
    .ctr_in_valid  (tc_sig_v ),
    .ctr_in_data   (tc_sig_d ),
    .size_in_valid (ts_sig_v ),
    .size_in_data  (ts_sig_d ),
    .size_in_sub   (1'b0     ),
    .off_in_valid  (to_v     ),
    .off_in_data   (to_d     ),
    .off_in_sub    (1'b0     ),
    .bbox_valid    (b_valid  ),
    .bbox_data     (b_data   ),
    .bbox_idx      (b_idx    )
);

reg [31:0] bb_fail_cnt ;
reg [31:0] bb_cmp_idx ;

always @(posedge clk) begin
    if (!rst_n) begin
        bb_fail_cnt <= 0 ;
        bb_cmp_idx  <= 0 ;
    end else if (b_valid) begin
        bbox_out[bb_cmp_idx] <= b_data ;
        if (b_data !== bbox_gold[bb_cmp_idx])
            bb_fail_cnt <= bb_fail_cnt + 1 ;
        bb_cmp_idx <= bb_cmp_idx + 1 ;
    end
end

// stimulus
integer c, h, w ;
integer src_idx, dst_idx ;

initial begin
    $readmemb({`GOLDEN_ACT, "/backbone_after_norm_backbone_out_bi.txt"}, raw_in    ) ;
    $readmemb({`GOLDEN_ACT, "/box_head_after_cal_bbox_bbox_bi.txt"     }, bbox_gold) ;

    for (c = 0; c < EMBED_DIM; c = c + 1) begin
        for (h = 0; h < FEAT_SZ; h = h + 1) begin
            for (w = 0; w < FEAT_SZ; w = w + 1) begin
                src_idx = (SKIP_TOKENS + h * FEAT_SZ + w) * EMBED_DIM + c ;
                dst_idx = c * FEAT_SZ * FEAT_SZ + h * FEAT_SZ + w ;
                x_buf[dst_idx] = raw_in[src_idx] ;
            end
        end
    end

    rst_n    = 1'b0 ;
    c1_start = 1'b0 ;
    c2_start = 1'b0 ;
    t_start  = 1'b0 ;
    b_start  = 1'b0 ;
    #25 ;
    @(posedge clk) ;
    rst_n = 1'b1 ;
    @(posedge clk) ;
    @(posedge clk) ;

    @(posedge clk) ;
    c1_start = 1'b1 ;
    @(posedge clk) ;
    c1_start = 1'b0 ;
    @(posedge c1_done) ;

    while (c2_busy)
        @(posedge clk) ;
    @(posedge clk) ;
    c2_start = 1'b1 ;
    @(posedge clk) ;
    c2_start = 1'b0 ;
    @(posedge c2_done) ;

    while (t_busy)
        @(posedge clk) ;
    @(posedge clk) ;
    t_start = 1'b1 ;
    @(posedge clk) ;
    t_start = 1'b0 ;
    @(posedge t_done) ;

    while (b_busy)
        @(posedge clk) ;
    @(posedge clk) ;
    b_start = 1'b1 ;
    @(posedge clk) ;
    b_start = 1'b0 ;
    @(posedge b_done) ;

    if (bb_fail_cnt == 0 && bb_cmp_idx == BBOX_LEN)
        $display("[BBOX] PASS cx=%h cy=%h w=%h h=%h (golden match)",
                 bbox_out[0], bbox_out[1], bbox_out[2], bbox_out[3]) ;
    else
        $display("[BBOX] FAIL cx got=%h exp=%h cy got=%h exp=%h w got=%h exp=%h h got=%h exp=%h",
                 bbox_out[0], bbox_gold[0], bbox_out[1], bbox_gold[1],
                 bbox_out[2], bbox_gold[2], bbox_out[3], bbox_gold[3]) ;

    $finish ;
end

initial begin
    #2_000_000_000 ;
    $display("[BBOX] TIMEOUT") ;
    $finish ;
end

endmodule
