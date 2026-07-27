// =============================================================================
// cal_bbox.v  (verilog_dim256_head)
// -----------------------------------------------------------------------------
// numpy cal_bbox (FEAT_SZ=16):
//   idx = argmax(score); idx_y=idx/16, idx_x=idx%16
//   cx = ((idx_x<<8) + off_x) >>> 4
//   cy = ((idx_y<<8) + off_y) >>> 4
//   w,h from size map at idx
// Map layout NCHW flat: addr = ch * 256 + spatial
// Golden: box_head_after_forward_head_pred_boxes_bi.txt
// =============================================================================

module cal_bbox #(
    parameter DATA_W  = 16,
    parameter FEAT_SZ = 16,
    parameter X_AW    = 10
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        busy,
    output reg         done,

    output reg         ctr_ceb_o, ctr_web_o,
    output reg  [X_AW-1:0] ctr_addr_o,
    input  wire [DATA_W-1:0] ctr_q_i,

    output reg         sz_ceb_o, sz_web_o,
    output reg  [X_AW-1:0] sz_addr_o,
    input  wire [DATA_W-1:0] sz_q_i,

    output reg         off_ceb_o, off_web_o,
    output reg  [X_AW-1:0] off_addr_o,
    input  wire [DATA_W-1:0] off_q_i,

    output reg         bbox_valid,
    output reg signed [DATA_W-1:0] bbox_data,
    output reg [1:0]   bbox_idx
);

localparam MAP_LEN = FEAT_SZ * FEAT_SZ;

localparam B_IDLE  = 4'd0;
localparam B_SCAN  = 4'd1;
localparam B_RD0   = 4'd2; // issue off_x
localparam B_RD1   = 4'd3; // cap off_x, issue off_y
localparam B_RD2   = 4'd4; // cap off_y, issue sz_w
localparam B_RD3   = 4'd5; // cap sz_w, issue sz_h
localparam B_RD4   = 4'd6; // cap sz_h
localparam B_E0    = 4'd7;
localparam B_E1    = 4'd8;
localparam B_E2    = 4'd9;
localparam B_E3    = 4'd10;
localparam B_DONE  = 4'd11;

reg [3:0] state, next_state;
reg [8:0] scan_idx;
reg       scan_fill;
reg signed [DATA_W-1:0] max_val;
reg [7:0] max_idx;
reg signed [DATA_W-1:0] off_x_r, off_y_r, sz_w_r, sz_h_r;

wire [3:0] idx_x = max_idx[3:0];
wire [3:0] idx_y = max_idx[7:4];
wire signed [16:0] sum_x = $signed({5'b0, idx_x, 8'b0}) + {off_x_r[15], off_x_r};
wire signed [16:0] sum_y = $signed({5'b0, idx_y, 8'b0}) + {off_y_r[15], off_y_r};
wire signed [16:0] cx_s = sum_x >>> 4;
wire signed [16:0] cy_s = sum_y >>> 4;
wire signed [DATA_W-1:0] cx_q =
    (cx_s > 17'sd32767) ? 16'sh7FFF : (cx_s < -17'sd32768) ? 16'sh8000 : cx_s[15:0];
wire signed [DATA_W-1:0] cy_q =
    (cy_s > 17'sd32767) ? 16'sh7FFF : (cy_s < -17'sd32768) ? 16'sh8000 : cy_s[15:0];

wire [X_AW-1:0] a_ox = {{(X_AW-8){1'b0}}, max_idx};           // ch0
wire [X_AW-1:0] a_oy = 10'd256 + {{(X_AW-8){1'b0}}, max_idx}; // ch1
wire [X_AW-1:0] a_sw = {{(X_AW-8){1'b0}}, max_idx};
wire [X_AW-1:0] a_sh = 10'd256 + {{(X_AW-8){1'b0}}, max_idx};

assign busy = (state != B_IDLE);

always @(posedge clk) begin
    if (reset) state <= B_IDLE;
    else state <= next_state;
end

always @(*) begin
    next_state = state;
    case (state)
        B_IDLE: if (start) next_state = B_SCAN;
        B_SCAN: if (!scan_fill && scan_idx == MAP_LEN[8:0]-9'd1) next_state = B_RD0;
        B_RD0:  next_state = B_RD1;
        B_RD1:  next_state = B_RD2;
        B_RD2:  next_state = B_RD3;
        B_RD3:  next_state = B_RD4;
        B_RD4:  next_state = B_E0;
        B_E0:   next_state = B_E1;
        B_E1:   next_state = B_E2;
        B_E2:   next_state = B_E3;
        B_E3:   next_state = B_DONE;
        B_DONE: next_state = B_IDLE;
        default: next_state = B_IDLE;
    endcase
end

always @(posedge clk) begin
    if (reset) done <= 1'b0;
    else if (state == B_DONE) done <= 1'b1;
    else done <= 1'b0;
end

always @(posedge clk) begin
    if (reset) begin
        scan_idx <= 9'd0; scan_fill <= 1'b0;
        max_val <= 16'sh8000; max_idx <= 8'd0;
    end else if (state == B_IDLE && start) begin
        scan_idx <= 9'd0; scan_fill <= 1'b1;
        max_val <= 16'sh8000; max_idx <= 8'd0;
    end else if (state == B_SCAN) begin
        if (scan_fill) scan_fill <= 1'b0;
        else begin
            if ($signed(ctr_q_i) > max_val) begin
                max_val <= $signed(ctr_q_i);
                max_idx <= scan_idx[7:0];
            end
            if (scan_idx != MAP_LEN[8:0]-9'd1)
                scan_idx <= scan_idx + 9'd1;
        end
    end
end

always @(posedge clk) begin
    if (reset) begin
        off_x_r <= 0; off_y_r <= 0; sz_w_r <= 0; sz_h_r <= 0;
    end else begin
        case (state)
            B_RD1: off_x_r <= $signed(off_q_i);
            B_RD2: off_y_r <= $signed(off_q_i);
            B_RD3: sz_w_r  <= $signed(sz_q_i);
            B_RD4: sz_h_r  <= $signed(sz_q_i);
            default: ;
        endcase
    end
end

always @(posedge clk) begin
    if (reset) begin
        bbox_valid <= 0; bbox_data <= 0; bbox_idx <= 0;
    end else begin
        bbox_valid <= 0;
        case (state)
            B_E0: begin bbox_valid <= 1; bbox_data <= cx_q;   bbox_idx <= 2'd0; end
            B_E1: begin bbox_valid <= 1; bbox_data <= cy_q;   bbox_idx <= 2'd1; end
            B_E2: begin bbox_valid <= 1; bbox_data <= sz_w_r; bbox_idx <= 2'd2; end
            B_E3: begin bbox_valid <= 1; bbox_data <= sz_h_r; bbox_idx <= 2'd3; end
            default: ;
        endcase
    end
end

always @(*) begin
    ctr_ceb_o = 1'b1; ctr_web_o = 1'b1; ctr_addr_o = 0;
    sz_ceb_o  = 1'b1; sz_web_o  = 1'b1; sz_addr_o  = 0;
    off_ceb_o = 1'b1; off_web_o = 1'b1; off_addr_o = 0;

    if (state == B_SCAN) begin
        if (scan_fill) begin
            ctr_ceb_o = 0; ctr_web_o = 1; ctr_addr_o = {{(X_AW-9){1'b0}}, scan_idx};
        end else if (scan_idx != MAP_LEN[8:0]-9'd1) begin
            ctr_ceb_o = 0; ctr_web_o = 1;
            ctr_addr_o = {{(X_AW-9){1'b0}}, scan_idx + 9'd1};
        end
    end

    // Present addr in RDn to capture in RD(n+1)
    case (state)
        B_RD0: begin off_ceb_o = 0; off_web_o = 1; off_addr_o = a_ox; end
        B_RD1: begin off_ceb_o = 0; off_web_o = 1; off_addr_o = a_oy; end
        B_RD2: begin sz_ceb_o  = 0; sz_web_o  = 1; sz_addr_o  = a_sw; end
        B_RD3: begin sz_ceb_o  = 0; sz_web_o  = 1; sz_addr_o  = a_sh; end
        default: ;
    endcase
end

endmodule
