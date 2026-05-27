// =============================================================================
// cal_bbox.v -- argmax(score_map_ctr) + size/offset lookup -> bbox (cx,cy,w,h)
// -----------------------------------------------------------------------------
// numpy: run_backbone_numpy_shared_trunk.py cal_bbox() L629-643
//   tail streams ctr/size/off -> buffers ; start (=tail.done) emits 4 bbox beats
// Golden: Activation/box_head_after_cal_bbox_bbox_bi.txt
// =============================================================================

module cal_bbox (
    clk           ,
    rst_n         ,
    start         ,
    busy          ,
    done          ,
    ctr_in_valid  ,
    ctr_in_data   ,
    size_in_valid ,
    size_in_data  ,
    size_in_sub   ,
    off_in_valid  ,
    off_in_data   ,
    off_in_sub    ,
    bbox_valid    ,
    bbox_data     ,
    bbox_idx
);

parameter DATA_W   = 16 ;
parameter FEAT_SZ  = 16 ;
parameter MAP_LEN  = FEAT_SZ * FEAT_SZ ;
parameter MAP2_LEN = 2 * MAP_LEN ;

input                       clk           ;
input                       rst_n         ;
input                       start         ;
output                      busy          ;
output                      done          ;

input                       ctr_in_valid  ;
input  signed [DATA_W-1:0]  ctr_in_data   ;

input                       size_in_valid ;
input  signed [DATA_W-1:0]  size_in_data  ;
input                       size_in_sub   ;

input                       off_in_valid  ;
input  signed [DATA_W-1:0]  off_in_data   ;
input                       off_in_sub    ;

output                      bbox_valid    ;
output signed [DATA_W-1:0]  bbox_data     ;
output [1:0]                bbox_idx      ;

parameter B_IDLE    = 3'd0 ;
parameter B_EMIT_CX = 3'd1 ;
parameter B_EMIT_CY = 3'd2 ;
parameter B_EMIT_W  = 3'd3 ;
parameter B_EMIT_H  = 3'd4 ;
parameter B_DONE    = 3'd5 ;

reg  [2:0]                  CS, NS ;
reg                         busy_r, done_r ;

reg  signed [DATA_W-1:0]    max_val_r ;
reg  [7:0]                  max_idx_r ;
reg  [8:0]                  ctr_cnt ;

reg  signed [DATA_W-1:0]    size_buf [0:MAP2_LEN-1] ;
reg  signed [DATA_W-1:0]    off_buf  [0:MAP2_LEN-1] ;
reg  [9:0]                  size_cnt ;
reg  [9:0]                  off_cnt ;

reg                         bbox_valid_r ;
reg  signed [DATA_W-1:0]    bbox_data_r ;
reg  [1:0]                  bbox_idx_r ;

wire                        _unused_sub ;

wire [3:0]                  idx_x ;
wire [3:0]                  idx_y ;
wire signed [DATA_W-1:0]    off_x ;
wire signed [DATA_W-1:0]    off_y ;
wire signed [DATA_W-1:0]    size_w ;
wire signed [DATA_W-1:0]    size_h ;
wire signed [16:0]          sum_x ;
wire signed [16:0]          sum_y ;
wire signed [16:0]          cx_shr ;
wire signed [16:0]          cy_shr ;
wire signed [DATA_W-1:0]    cx_q88 ;
wire signed [DATA_W-1:0]    cy_q88 ;

assign busy       = busy_r ;
assign done       = done_r ;
assign bbox_valid = bbox_valid_r ;
assign bbox_data  = bbox_data_r ;
assign bbox_idx   = bbox_idx_r ;

assign _unused_sub = size_in_sub | off_in_sub ;

assign idx_x  = max_idx_r[3:0] ;
assign idx_y  = max_idx_r[7:4] ;
assign off_x  = off_buf [{1'b0, max_idx_r}] ;
assign off_y  = off_buf [{1'b1, max_idx_r}] ;
assign size_w = size_buf[{1'b0, max_idx_r}] ;
assign size_h = size_buf[{1'b1, max_idx_r}] ;

assign sum_x = $signed({5'b0, idx_x, 8'b0}) + {off_x[DATA_W-1], off_x} ;
assign sum_y = $signed({5'b0, idx_y, 8'b0}) + {off_y[DATA_W-1], off_y} ;
assign cx_shr = sum_x >>> 4 ;
assign cy_shr = sum_y >>> 4 ;

assign cx_q88 = (cx_shr >  17'sd32767) ? 16'sh7fff :
                (cx_shr < -17'sd32768) ? 16'sh8000 : cx_shr[DATA_W-1:0] ;

assign cy_q88 = (cy_shr >  17'sd32767) ? 16'sh7fff :
                (cy_shr < -17'sd32768) ? 16'sh8000 : cy_shr[DATA_W-1:0] ;

// FSM CS
always @(posedge clk) begin
    if (!rst_n)
        CS <= B_IDLE ;
    else
        CS <= NS ;
end

// FSM NS
always @(*) begin
    NS = CS ;
    case (CS)
        B_IDLE    : if (start) NS = B_EMIT_CX ;
        B_EMIT_CX :          NS = B_EMIT_CY ;
        B_EMIT_CY :          NS = B_EMIT_W  ;
        B_EMIT_W  :          NS = B_EMIT_H  ;
        B_EMIT_H  :          NS = B_DONE    ;
        B_DONE    :          NS = B_IDLE    ;
        default   :          NS = B_IDLE    ;
    endcase
end

// busy_r, done_r
always @(posedge clk) begin
    if (!rst_n) begin
        busy_r <= 1'b0 ;
        done_r <= 1'b0 ;
    end else begin
        busy_r <= (NS != B_IDLE) && (NS != B_DONE) ;
        done_r <= (NS == B_DONE) ;
    end
end

// argmax: max_val_r, max_idx_r, ctr_cnt
always @(posedge clk) begin
    if (!rst_n) begin
        max_val_r <= 16'sd0 ;
        max_idx_r <= 8'd0 ;
        ctr_cnt   <= 9'd0 ;
    end else if (ctr_in_valid) begin
        if (ctr_in_data > max_val_r) begin
            max_val_r <= ctr_in_data ;
            max_idx_r <= ctr_cnt[7:0] ;
        end
        ctr_cnt <= ctr_cnt + 9'd1 ;
    end
end

// size_buf, size_cnt
always @(posedge clk) begin
    if (!rst_n)
        size_cnt <= 10'd0 ;
    else if (size_in_valid) begin
        size_buf[size_cnt] <= size_in_data ;
        size_cnt <= size_cnt + 10'd1 ;
    end
end

// off_buf, off_cnt
always @(posedge clk) begin
    if (!rst_n)
        off_cnt <= 10'd0 ;
    else if (off_in_valid) begin
        off_buf[off_cnt] <= off_in_data ;
        off_cnt <= off_cnt + 10'd1 ;
    end
end

// bbox_valid_r, bbox_data_r, bbox_idx_r  (preload from NS)
always @(posedge clk) begin
    if (!rst_n) begin
        bbox_valid_r <= 1'b0 ;
        bbox_data_r  <= 16'sd0 ;
        bbox_idx_r   <= 2'd0 ;
    end else begin
        case (NS)
            B_EMIT_CX : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= cx_q88 ;
                bbox_idx_r   <= 2'd0 ;
            end
            B_EMIT_CY : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= cy_q88 ;
                bbox_idx_r   <= 2'd1 ;
            end
            B_EMIT_W : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= size_w ;
                bbox_idx_r   <= 2'd2 ;
            end
            B_EMIT_H : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= size_h ;
                bbox_idx_r   <= 2'd3 ;
            end
            default : begin
                bbox_valid_r <= 1'b0 ;
                bbox_idx_r   <= 2'd0 ;
            end
        endcase
    end
end

endmodule
