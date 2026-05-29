// =============================================================================
// cal_bbox.v -- argmax(score_map_ctr) + size/offset lookup -> bbox (cx,cy,w,h)
// -----------------------------------------------------------------------------
// numpy: run_backbone_numpy_shared_trunk.py cal_bbox() L629-643
//   tail streams ctr/size/off -> Sram_qkm ; start (=tail.done) -> 4 reg latch -> emit
// Golden: Activation/box_head_after_cal_bbox_bbox_bi.txt
// Golden-Weight (SRAM pool): Sram_qkm 1280x16; size/off use addr[9:0] (1024 words)
//   [9:8]=00 size_w, 01 size_h, 10 off_x, 11 off_y ; [7:0]=max_idx_r at emit read
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
    so_ceb_o      ,
    so_web_o      ,
    so_addr_o     ,
    so_din_o      ,
    so_q_i        ,
    bbox_valid    ,
    bbox_data     ,
    bbox_idx
);

parameter DATA_W   = 16 ;
parameter FEAT_SZ  = 16 ;
parameter MAP_LEN  = FEAT_SZ * FEAT_SZ ;
parameter MAP2_LEN = 2 * MAP_LEN ;
parameter SO_AW    = 10 ;   // 512 size + 512 off
parameter QKM_AW   = 11 ;   // Sram_qkm macro address width (1280 depth)

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

output                      so_ceb_o      ;
output                      so_web_o      ;
output [QKM_AW-1:0]         so_addr_o     ;
output [DATA_W-1:0]         so_din_o      ;
input  [DATA_W-1:0]         so_q_i        ;

output                      bbox_valid    ;
output signed [DATA_W-1:0]  bbox_data     ;
output [1:0]                bbox_idx      ;

parameter B_IDLE      = 4'd0 ;
parameter B_RD_OFF_X  = 4'd1 ;
parameter B_RD_OFF_Y  = 4'd2 ;
parameter B_RD_SZ_W   = 4'd3 ;
parameter B_RD_SZ_H   = 4'd4 ;
parameter B_EMIT_CX   = 4'd5 ;
parameter B_EMIT_CY   = 4'd6 ;
parameter B_EMIT_W    = 4'd7 ;
parameter B_EMIT_H    = 4'd8 ;
parameter B_DONE      = 4'd9 ;

reg  [3:0]                  CS, NS ;
reg                         busy_r, done_r ;

reg  signed [DATA_W-1:0]    max_val_r ;
reg  [7:0]                  max_idx_r ;
reg  [8:0]                  ctr_cnt ;

reg  [9:0]                  size_cnt ;
reg  [9:0]                  off_cnt ;

// Emit lookup: latched from Sram_qkm after start (4 single-port reads)
reg  signed [DATA_W-1:0]    off_x_r ;
reg  signed [DATA_W-1:0]    off_y_r ;
reg  signed [DATA_W-1:0]    size_w_r ;
reg  signed [DATA_W-1:0]    size_h_r ;

reg                         bbox_valid_r ;
reg  signed [DATA_W-1:0]    bbox_data_r ;
reg  [1:0]                  bbox_idx_r ;

// Sram_qkm: size_buf[0:511] + off_buf[0:511] merged (addr[9:0])
reg        so_ceb ;
reg        so_web ;
reg [QKM_AW-1:0] so_addr ;
reg [DATA_W-1:0] so_din ;

reg        so_ceb_n ;
reg        so_web_n ;
reg [QKM_AW-1:0] so_addr_n ;
reg [DATA_W-1:0] so_din_n ;

wire                        _unused_sub ;

wire [SO_AW-1:0]            size_wr_addr ;
wire [SO_AW-1:0]            off_wr_addr ;
wire [SO_AW-1:0]            rd_off_x_addr ;
wire [SO_AW-1:0]            rd_off_y_addr ;
wire [SO_AW-1:0]            rd_size_w_addr ;
wire [SO_AW-1:0]            rd_size_h_addr ;

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

wire [DATA_W-1:0] so_q ;

assign busy       = busy_r ;
assign done       = done_r ;
assign bbox_valid = bbox_valid_r ;
assign bbox_data  = bbox_data_r ;
assign bbox_idx   = bbox_idx_r ;
assign so_ceb_o   = so_ceb ;
assign so_web_o   = so_web ;
assign so_addr_o  = so_addr ;
assign so_din_o   = so_din ;
assign so_q       = so_q_i ;

assign _unused_sub = size_in_sub | off_in_sub ;

// Linear stream write: size uses addr[9]=0 (0..511), off uses addr[9]=1 (512..1023)
assign size_wr_addr = {1'b0, size_cnt[8:0]} ;
assign off_wr_addr  = {1'b1, off_cnt[8:0]} ;

// Emit read: {sub[1:0], max_idx_r[7:0]} within size/off 512-word halves
assign rd_off_x_addr   = {2'b10, max_idx_r} ;
assign rd_off_y_addr   = {2'b11, max_idx_r} ;
assign rd_size_w_addr  = {2'b00, max_idx_r} ;
assign rd_size_h_addr  = {2'b01, max_idx_r} ;

assign idx_x  = max_idx_r[3:0] ;
assign idx_y  = max_idx_r[7:4] ;
assign off_x  = off_x_r ;
assign off_y  = off_y_r ;
assign size_w = size_w_r ;
assign size_h = size_h_r ;

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
        B_IDLE     : if (start) NS = B_RD_OFF_X ;
        B_RD_OFF_X :          NS = B_RD_OFF_Y ;
        B_RD_OFF_Y :          NS = B_RD_SZ_W  ;
        B_RD_SZ_W  :          NS = B_RD_SZ_H  ;
        B_RD_SZ_H  :          NS = B_EMIT_CX   ;
        B_EMIT_CX  :          NS = B_EMIT_CY  ;
        B_EMIT_CY  :          NS = B_EMIT_W   ;
        B_EMIT_W   :          NS = B_EMIT_H   ;
        B_EMIT_H   :          NS = B_DONE     ;
        B_DONE     :          NS = B_IDLE     ;
        default    :          NS = B_IDLE     ;
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

// size_cnt (stream write index; data in Sram_qkm via comb/posedge port)
always @(posedge clk) begin
    if (!rst_n)
        size_cnt <= 10'd0 ;
    else if (size_in_valid)
        size_cnt <= size_cnt + 10'd1 ;
end

// off_cnt
always @(posedge clk) begin
    if (!rst_n)
        off_cnt <= 10'd0 ;
    else if (off_in_valid)
        off_cnt <= off_cnt + 10'd1 ;
end

// Latch 4 map samples from so_q (read issued previous cycle; Q comb-valid in RD_*)
always @(posedge clk) begin
    if (!rst_n) begin
        off_x_r  <= 16'sd0 ;
        off_y_r  <= 16'sd0 ;
        size_w_r <= 16'sd0 ;
        size_h_r <= 16'sd0 ;
    end else begin
        case (CS)
            B_RD_OFF_X : off_x_r  <= so_q ;
            B_RD_OFF_Y : off_y_r  <= so_q ;
            B_RD_SZ_W  : size_w_r <= so_q ;
            B_RD_SZ_H  : size_h_r <= so_q ;
            default    : ;
        endcase
    end
end

// Sram_qkm mux: comb _n -> posedge reg -> parent macro (align with head_top SRAM)
always @(*) begin
    so_ceb_n  = 1'b1 ;
    so_web_n  = 1'b1 ;
    so_addr_n = {QKM_AW{1'b0}} ;
    so_din_n  = {DATA_W{1'b0}} ;

    if (size_in_valid) begin
        so_ceb_n  = 1'b0 ;
        so_web_n  = 1'b0 ;
        so_addr_n = {{(QKM_AW-SO_AW){1'b0}}, size_wr_addr} ;
        so_din_n  = size_in_data ;
    end else if (off_in_valid) begin
        so_ceb_n  = 1'b0 ;
        so_web_n  = 1'b0 ;
        so_addr_n = {{(QKM_AW-SO_AW){1'b0}}, off_wr_addr} ;
        so_din_n  = off_in_data ;
    end else begin
        case (CS)
            B_IDLE : begin
                if (start) begin
                    so_ceb_n  = 1'b0 ;
                    so_web_n  = 1'b1 ;
                    so_addr_n = {{(QKM_AW-SO_AW){1'b0}}, rd_off_x_addr} ;
                end
            end
            B_RD_OFF_X : begin
                so_ceb_n  = 1'b0 ;
                so_web_n  = 1'b1 ;
                so_addr_n = {{(QKM_AW-SO_AW){1'b0}}, rd_off_y_addr} ;
            end
            B_RD_OFF_Y : begin
                so_ceb_n  = 1'b0 ;
                so_web_n  = 1'b1 ;
                so_addr_n = {{(QKM_AW-SO_AW){1'b0}}, rd_size_w_addr} ;
            end
            B_RD_SZ_W : begin
                so_ceb_n  = 1'b0 ;
                so_web_n  = 1'b1 ;
                so_addr_n = {{(QKM_AW-SO_AW){1'b0}}, rd_size_h_addr} ;
            end
            default : ;
        endcase
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        so_ceb  <= 1'b1 ;
        so_web  <= 1'b1 ;
        so_addr <= {QKM_AW{1'b0}} ;
        so_din  <= {DATA_W{1'b0}} ;
    end else begin
        so_ceb  <= so_ceb_n ;
        so_web  <= so_web_n ;
        so_addr <= so_addr_n ;
        so_din  <= so_din_n ;
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
                bbox_data_r  <= size_w_r ;
                bbox_idx_r   <= 2'd2 ;
            end
            B_EMIT_H : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= size_h_r ;
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
