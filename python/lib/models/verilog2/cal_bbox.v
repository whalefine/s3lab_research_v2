// =============================================================================
// cal_bbox.v -- argmax(score_map_ctr) + lookup size/offset -> bbox(cx,cy,w,h)
// -----------------------------------------------------------------------------
// 對應 numpy : run_backbone_numpy_shared_trunk.py L629-643 (cal_bbox)
//
//   idx        = argmax(score_map_ctr[0..255])     ; first-occurrence tie-break
//   idx_y      = idx >> 4         ; idx_x = idx & 0xF
//   offset_x   = offset_map[0, 0, idx_y, idx_x]    ; sub=0 portion (first 256)
//   offset_y   = offset_map[0, 1, idx_y, idx_x]    ; sub=1 portion (next 256)
//   size_w     = size_map  [0, 0, idx_y, idx_x]
//   size_h     = size_map  [0, 1, idx_y, idx_x]
//   cx_q88     = ((idx_x << 8) + offset_x_q88) >>> 4   ; arith right-shift by 4
//   cy_q88     = ((idx_y << 8) + offset_y_q88) >>> 4
//   bbox[0..3] = {cx, cy, size_w, size_h}              ; all Q8.8
//
// Streaming input layout (tail.v drives in this order) :
//   ctr_in   : 256 valids in row-major (oh,ow) order ; sigmoid_clamped values
//   size_in  : 512 valids ; sub=0 (w) first 256 then sub=1 (h) next 256
//   off_in   : 512 valids ; sub=0 (x) first 256 then sub=1 (y) next 256
//   All three streams arrive concurrently with the tail FSM ; cal_bbox passively
//   accumulates argmax + writes size_buf / off_buf during tail run, then on
//   `start` (= tail.done) emits 4 bbox values over 4 cycles.
//
// Buffer access pattern (no read/write hazard) :
//   - WRITE phase : while tail runs , size/off_in_valid -> write at running index
//   - READ phase  : after start pulse , combinational lookup at argmax index
//
// Golden : Activation/box_head_after_cal_bbox_bbox_bi.txt   (4 lines : cx,cy,w,h)
// =============================================================================

module cal_bbox (
    clk           ,
    rst_n         ,
    start         ,
    busy          ,
    done          ,
    // streaming inputs from tail
    ctr_in_valid  ,
    ctr_in_data   ,
    size_in_valid ,
    size_in_data  ,
    size_in_sub   ,    // 0 = w , 1 = h   (kept on port for symmetry / future use)
    off_in_valid  ,
    off_in_data   ,
    off_in_sub    ,    // 0 = x , 1 = y   (kept on port for symmetry / future use)
    // bbox output (4 valids in order : cx , cy , w , h)
    bbox_valid    ,
    bbox_data     ,
    bbox_idx
);

// ---------------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------------
parameter DATA_W   = 16 ;
parameter FEAT_SZ  = 16 ;
parameter MAP_LEN  = FEAT_SZ*FEAT_SZ ;        // 256
parameter MAP2_LEN = 2*MAP_LEN ;              // 512

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// FSM state encoding
// ---------------------------------------------------------------------------
parameter B_IDLE    = 3'd0 ;
parameter B_EMIT_CX = 3'd1 ;
parameter B_EMIT_CY = 3'd2 ;
parameter B_EMIT_W  = 3'd3 ;
parameter B_EMIT_H  = 3'd4 ;
parameter B_DONE    = 3'd5 ;

// ---------------------------------------------------------------------------
// Reg / wire declarations
// ---------------------------------------------------------------------------
reg  [2:0]                  state, next_state ;
reg                         busy_r, done_r ;

// argmax tracking
reg  signed [DATA_W-1:0]    max_val_r ;
reg  [7:0]                  max_idx_r ;       // 0..255
reg  [8:0]                  ctr_cnt ;         // 0..256 ; sized for safety

// streaming buffers (no read/write race : write during tail run , read after start)
reg  signed [DATA_W-1:0]    size_buf [0:MAP2_LEN-1] ;
reg  signed [DATA_W-1:0]    off_buf  [0:MAP2_LEN-1] ;
reg  [9:0]                  size_cnt ;        // 0..512
reg  [9:0]                  off_cnt ;         // 0..512

// bbox emit register
reg                         bbox_valid_r ;
reg  signed [DATA_W-1:0]    bbox_data_r ;
reg  [1:0]                  bbox_idx_r ;

// combinational lookup values
reg  [3:0]                  idx_x_c ;
reg  [3:0]                  idx_y_c ;
reg  signed [DATA_W-1:0]    off_x_c ;
reg  signed [DATA_W-1:0]    off_y_c ;
reg  signed [DATA_W-1:0]    size_w_c ;
reg  signed [DATA_W-1:0]    size_h_c ;
reg  signed [16:0]          sum_x_c ;         // (idx*256) + signed off : 17-bit signed
reg  signed [16:0]          sum_y_c ;
reg  signed [16:0]          cx_shr_c ;        // arith shr 4
reg  signed [16:0]          cy_shr_c ;
reg  signed [DATA_W-1:0]    cx_q88_c ;
reg  signed [DATA_W-1:0]    cy_q88_c ;

// silenced (kept for future routing) - prevent unused-input warnings
wire                        _unused_sub = size_in_sub | off_in_sub ;

// ---------------------------------------------------------------------------
// Output assigns
// ---------------------------------------------------------------------------
assign busy       = busy_r ;
assign done       = done_r ;
assign bbox_valid = bbox_valid_r ;
assign bbox_data  = bbox_data_r ;
assign bbox_idx   = bbox_idx_r ;

// ---------------------------------------------------------------------------
// (seq) FSM state register
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) state <= B_IDLE ;
    else        state <= next_state ;
end

// ---------------------------------------------------------------------------
// (comb) FSM next-state  (one shot : IDLE -> CX -> CY -> W -> H -> DONE -> IDLE)
// ---------------------------------------------------------------------------
always @(*) begin
    next_state = state ;
    case (state)
        B_IDLE    : if (start) next_state = B_EMIT_CX ;
        B_EMIT_CX :            next_state = B_EMIT_CY ;
        B_EMIT_CY :            next_state = B_EMIT_W  ;
        B_EMIT_W  :            next_state = B_EMIT_H  ;
        B_EMIT_H  :            next_state = B_DONE    ;
        B_DONE    :            next_state = B_IDLE    ;
        default   :            next_state = B_IDLE    ;
    endcase
end

// ---------------------------------------------------------------------------
// (seq) busy / done
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        busy_r <= 1'b0 ;
        done_r <= 1'b0 ;
    end else begin
        busy_r <= (next_state != B_IDLE) && (next_state != B_DONE) ;
        done_r <= (next_state == B_DONE) ;
    end
end

// ---------------------------------------------------------------------------
// (seq) argmax tracker + ctr counter
//   First-occurrence tie-break (strict >) matches numpy np.argmax.
//   max_val_r init = 0 ; sigmoid_clamped ctr range is [1, 255] so first ctr_in
//   always updates (matching numpy's argmax of all-equal sigmoid=1/256 case).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// (seq) size_buf writes  (stream order : sub=0 [0..255] then sub=1 [256..511])
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        size_cnt <= 10'd0 ;
    end else if (size_in_valid) begin
        size_buf[size_cnt] <= size_in_data ;
        size_cnt <= size_cnt + 10'd1 ;
    end
end

// ---------------------------------------------------------------------------
// (seq) off_buf writes  (stream order : sub=0 [0..255] then sub=1 [256..511])
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        off_cnt <= 10'd0 ;
    end else if (off_in_valid) begin
        off_buf[off_cnt] <= off_in_data ;
        off_cnt <= off_cnt + 10'd1 ;
    end
end

// ---------------------------------------------------------------------------
// (comb) bbox compute path
//   off_x  = off_buf [ idx ]                 (sub=0 portion 0..255)
//   off_y  = off_buf [ 256 + idx ]           (sub=1 portion 256..511)
//   size_w = size_buf[ idx ]
//   size_h = size_buf[ 256 + idx ]
//   sum_x  = (idx_x << 8) + off_x            (17-bit signed)
//   cx     = sat16(sum_x >>> 4)              ; matches numpy floor-div for negatives
// ---------------------------------------------------------------------------
always @(*) begin
    idx_x_c   = max_idx_r[3:0] ;
    idx_y_c   = max_idx_r[7:4] ;
    off_x_c   = off_buf [{1'b0, max_idx_r}] ;
    off_y_c   = off_buf [{1'b1, max_idx_r}] ;
    size_w_c  = size_buf[{1'b0, max_idx_r}] ;
    size_h_c  = size_buf[{1'b1, max_idx_r}] ;

    // (idx_x << 8) zero-extended into 17-bit signed ; off_x sign-extended.
    sum_x_c   = $signed({5'b0, idx_x_c, 8'b0}) + {off_x_c[DATA_W-1], off_x_c} ;
    sum_y_c   = $signed({5'b0, idx_y_c, 8'b0}) + {off_y_c[DATA_W-1], off_y_c} ;
    cx_shr_c  = sum_x_c >>> 4 ;
    cy_shr_c  = sum_y_c >>> 4 ;

    // sat16 with explicit signed bounds (Verilog literal width = 32 by default)
    if (cx_shr_c >  17'sd32767)        cx_q88_c = 16'sh7fff ;
    else if (cx_shr_c < -17'sd32768)   cx_q88_c = 16'sh8000 ;
    else                               cx_q88_c = cx_shr_c[DATA_W-1:0] ;
    if (cy_shr_c >  17'sd32767)        cy_q88_c = 16'sh7fff ;
    else if (cy_shr_c < -17'sd32768)   cy_q88_c = 16'sh8000 ;
    else                               cy_q88_c = cy_shr_c[DATA_W-1:0] ;
end

// ---------------------------------------------------------------------------
// (seq) bbox output emit (4 cycles : cx , cy , w , h)
//   Uses `next_state` to pre-load the register so the value is visible during
//   the state that names it (B_EMIT_CX -> bbox_data == cx_q88_c , etc.).
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        bbox_valid_r <= 1'b0 ;
        bbox_data_r  <= 16'sd0 ;
        bbox_idx_r   <= 2'd0 ;
    end else begin
        case (next_state)
            B_EMIT_CX : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= cx_q88_c ;
                bbox_idx_r   <= 2'd0 ;
            end
            B_EMIT_CY : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= cy_q88_c ;
                bbox_idx_r   <= 2'd1 ;
            end
            B_EMIT_W : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= size_w_c ;
                bbox_idx_r   <= 2'd2 ;
            end
            B_EMIT_H : begin
                bbox_valid_r <= 1'b1 ;
                bbox_data_r  <= size_h_c ;
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
