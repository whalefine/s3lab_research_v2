// =============================================================================
// attn_z_recip.v
//
// CARE attention S_Z_RECIP: in-place qkm -> zr on Sram_qkm via recip_nr.
// Bit-accurate vs run_backbone_numpy_shared_trunk.py _recip_nr_q88_fixed and
// verilog_backbone2/care_attention.v S_Z_RECIP.
//
// Per index: 2-phase SRAM read qkm -> recip_nr(1 NR iter) -> write zr same addr.
// Caller must preload qkm >= 1 (preprocess qkm_wr_val already clamps).
//
// SRAM read: 1-cycle latency (phase-0 ADDR, phase-1 USE + recip start).
// Golden: numpy _recip_nr_q88_fixed(np.maximum(qkm, 1)) per (h,n) flat index.
// =============================================================================

module attn_z_recip #(
    parameter NUM_HEADS = 4,
    parameter N_TOKENS  = 320
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output wire        busy,
    output reg         done,

    output reg         sram_qkm_ceb_o,
    output reg         sram_qkm_web_o,
    output reg [13:0]  sram_qkm_addr_o,
    output reg [15:0]  sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i
);

localparam QKM_ELEMS = NUM_HEADS * N_TOKENS;

parameter S_IDLE    = 2'd0;
parameter S_Z_RECIP = 2'd1;
parameter S_DONE_ST = 2'd2;

reg [1:0] state;
reg [1:0] next_state;

reg [10:0]        zr_idx;
reg               zr_phase;
reg signed [15:0] zr_qkm_r;

reg               recip_start;
reg signed [15:0] recip_x;
wire              recip_busy;
wire              recip_done;
wire signed [15:0] recip_y;

assign busy = (state != S_IDLE);

recip_nr u_recip (
    .clk   (clk),
    .reset (reset),
    .start (recip_start),
    .x_i   (recip_x),
    .busy  (recip_busy),
    .done  (recip_done),
    .y_o   (recip_y)
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
    case (state)
        S_IDLE:    next_state = start ? S_Z_RECIP : S_IDLE;
        S_Z_RECIP: next_state = (recip_done && zr_idx == QKM_ELEMS[10:0] - 11'd1) ?
                                 S_DONE_ST : S_Z_RECIP;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// done pulse
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE_ST)
        done <= 1'b1;
    else
        done <= 1'b0;
end

// S_Z_RECIP: latch qkm read at phase-0 (USE beat consumes prior ADDR read)
always @(posedge clk) begin
    if (reset)
        zr_qkm_r <= 16'sd0;
    else if (state == S_Z_RECIP && zr_phase == 1'b0)
        zr_qkm_r <= sram_qkm_q_i;
end

// S_Z_RECIP datapath + sub-block start pulse
always @(posedge clk) begin
    recip_start <= 1'b0;

    if (reset) begin
        zr_idx   <= 11'd0;
        zr_phase <= 1'b0;
        recip_x  <= 16'sd0;
    end else begin
        case (state)
            S_IDLE: begin
                zr_idx   <= 11'd0;
                zr_phase <= 1'b0;
            end

            S_Z_RECIP: begin
                if (recip_done) begin
                    if (zr_idx == QKM_ELEMS[10:0] - 11'd1)
                        zr_idx <= 11'd0;
                    else
                        zr_idx <= zr_idx + 11'd1;
                    zr_phase <= 1'b0;
                end else if (!recip_busy) begin
                    if (zr_phase == 1'b0)
                        zr_phase <= 1'b1;
                    else begin
                        recip_start <= 1'b1;
                        recip_x     <= zr_qkm_r;
                    end
                end
            end

            default: ;
        endcase
    end
end

// SRAM port mux
always @(*) begin
    sram_qkm_ceb_o  = 1'b1;
    sram_qkm_web_o  = 1'b1;
    sram_qkm_addr_o = 14'd0;
    sram_qkm_din_o  = 16'd0;

    if (state == S_Z_RECIP) begin
        if (recip_done) begin
            sram_qkm_ceb_o  = 1'b0;
            sram_qkm_web_o  = 1'b0;
            sram_qkm_addr_o = {3'd0, zr_idx[10:0]};
            sram_qkm_din_o  = recip_y;
        end else if (zr_phase == 1'b0) begin
            sram_qkm_ceb_o  = 1'b0;
            sram_qkm_web_o  = 1'b1;
            sram_qkm_addr_o = {3'd0, zr_idx[10:0]};
        end
    end
end

endmodule
