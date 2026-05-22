// =============================================================================
// linear_wide.v
//
// Q8.8 linear (MAC + sat16) — fc2 variant with IN_DIM=128, OUT_DIM=32.
// Mirrors the FSM/handshake of linear.v but with widened feat counters and a
// different w_addr_o packing matching backbone_top.v's fc2 decoding:
//   addr_fc2w = block*4096 + local[12:0]       (weight)
//   addr_fc2b = block*32   + local[11:7]       (bias)
// → local[6:0]   = feat (7 bits, 0..127)
// → local[11:7]  = neu  (5 bits, 0..31)
// → local[12]    = 0
//
// Phases (per neuron):
//   S_LOAD : caller streams IN_DIM=128 x values into x_buf.
//   S_WPRE : pre-fetch 128 weights into wgt_buf via ROM (CLK=~clk).
//   S_MAC  : 128-cycle MAC using x_buf × wgt_buf; final cycle outputs y_o.
//
// Bias timing mirrors linear.v: bias_hold latched on the first wgt write
// (wpre_stream_r && wpre_feat==0), i.e. when w_addr_o = neu*128 + 0 settled
// one posedge earlier so ROM bias addr = blk*32 + neu is correctly resolved.
//
// Used only by mlp.v for the fc2 stage. fc1 uses the existing linear.v
// (IN=32, OUT=128) because feat fits in 5 bits there.
// =============================================================================

module linear_wide #(
    parameter IN_DIM  = 128,
    parameter OUT_DIM = 32
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    input  wire signed [15:0] x_i,
    input  wire        x_valid,

    input  wire signed [15:0] w_i,
    input  wire signed [15:0] b_i,
    output wire [12:0] w_addr_o,

    output wire        busy,
    output reg         done,
    output reg  signed [15:0] y_o,
    output reg         y_valid,
    output reg  [4:0]  y_neu_o
);

parameter S_IDLE = 3'd0;
parameter S_LOAD = 3'd1;
parameter S_WPRE = 3'd2;
parameter S_MAC  = 3'd3;
parameter S_DONE = 3'd4;

reg signed [15:0] x_buf   [0:IN_DIM-1];
reg signed [15:0] wgt_buf [0:IN_DIM-1];
reg signed [15:0] bias_hold;

reg [2:0] state, next_state;
reg [6:0] load_cnt;       // 0..127 (7 bits for IN=128)
reg [6:0] wpre_feat;
reg [6:0] mac_feat;
reg       wpre_stream;
reg       wpre_stream_r;
reg [4:0] neu_cnt;        // 0..31 (5 bits for OUT=32)

reg signed [31:0] acc;

`ifndef SYNTHESIS
integer lw_ii;
initial begin
    for (lw_ii = 0; lw_ii < IN_DIM; lw_ii = lw_ii + 1)
        wgt_buf[lw_ii] = 16'sd0;
end
`endif

// Address packing for fc2: {1'b0, neu[4:0], feat[6:0]} = 13 bits.
// local[11:7] = neu, local[6:0] = feat — matches backbone_top fc2 decoders.
wire [4:0] neu_for_addr  = neu_cnt;
wire [6:0] feat_for_addr =
    (state == S_WPRE) ? wpre_feat :
    (state == S_MAC)  ? mac_feat :
                        7'd0;

assign w_addr_o = {1'b0, neu_for_addr, feat_for_addr};

wire wgt_wr_ce     = (state == S_WPRE) && wpre_stream_r;
wire bias_latch_ce =
    (state == S_WPRE) && wpre_stream_r && (wpre_feat == 7'd0);

// MAC product (no ROM latency during S_MAC since wgt was pre-fetched)
wire signed [31:0] mac_prod =
    $signed(x_buf[mac_feat]) * $signed(wgt_buf[mac_feat]);

wire signed [31:0] acc_final  = acc + mac_prod;
wire signed [31:0] acc_shr8   = acc_final >>> 8;
wire signed [31:0] acc_plus_b = acc_shr8 + $signed({{16{bias_hold[15]}}, bias_hold});

// Saturate Q16.16 accumulator (post-shift) to Q8.8 16-bit signed
function signed [15:0] sat16_q88;
    input signed [31:0] v;
    begin
        if (v > 32'sd32767)        sat16_q88 = 16'sh7FFF;
        else if (v < -32'sd32768)  sat16_q88 = 16'sh8000;
        else                        sat16_q88 = v[15:0];
    end
endfunction

wire signed [15:0] y_next_c = sat16_q88(acc_plus_b);

wire mac_last = (mac_feat == IN_DIM[6:0] - 7'd1);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    case (state)
        S_IDLE:  next_state = start ? S_LOAD : S_IDLE;
        S_LOAD:  next_state = (load_cnt == IN_DIM[6:0] - 7'd1 && x_valid) ? S_WPRE : S_LOAD;
        S_WPRE:  next_state = ((wpre_feat == IN_DIM[6:0] - 7'd1) && wpre_stream_r) ? S_MAC : S_WPRE;
        S_MAC:   next_state = mac_last ? ((neu_cnt == OUT_DIM[4:0] - 5'd1) ? S_DONE : S_WPRE)
                                        : S_MAC;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

// FSM segment 3: datapath / output regs
always @(posedge clk) begin
    done    <= 1'b0;
    y_valid <= 1'b0;

    wpre_stream_r <= wpre_stream;

    if (state == S_LOAD && x_valid)
        x_buf[load_cnt] <= x_i;

    if (wgt_wr_ce)
        wgt_buf[wpre_feat] <= w_i;

    if (bias_latch_ce)
        bias_hold <= b_i;

    if (reset) begin
        load_cnt      <= 7'd0;
        wpre_feat     <= 7'd0;
        wpre_stream   <= 1'b0;
        mac_feat      <= 7'd0;
        neu_cnt       <= 5'd0;
        acc           <= 32'sd0;
        bias_hold     <= 16'sd0;
        y_o           <= 16'sd0;
        y_neu_o       <= 5'd0;
    end else begin
        case (state)
            S_IDLE: begin
                load_cnt    <= 7'd0;
                wpre_feat   <= 7'd0;
                wpre_stream <= 1'b0;
                mac_feat    <= 7'd0;
                neu_cnt     <= 5'd0;
                acc         <= 32'sd0;
                bias_hold   <= 16'sd0;
            end

            S_LOAD: begin
                if (x_valid) begin
                    if (load_cnt == IN_DIM[6:0] - 7'd1) begin
                        load_cnt    <= 7'd0;
                        wpre_feat   <= 7'd0;
                        wpre_stream <= 1'b0;
                        neu_cnt     <= 5'd0;
                        acc         <= 32'sd0;
                    end else begin
                        load_cnt <= load_cnt + 7'd1;
                    end
                end
            end

            S_WPRE: begin
                if (!wpre_stream)
                    wpre_stream <= 1'b1;
                else if (wpre_stream_r) begin
                    if (wpre_feat != IN_DIM[6:0] - 7'd1) begin
                        wpre_feat <= wpre_feat + 7'd1;
                    end else begin
                        wpre_stream <= 1'b0;
                        mac_feat    <= 7'd0;
                        acc         <= 32'sd0;
                    end
                end
            end

            S_MAC: begin
                if (mac_last) begin
                    y_o     <= y_next_c;
                    y_neu_o <= neu_cnt;
                    y_valid <= 1'b1;
                    if (neu_cnt == OUT_DIM[4:0] - 5'd1) begin
                        neu_cnt <= 5'd0;
                    end else begin
                        neu_cnt     <= neu_cnt + 5'd1;
                        wpre_feat   <= 7'd0;
                        wpre_stream <= 1'b0;
                    end
                end else begin
                    acc      <= acc + mac_prod;
                    mac_feat <= mac_feat + 7'd1;
                end
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: ;
        endcase
    end
end

assign busy = (state != S_IDLE);

endmodule
