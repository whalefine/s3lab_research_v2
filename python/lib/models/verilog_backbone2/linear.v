// =============================================================================
// linear.v
//
// Q8.8 linear (MAC + sat16). ROM CLK=~clk: addr @ posedge T -> w_i valid @ T+1.
//
// Weight prefetch (matches verilog_backbone/care_attention.v rp_feat_mac):
//   Each posedge with wpre_stream=1: present addr = wpre_feat on ROM bus.
//   Each posedge with wpre_stream_r=1: wgt_buf[wpre_feat] <= w_i (feat not yet
//   incremented); then wpre_feat++ if not last. Do NOT use a separate wr_slot
//   updated while stream&&stream_r (that double-writes slot 0).
//
// bias_hold latched on WPRE stream rising edge at feat==0.
// =============================================================================

module linear #(
    parameter IN_DIM  = 32,
    parameter OUT_DIM = 96,
    parameter DUMP_WGT = 0   // 1: enable [WGT_*] when +define+DUMP_WGT_PREFETCH (QKV only)
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
    output reg  [6:0]  y_neu_o
);

parameter S_IDLE  = 3'd0;
parameter S_LOAD  = 3'd1;
parameter S_WPRE  = 3'd2;
parameter S_MAC   = 3'd3;
parameter S_DONE  = 3'd4;

reg signed [15:0] x_buf   [0:IN_DIM-1];
reg signed [15:0] wgt_buf [0:IN_DIM-1];
reg signed [15:0] bias_hold;

reg [2:0] state, next_state;
reg [4:0] load_cnt;
reg [4:0] wpre_feat;
reg       wpre_stream;
reg       wpre_stream_r;
reg [4:0] mac_feat;
reg [6:0] neu_cnt;

reg signed [31:0] acc;

`ifndef SYNTHESIS
integer lin_ii;
initial begin
    for (lin_ii = 0; lin_ii < IN_DIM; lin_ii = lin_ii + 1)
        wgt_buf[lin_ii] = 16'sd0;
end
`endif

wire [6:0] neu_for_addr = neu_cnt;
wire [4:0] feat_for_addr =
    (state == S_WPRE) ? wpre_feat :
    (state == S_MAC)  ? mac_feat :
                        5'd0;

assign w_addr_o = {1'b0, neu_for_addr, feat_for_addr};

wire wgt_wr_ce     = (state == S_WPRE) && wpre_stream_r;
// Latch bias when first weight is written (feat still 0); matches care_attention *_bias_ce.
wire bias_latch_ce = wpre_stream_r && (wpre_feat == 5'd0);

wire signed [31:0] mac_prod =
    $signed(x_buf[mac_feat]) * $signed(wgt_buf[mac_feat]);

wire signed [31:0] acc_final  = acc + mac_prod;
wire signed [31:0] acc_shr8   = acc_final >>> 8;
wire signed [31:0] acc_plus_b = acc_shr8 + $signed({{16{bias_hold[15]}}, bias_hold});

function signed [15:0] sat16_q88;
    input signed [31:0] v;
    begin
        if (v > 32'sd32767)        sat16_q88 = 16'sh7FFF;
        else if (v < -32'sd32768)  sat16_q88 = 16'sh8000;
        else                        sat16_q88 = v[15:0];
    end
endfunction

wire signed [15:0] y_next_c = sat16_q88(acc_plus_b);

wire mac_last = (mac_feat == IN_DIM[4:0] - 5'd1);

always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

always @(*) begin
    case (state)
        S_IDLE:  next_state = start ? S_LOAD : S_IDLE;
        S_LOAD:  next_state = (load_cnt == IN_DIM[4:0] - 5'd1 && x_valid) ? S_WPRE : S_LOAD;
        S_WPRE:  next_state = ((wpre_feat == IN_DIM[4:0] - 5'd1) && wpre_stream_r) ? S_MAC : S_WPRE;
        S_MAC:   next_state = mac_last ? ((neu_cnt == OUT_DIM[6:0] - 7'd1) ? S_DONE : S_WPRE)
                                        : S_MAC;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

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
        load_cnt    <= 5'd0;
        wpre_feat   <= 5'd0;
        wpre_stream <= 1'b0;
        mac_feat    <= 5'd0;
        neu_cnt     <= 7'd0;
        acc         <= 32'sd0;
        bias_hold   <= 16'sd0;
        y_o         <= 16'sd0;
        y_neu_o     <= 7'd0;
    end else begin
        case (state)
            S_IDLE: begin
                load_cnt    <= 5'd0;
                wpre_feat   <= 5'd0;
                wpre_stream <= 1'b0;
                mac_feat    <= 5'd0;
                neu_cnt     <= 7'd0;
                acc         <= 32'sd0;
                bias_hold   <= 16'sd0;
            end

            S_LOAD: begin
                if (x_valid) begin
                    if (load_cnt == IN_DIM[4:0] - 5'd1) begin
                        load_cnt    <= 5'd0;
                        wpre_feat   <= 5'd0;
                        wpre_stream <= 1'b0;
                        neu_cnt     <= 7'd0;
                        acc         <= 32'sd0;
                    end else begin
                        load_cnt <= load_cnt + 5'd1;
                    end
                end
            end

            S_WPRE: begin
                if (!wpre_stream)
                    wpre_stream <= 1'b1;
                else if (wpre_stream_r) begin
                    if (wpre_feat != IN_DIM[4:0] - 5'd1) begin
                        wpre_feat <= wpre_feat + 5'd1;
                    end else begin
                        wpre_stream <= 1'b0;
                        mac_feat    <= 5'd0;
                        acc         <= 32'sd0;
                    end
                end
            end

            S_MAC: begin
                if (mac_last) begin
                    y_o     <= y_next_c;
                    y_neu_o <= neu_cnt;
                    y_valid <= 1'b1;
                    if (neu_cnt == OUT_DIM[6:0] - 7'd1) begin
                        neu_cnt <= 7'd0;
                    end else begin
                        neu_cnt     <= neu_cnt + 7'd1;
                        wpre_feat   <= 5'd0;
                        wpre_stream <= 1'b0;
                    end
                end else begin
                    acc      <= acc + mac_prod;
                    mac_feat <= mac_feat + 5'd1;
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

`ifdef DUMP_WGT_PREFETCH
always @(posedge clk) begin
    if (DUMP_WGT && !reset && wgt_wr_ce && (neu_cnt == 7'd0))
        $display("[WGT_WR] neu=%0d slot=%0d w_i=%h",
                 neu_cnt, wpre_feat, w_i);
end

always @(posedge clk) begin
    if (DUMP_WGT && !reset && state == S_MAC && (neu_cnt == 7'd0) &&
        (mac_feat == 5'd0))
        $display("[WGT_CHK] wgt_buf[0..7]=%h %h %h %h %h %h %h %h gold=ffe2 002f ffe6 ffff 0015 0004 fff2 002a",
                 wgt_buf[0], wgt_buf[1], wgt_buf[2], wgt_buf[3],
                 wgt_buf[4], wgt_buf[5], wgt_buf[6], wgt_buf[7]);
end
`endif

endmodule
