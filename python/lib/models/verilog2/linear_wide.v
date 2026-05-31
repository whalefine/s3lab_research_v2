// =============================================================================
// linear_wide.v
//
// Q8.8 linear (MAC + sat16) — fc2 variant with IN_DIM=128, OUT_DIM=32.
// A2: same ROM/MAC pipeline as linear.v (no wgt_buf / S_WPRE).
//
// w_addr_o packing (backbone_top fc2 decode):
//   {1'b0, neu[4:0], feat[6:0]}  -> local[11:7]=neu, local[6:0]=feat
//
// Per output neuron:
//   MAC_PREF (1 cycle): w_addr_o = {neu, 0}.
//   MAC_RUN  (IN_DIM cycles): acc += x_buf[k] * w_i_lat (w_i latched prev posedge).
//   mac_last keeps feat = 127 (7-bit feat+1 would wrap to 0).
//
// Used only by mlp.v u_fc2 (fc1 uses linear.v).
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
parameter S_MAC  = 3'd2;
parameter S_DONE = 3'd3;

parameter MAC_PREF = 1'b0;
parameter MAC_RUN  = 1'b1;

reg signed [15:0] x_buf    [0:IN_DIM-1];
reg signed [15:0] bias_hold;

reg [2:0] state, next_state;
reg [6:0] load_cnt;
reg [6:0] mac_feat;
reg [4:0] neu_cnt;
reg       mac_sub;

reg signed [31:0] acc;
reg signed [15:0] w_i_lat;

wire [4:0] neu_for_addr = neu_cnt;

wire mac_last = (mac_feat == IN_DIM[6:0] - 7'd1);

wire [6:0] feat_for_addr =
    (state != S_MAC) ? 7'd0 :
    (mac_sub == MAC_PREF) ? 7'd0 :
    mac_last ? (IN_DIM[6:0] - 7'd1) :
    (mac_feat + 7'd1);

assign w_addr_o = {1'b0, neu_for_addr, feat_for_addr};

wire bias_latch_ce =
    (state == S_MAC) && (mac_sub == MAC_RUN) && (mac_feat == 7'd0);

wire w_i_lat_ce =
    (state == S_MAC) && ((mac_sub == MAC_PREF) || (mac_sub == MAC_RUN));

wire signed [31:0] mac_prod =
    (state == S_MAC) && (mac_sub == MAC_RUN) ?
        ($signed(x_buf[mac_feat]) * $signed(w_i_lat)) : 32'sd0;

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

always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
end

always @(*) begin
    case (state)
        S_IDLE:  next_state = start ? S_LOAD : S_IDLE;
        S_LOAD:  next_state = (load_cnt == IN_DIM[6:0] - 7'd1 && x_valid) ? S_MAC : S_LOAD;
        S_MAC:   next_state = (mac_sub == MAC_RUN) && mac_last &&
                              (neu_cnt == OUT_DIM[4:0] - 5'd1) ? S_DONE : S_MAC;
        S_DONE:  next_state = S_IDLE;
        default: next_state = S_IDLE;
    endcase
end

always @(posedge clk) begin
    done    <= 1'b0;
    y_valid <= 1'b0;

    if (state == S_LOAD && x_valid)
        x_buf[load_cnt] <= x_i;

    if (bias_latch_ce)
        bias_hold <= b_i;

    if (w_i_lat_ce)
        w_i_lat <= w_i;

    if (reset) begin
        load_cnt  <= 7'd0;
        mac_feat  <= 7'd0;
        neu_cnt   <= 5'd0;
        mac_sub   <= MAC_PREF;
        acc       <= 32'sd0;
        bias_hold <= 16'sd0;
        w_i_lat   <= 16'sd0;
        y_o       <= 16'sd0;
        y_neu_o   <= 5'd0;
    end else begin
        case (state)
            S_IDLE: begin
                load_cnt  <= 7'd0;
                mac_feat  <= 7'd0;
                neu_cnt   <= 5'd0;
                mac_sub   <= MAC_PREF;
                acc       <= 32'sd0;
                bias_hold <= 16'sd0;
                w_i_lat   <= 16'sd0;
            end

            S_LOAD: begin
                if (x_valid) begin
                    if (load_cnt == IN_DIM[6:0] - 7'd1) begin
                        load_cnt  <= 7'd0;
                        mac_feat  <= 7'd0;
                        neu_cnt   <= 5'd0;
                        mac_sub   <= MAC_PREF;
                        acc       <= 32'sd0;
                    end else begin
                        load_cnt <= load_cnt + 7'd1;
                    end
                end
            end

            S_MAC: begin
                if (mac_sub == MAC_PREF) begin
                    mac_sub  <= MAC_RUN;
                    mac_feat <= 7'd0;
                    acc      <= 32'sd0;
                end else begin
                    if (mac_last) begin
                        y_o     <= y_next_c;
                        y_neu_o <= neu_cnt;
                        y_valid <= 1'b1;
                        if (neu_cnt == OUT_DIM[4:0] - 5'd1) begin
                            neu_cnt <= 5'd0;
                        end else begin
                            neu_cnt   <= neu_cnt + 5'd1;
                            mac_sub   <= MAC_PREF;
                            mac_feat  <= 7'd0;
                            acc       <= 32'sd0;
                        end
                    end else begin
                        acc      <= acc + mac_prod;
                        mac_feat <= mac_feat + 7'd1;
                    end
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
