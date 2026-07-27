// =============================================================================
// mlp_ws.v  (dim256)
//
// Q8.8 Weight-Stationary MLP: fc1 (EMBED_DIM=256 -> MLP_DIM=1024) -> ReLU ->
// fc2 (MLP_DIM=1024 -> EMBED_DIM=256).
//
// Dataflow: for each neuron group (LANES=8 neurons), preload weights into
// per-lane register files, then process all N_TOKENS=320. Each weight is read
// from ROM exactly once per output neuron.
//
// Per neuron group:
//   S_WLOAD : 2-phase ROM read -> w_lane{0..7}[0..IN_DIM-1]
//   S_BLOAD : 2-phase ROM read -> bias_reg[0..7]
//   S_MAC   : pipelined SRAM/norm read, 8-way broadcast MAC
//   S_SAT   : rnd_shr8 (+128) + bias + sat16 + ReLU(fc1) -> write output SRAM
//
// Sizes / parameters (dim256):
//   FC1_GROUPS = MLP_DIM / LANES   = 1024 / 8 = 128
//   FC2_GROUPS = EMBED_DIM / LANES =  256 / 8 =  32
//   FC1 WL_MAX = LANES * EMBED_DIM - 1 = 8 * 256 - 1  = 2047
//   FC2 WL_MAX = LANES * MLP_DIM   - 1 = 8 * 1024 - 1 = 8191
//   Weight ROM local index: neuron * IN_DIM + feat, 18 bits.
//
// SRAM allocation during MLP (all single-port, CLK = clk):
//   sram_q   : norm2 output (FC1 input); read via parent norm_rd interface.
//              parent flat = tok * EMBED_DIM + feat (17-bit) = {tok[8:0], feat[7:0]}
//   sram_k   : FC1 intermediate, tokens 0..127   (7-bit sub-tok)
//              layout: addr = tok * MLP_DIM + neuron = {tok[6:0], neuron[9:0]}
//              depth >= 128 * 1024 = 131072 (17-bit)
//   sram_v   : FC1 intermediate, tokens 128..255 (tok[6:0] naturally = tok-128)
//              same layout as sram_k, depth >= 131072
//   sram_qkm : FC1 intermediate, tokens 256..319 (tok[5:0] naturally = tok-256)
//              depth >= 64 * 1024 = 65536 (16-bit); high addr bit = 0
//   sram_ao  : FC2 output buffer, addr = tok * EMBED_DIM + feat = {tok[8:0], feat[7:0]}
//              depth >= 320 * 256 = 81920 (17-bit)
//
// SRAM read contract (1P, CLK = clk, posedge registered-data):
//   posedge T:   addr, CEB=0, WEB=1
//   posedge T+1: Q valid for addr@T  (macro Q registered on posedge clk)
// FC1 norm_rd and FC2 scratch read addresses are COMBINATIONAL (feat = mac_cnt).
// The address is presented one cycle before consumption so addr@T -> Q@T+1
// matches the existing 3-stage MAC pipeline (no prefetch / no registered addr).
//
// S_MAC 3-stage x path (breaks SRAM Q -> multiplier STA path):
//   posedge T:     issue x read (norm_rd / fc2 scratch); latch w_lane[T]
//   posedge T+1:   mac_x_r <= x@T; w_mul_r* <= w_rd_r*@T
//   posedge T+2:   prod_r <= w_mul_r * mac_x_r; acc += prod_r@T+1
//
// ROM read contract (CLK = clk, posedge registered-data):
//   posedge T:   wgt_addr_o / bias_addr_o
//   posedge T+1: wgt_i / bias_i valid for addr@T
//                (addr stable across each WLOAD/BLOAD 2-phase window,
//                 consumed on phase 1 -> unchanged)
//
// wgt_addr_o[19:0] = {1'b1, layer, local[17:0]}   (exactly 20 bits; do NOT use 2'b10
//                                                   prefix -- that is 21 bits and truncates
//                                                   bit[19] to 0 in Verilog-2001)
//   bit[19]        = 1        : MLP module identifier
//   bit[18]        = layer    : 0 = FC1, 1 = FC2
//   bits[17:0]     = local    : neuron * IN_DIM + feat (18-bit ROM index)
// bias_addr_o[9:0] = neuron index (group * 8 + lane); high bits zero for FC2.
//
// Golden: Activation/backbone_blocks_<b>_mlp_after_mlp_out_bi.txt
//         Flatten: row-major (tok, feat), tok*EMBED_DIM + feat.
// =============================================================================

module mlp_ws #(
    parameter EMBED_DIM = 256,
    parameter MLP_DIM   = 1024,
    parameter N_TOKENS  = 320,
    parameter LANES     = 8
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // FC1 input read (parent-owned sram_q)
    output reg         norm_rd_en,
    output reg [16:0]  norm_rd_flat,
    input  wire signed [15:0] norm_x,

    // ROM interface
    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [19:0] wgt_addr_o,
    output wire [9:0]  bias_addr_o,

    // FC1 scratch bank: tokens 0..127
    output reg         sram_k_ceb_o,
    output reg         sram_k_web_o,
    output reg  [16:0] sram_k_addr_o,
    output reg  [15:0] sram_k_din_o,
    input  wire [15:0] sram_k_q_i,

    // FC1 scratch bank: tokens 128..255
    output reg         sram_v_ceb_o,
    output reg         sram_v_web_o,
    output reg  [16:0] sram_v_addr_o,
    output reg  [15:0] sram_v_din_o,
    input  wire [15:0] sram_v_q_i,

    // FC1 scratch bank: tokens 256..319
    output reg         sram_qkm_ceb_o,
    output reg         sram_qkm_web_o,
    output reg  [16:0] sram_qkm_addr_o,
    output reg  [15:0] sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i,

    // FC2 output buffer + streaming readout
    output reg         sram_ao_ceb_o,
    output reg         sram_ao_web_o,
    output reg  [16:0] sram_ao_addr_o,
    output reg  [15:0] sram_ao_din_o,
    input  wire [15:0] sram_ao_q_i,

    output wire        busy,
    output reg         done,

    output reg  signed [15:0] y_o,
    output reg         y_valid
);

// -------------------------------------------------------------------------
// Parameters
// -------------------------------------------------------------------------
parameter S_IDLE    = 4'd0;
parameter S_WLOAD   = 4'd1;
parameter S_BLOAD   = 4'd2;
parameter S_MAC     = 4'd3;
parameter S_SAT     = 4'd4;
parameter S_OUT     = 4'd5;
parameter S_DONE_ST = 4'd6;

localparam FC1_GROUPS = MLP_DIM   / LANES;                  // 128
localparam FC2_GROUPS = EMBED_DIM / LANES;                  // 32
localparam FC1_WL_MAX = LANES * EMBED_DIM - 1;              // 2047
localparam FC2_WL_MAX = LANES * MLP_DIM   - 1;              // 8191
localparam OUT_TOTAL  = N_TOKENS * EMBED_DIM;               // 81920

// -------------------------------------------------------------------------
// Registers
// -------------------------------------------------------------------------
reg [3:0]  state;
reg [3:0]  next_state;

reg        layer;                                           // 0 = FC1, 1 = FC2
reg [7:0]  group_cnt;                                       // 0..127 (FC1) / 0..31 (FC2)
reg [8:0]  tok_cnt;                                         // 0..319

reg [12:0] wl_cnt;                                          // 0..8191 (FC2 max)
reg        wl_phase;

reg [3:0]  bl_cnt;                                          // 0..7
reg        bl_phase;

reg [10:0] mac_cnt;                                         // 0..1026 (mac_limit+2 for FC2)

reg [3:0]  sat_lane;                                        // 0..9

reg [16:0] out_cnt;                                         // 0..OUT_TOTAL-1
reg        out_phase;

// Weight register files: sized for FC2 (IN_DIM = MLP_DIM = 1024).
// FC1 uses only entries [0..EMBED_DIM-1] = [0..255].
reg signed [15:0] w_lane0 [0:1023];
reg signed [15:0] w_lane1 [0:1023];
reg signed [15:0] w_lane2 [0:1023];
reg signed [15:0] w_lane3 [0:1023];
reg signed [15:0] w_lane4 [0:1023];
reg signed [15:0] w_lane5 [0:1023];
reg signed [15:0] w_lane6 [0:1023];
reg signed [15:0] w_lane7 [0:1023];

reg signed [15:0] bias_reg [0:7];
reg signed [31:0] acc      [0:7];

reg signed [31:0] prod_r [0:7];

reg signed [31:0] acc_pick;
reg signed [15:0] bias_pick;

reg signed [15:0] w_rd_r0;
reg signed [15:0] w_rd_r1;
reg signed [15:0] w_rd_r2;
reg signed [15:0] w_rd_r3;
reg signed [15:0] w_rd_r4;
reg signed [15:0] w_rd_r5;
reg signed [15:0] w_rd_r6;
reg signed [15:0] w_rd_r7;

reg signed [15:0] mac_x_r;
reg signed [15:0] w_mul_r0;
reg signed [15:0] w_mul_r1;
reg signed [15:0] w_mul_r2;
reg signed [15:0] w_mul_r3;
reg signed [15:0] w_mul_r4;
reg signed [15:0] w_mul_r5;
reg signed [15:0] w_mul_r6;
reg signed [15:0] w_mul_r7;

reg signed [15:0] sat_val_r;
reg               sat_wr_pending;
reg [2:0]         sat_wr_lane;

reg signed [31:0] sat_mid_r;
reg               sat_s1_valid;
reg [2:0]         sat_s1_lane;

integer i_lane;

`ifndef SYNTHESIS
integer init_i;
initial begin
    for (init_i = 0; init_i < 1024; init_i = init_i + 1) begin
        w_lane0[init_i] = 16'sd0; w_lane1[init_i] = 16'sd0;
        w_lane2[init_i] = 16'sd0; w_lane3[init_i] = 16'sd0;
        w_lane4[init_i] = 16'sd0; w_lane5[init_i] = 16'sd0;
        w_lane6[init_i] = 16'sd0; w_lane7[init_i] = 16'sd0;
    end
    for (init_i = 0; init_i < 8; init_i = init_i + 1) begin
        bias_reg[init_i] = 16'sd0;
        acc[init_i]      = 32'sd0;
    end
end
`endif

// -------------------------------------------------------------------------
// Wires
// -------------------------------------------------------------------------
// MAC iteration length (per-neuron dot-product depth).
// CRITICAL: use full 11-bit; EMBED_DIM[7:0] would truncate 256 -> 0.
wire [10:0] mac_limit = (layer == 1'b0) ? 11'd256 : 11'd1024;

wire        mac_x_cap = (state == S_MAC) && (mac_cnt >= 11'd1) &&
                        (mac_cnt <= mac_limit);
wire        mul_en    = (state == S_MAC) && (mac_cnt >= 11'd2) &&
                        (mac_cnt <= mac_limit + 11'd1);
wire        acc_en    = (state == S_MAC) && (mac_cnt >= 11'd3) &&
                        (mac_cnt <= mac_limit + 11'd2);

// Weight preload counter unpacking (packed = {lane[2:0], feat[FEAT_W-1:0]}):
//   FC1: FEAT_W = 8  -> lane = wl_cnt[10:8], feat = wl_cnt[7:0]
//   FC2: FEAT_W = 10 -> lane = wl_cnt[12:10], feat = wl_cnt[9:0]
wire [2:0]  wl_lane = (layer == 1'b0) ? wl_cnt[10:8] : wl_cnt[12:10];
wire [9:0]  wl_feat = (layer == 1'b0) ? {2'b0, wl_cnt[7:0]} : wl_cnt[9:0];

// Absolute output-neuron index for this group + lane.
//   FC1: neu = group_cnt[6:0]*8 + wl_lane -> 10 bits (0..1023)
//   FC2: neu = group_cnt[4:0]*8 + wl_lane ->  8 bits (0..255)
// Concatenation gives 10-bit; upper bits are zero for FC2.
wire [9:0]  wl_neu  = {group_cnt[6:0], wl_lane[2:0]};

// Weight ROM local address = neuron * IN_DIM + feat (18-bit).
//   FC1: {neu[9:0], feat[7:0]}
//   FC2: {neu[7:0], feat[9:0]}
wire [17:0] wgt_local = (layer == 1'b0) ? {wl_neu[9:0], wl_feat[7:0]}
                                        : {wl_neu[7:0], wl_feat[9:0]};

wire        wl_last  = (layer == 1'b0) ? (wl_cnt == FC1_WL_MAX[12:0])
                                       : (wl_cnt == FC2_WL_MAX[12:0]);
wire        wl_done  = wl_last && (wl_phase == 1'b1);
wire        bl_done  = (bl_cnt == 4'd7) && (bl_phase == 1'b1);
wire        mac_done = (mac_cnt == mac_limit + 11'd2);

wire        sat_done   = (sat_lane == 4'd9);
wire        tok_last   = (tok_cnt == N_TOKENS[8:0] - 9'd1);
wire        group_last = (layer == 1'b0) ? (group_cnt == FC1_GROUPS[7:0] - 8'd1)
                                         : (group_cnt == FC2_GROUPS[7:0] - 8'd1);
wire        out_done   = (out_cnt == OUT_TOTAL[16:0] - 17'd1) &&
                        (out_phase == 1'b1);

// FC2 x source (post-ReLU FC1 output) demux by token range.
wire signed [15:0] fc2_x = (tok_cnt < 9'd128) ? $signed(sram_k_q_i) :
                           (tok_cnt < 9'd256) ? $signed(sram_v_q_i) :
                                                $signed(sram_qkm_q_i);
wire signed [15:0] mac_x = (layer == 1'b0) ? norm_x : fc2_x;

// FC2 scratch read enable (combinational; feat = mac_cnt).
wire        fc2_sram_rd_fire      = (state == S_MAC) && (layer == 1'b1) &&
                                    (mac_cnt < mac_limit);
wire [9:0]  fc2_sram_rd_feat_mux  = mac_cnt[9:0];

assign busy = (state != S_IDLE);

// -------------------------------------------------------------------------
// ROM address
//   wgt_addr_o: {1'b1, layer, local[17:0]} (20-bit; bit[19]=1 marks MLP).
//   bias_addr_o: 10-bit neuron index = {group_cnt[6:0], bl_cnt[2:0]}.
// -------------------------------------------------------------------------
assign wgt_addr_o = (state == S_WLOAD) ? {1'b1, layer, wgt_local[17:0]} :
                    (state == S_BLOAD) ? {1'b1, layer, 18'd0}           :
                    20'd0;

assign bias_addr_o = (state == S_BLOAD) ? {group_cnt[6:0], bl_cnt[2:0]}
                                        : 10'd0;

// -------------------------------------------------------------------------
// SAT stage combinational
// -------------------------------------------------------------------------
wire signed [31:0] sat_shr8   = (acc_pick + 32'sd128) >>> 8;
wire signed [31:0] sat_add_b  = sat_shr8 + {{16{bias_pick[15]}}, bias_pick};
wire signed [15:0] sat_pre    =
    (sat_mid_r > 32'sd32767)  ? 16'sh7FFF :
    (sat_mid_r < -32'sd32768) ? 16'sh8000 :
    sat_mid_r[15:0];
wire signed [15:0] sat_val = (layer == 1'b0 && sat_pre[15]) ? 16'sd0 : sat_pre;

// -------------------------------------------------------------------------
// FSM segment 1: state register
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state logic
always @(*) begin
    next_state = state;
    case (state)
        S_IDLE:    if (start)   next_state = S_WLOAD;
        S_WLOAD:   if (wl_done) next_state = S_BLOAD;
        S_BLOAD:   if (bl_done) next_state = S_MAC;
        S_MAC:     if (mac_done) next_state = S_SAT;
        S_SAT: begin
            if (sat_done && !tok_last)
                next_state = S_MAC;
            else if (sat_done && tok_last && !group_last)
                next_state = S_WLOAD;
            else if (sat_done && tok_last && group_last && layer == 1'b0)
                next_state = S_WLOAD;
            else if (sat_done && tok_last && group_last)
                next_state = S_OUT;
        end
        S_OUT:     if (out_done) next_state = S_DONE_ST;
        S_DONE_ST: next_state = S_IDLE;
        default:   next_state = S_IDLE;
    endcase
end

// -------------------------------------------------------------------------
// Layer register (0 = FC1, 1 = FC2)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        layer <= 1'b0;
    else if (state == S_IDLE && start)
        layer <= 1'b0;
    else if (state == S_SAT && sat_done && tok_last && group_last && layer == 1'b0)
        layer <= 1'b1;
end

// Group counter
always @(posedge clk) begin
    if (reset)
        group_cnt <= 8'd0;
    else if (state == S_IDLE)
        group_cnt <= 8'd0;
    else if (state == S_SAT && sat_done && tok_last && !group_last)
        group_cnt <= group_cnt + 8'd1;
    else if (state == S_SAT && sat_done && tok_last && group_last)
        group_cnt <= 8'd0;
end

// Token counter
always @(posedge clk) begin
    if (reset)
        tok_cnt <= 9'd0;
    else if (state == S_IDLE)
        tok_cnt <= 9'd0;
    else if (state == S_SAT && sat_done && !tok_last)
        tok_cnt <= tok_cnt + 9'd1;
    else if (state == S_SAT && sat_done && tok_last)
        tok_cnt <= 9'd0;
end

// -------------------------------------------------------------------------
// Weight preload counter + 2-phase ROM read.
//   posedge T:   wgt_addr_o (phase 0)
//   posedge T+1: wgt_i valid, capture into w_laneN[wl_feat]; advance wl_cnt.
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        wl_cnt   <= 13'd0;
        wl_phase <= 1'b0;
    end else if (state != S_WLOAD) begin
        wl_cnt   <= 13'd0;
        wl_phase <= 1'b0;
    end else if (wl_phase == 1'b0)
        wl_phase <= 1'b1;
    else if (!wl_last) begin
        wl_phase <= 1'b0;
        wl_cnt   <= wl_cnt + 13'd1;
    end else
        wl_phase <= 1'b0;
end

// Weight capture into per-lane register files.
// FC1 writes indices 0..255 (upper 768 slots unused for that layer).
always @(posedge clk) begin
    if (state == S_WLOAD && wl_phase == 1'b1) begin
        case (wl_lane)
            3'd0: w_lane0[wl_feat] <= wgt_i;
            3'd1: w_lane1[wl_feat] <= wgt_i;
            3'd2: w_lane2[wl_feat] <= wgt_i;
            3'd3: w_lane3[wl_feat] <= wgt_i;
            3'd4: w_lane4[wl_feat] <= wgt_i;
            3'd5: w_lane5[wl_feat] <= wgt_i;
            3'd6: w_lane6[wl_feat] <= wgt_i;
            3'd7: w_lane7[wl_feat] <= wgt_i;
            default: ;
        endcase
    end
end

// -------------------------------------------------------------------------
// Bias preload counter + 2-phase ROM read.
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        bl_cnt   <= 4'd0;
        bl_phase <= 1'b0;
    end else if (state != S_BLOAD) begin
        bl_cnt   <= 4'd0;
        bl_phase <= 1'b0;
    end else if (bl_phase == 1'b0)
        bl_phase <= 1'b1;
    else if (bl_cnt != 4'd7) begin
        bl_phase <= 1'b0;
        bl_cnt   <= bl_cnt + 4'd1;
    end else
        bl_phase <= 1'b0;
end

// Bias capture
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            bias_reg[i_lane] <= 16'sd0;
    end else if (state == S_BLOAD && bl_phase == 1'b1)
        bias_reg[bl_cnt[2:0]] <= bias_i;
end

// -------------------------------------------------------------------------
// MAC counter (pipelined: 0 = issue x[0] + latch w[0];
//                        1 = capture; 2..mac_limit+1 = MUL + issue; last +1 = flush)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        mac_cnt <= 11'd0;
    else if (state != S_MAC)
        mac_cnt <= 11'd0;
    else if (!mac_done)
        mac_cnt <= mac_cnt + 11'd1;
end

// Weight read pipeline register (cuts reg-file 1024:1 mux from MAC critical path)
always @(posedge clk) begin
    if (state == S_MAC && mac_cnt < mac_limit) begin
        w_rd_r0 <= w_lane0[mac_cnt[9:0]];
        w_rd_r1 <= w_lane1[mac_cnt[9:0]];
        w_rd_r2 <= w_lane2[mac_cnt[9:0]];
        w_rd_r3 <= w_lane3[mac_cnt[9:0]];
        w_rd_r4 <= w_lane4[mac_cnt[9:0]];
        w_rd_r5 <= w_lane5[mac_cnt[9:0]];
        w_rd_r6 <= w_lane6[mac_cnt[9:0]];
        w_rd_r7 <= w_lane7[mac_cnt[9:0]];
    end
end

// MAC x/w capture (1 cycle after SRAM Q valid; pairs with w_rd_r@issue)
always @(posedge clk) begin
    if (reset) begin
        mac_x_r  <= 16'sd0;
        w_mul_r0 <= 16'sd0;
        w_mul_r1 <= 16'sd0;
        w_mul_r2 <= 16'sd0;
        w_mul_r3 <= 16'sd0;
        w_mul_r4 <= 16'sd0;
        w_mul_r5 <= 16'sd0;
        w_mul_r6 <= 16'sd0;
        w_mul_r7 <= 16'sd0;
    end else if (mac_x_cap) begin
        mac_x_r  <= mac_x;
        w_mul_r0 <= w_rd_r0;
        w_mul_r1 <= w_rd_r1;
        w_mul_r2 <= w_rd_r2;
        w_mul_r3 <= w_rd_r3;
        w_mul_r4 <= w_rd_r4;
        w_mul_r5 <= w_rd_r5;
        w_mul_r6 <= w_rd_r6;
        w_mul_r7 <= w_rd_r7;
    end
end

// MAC multiply stage (reg x reg; SRAM Q not on comb multiplier input)
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            prod_r[i_lane] <= 32'sd0;
    end else if (mul_en) begin
        prod_r[0] <= w_mul_r0 * mac_x_r;
        prod_r[1] <= w_mul_r1 * mac_x_r;
        prod_r[2] <= w_mul_r2 * mac_x_r;
        prod_r[3] <= w_mul_r3 * mac_x_r;
        prod_r[4] <= w_mul_r4 * mac_x_r;
        prod_r[5] <= w_mul_r5 * mac_x_r;
        prod_r[6] <= w_mul_r6 * mac_x_r;
        prod_r[7] <= w_mul_r7 * mac_x_r;
    end
end

// Accumulator update
always @(posedge clk) begin
    if (reset) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc[i_lane] <= 32'sd0;
    end else if (state == S_MAC && mac_cnt == 11'd0) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc[i_lane] <= 32'sd0;
    end else if (acc_en) begin
        for (i_lane = 0; i_lane < LANES; i_lane = i_lane + 1)
            acc[i_lane] <= acc[i_lane] + prod_r[i_lane];
    end
end

// -------------------------------------------------------------------------
// SAT counter (8 lanes + 2 pipeline flush cycles -> 0..9)
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset)
        sat_lane <= 4'd0;
    else if (state != S_SAT)
        sat_lane <= 4'd0;
    else
        sat_lane <= sat_lane + 4'd1;
end

// SAT accumulator and bias mux (combinational)
always @(*) begin
    acc_pick  = 32'sd0;
    bias_pick = 16'sd0;
    case (sat_lane[2:0])
        3'd0: begin acc_pick = acc[0]; bias_pick = bias_reg[0]; end
        3'd1: begin acc_pick = acc[1]; bias_pick = bias_reg[1]; end
        3'd2: begin acc_pick = acc[2]; bias_pick = bias_reg[2]; end
        3'd3: begin acc_pick = acc[3]; bias_pick = bias_reg[3]; end
        3'd4: begin acc_pick = acc[4]; bias_pick = bias_reg[4]; end
        3'd5: begin acc_pick = acc[5]; bias_pick = bias_reg[5]; end
        3'd6: begin acc_pick = acc[6]; bias_pick = bias_reg[6]; end
        3'd7: begin acc_pick = acc[7]; bias_pick = bias_reg[7]; end
        default: ;
    endcase
end

// SAT pipeline stage 1 (8:1 mux + >>>8 + bias add -> sat_mid_r)
always @(posedge clk) begin
    if (reset) begin
        sat_mid_r    <= 32'sd0;
        sat_s1_valid <= 1'b0;
        sat_s1_lane  <= 3'd0;
    end else if (state == S_SAT && sat_lane <= 4'd7) begin
        sat_mid_r    <= sat_add_b;
        sat_s1_valid <= 1'b1;
        sat_s1_lane  <= sat_lane[2:0];
    end else begin
        sat_s1_valid <= 1'b0;
    end
end

// SAT pipeline stage 2 (clamp + ReLU -> sat_val_r; drive sat_wr_pending)
always @(posedge clk) begin
    if (reset) begin
        sat_val_r      <= 16'sd0;
        sat_wr_pending <= 1'b0;
        sat_wr_lane    <= 3'd0;
    end else if (sat_s1_valid) begin
        sat_val_r      <= sat_val;
        sat_wr_pending <= 1'b1;
        sat_wr_lane    <= sat_s1_lane;
    end else begin
        sat_wr_pending <= 1'b0;
    end
end

// -------------------------------------------------------------------------
// Output readout counter + 2-phase read from sram_ao.
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        out_cnt   <= 17'd0;
        out_phase <= 1'b0;
    end else if (state != S_OUT) begin
        out_cnt   <= 17'd0;
        out_phase <= 1'b0;
    end else if (out_phase == 1'b0)
        out_phase <= 1'b1;
    else if (out_cnt < OUT_TOTAL[16:0] - 17'd1) begin
        out_phase <= 1'b0;
        out_cnt   <= out_cnt + 17'd1;
    end else
        out_phase <= 1'b0;
end

// -------------------------------------------------------------------------
// Norm read control (FC1 MAC: pipelined read from parent sram_q).
//   CLK(clk): combinational read address. Present feat = mac_cnt during cycle T
//   so the posedge macro returns norm_x in T+1; mac_x_cap@T+1 latches into
//   mac_x_r, pairing with w_mul_r <= w_lane[T]. A registered norm_rd would slip
//   x one cycle late.
//   flat = tok * EMBED_DIM + feat = {tok_cnt[8:0], mac_cnt[7:0]} (17-bit).
// -------------------------------------------------------------------------
always @(*) begin
    if (state == S_MAC && layer == 1'b0 && mac_cnt < mac_limit) begin
        norm_rd_en   = 1'b1;
        norm_rd_flat = {tok_cnt[8:0], mac_cnt[7:0]};
    end else begin
        norm_rd_en   = 1'b0;
        norm_rd_flat = 17'd0;
    end
end

// -------------------------------------------------------------------------
// FC2 scratch SRAM read: CLK(clk) combinational addr (feat = mac_cnt) in mux.
//   posedge T:     issue scratch[mac_cnt]; latch w_lane[mac_cnt] -> w_rd_r
//   posedge T+1:   sram_*_q valid for feat@T; mac_x_cap latches mac_x_r / w_mul_r
//   posedge T+2:   prod_r <= w_mul_r * mac_x_r
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Output y_o / y_valid
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        y_o     <= 16'sd0;
        y_valid <= 1'b0;
    end else if (state == S_OUT && out_phase == 1'b1) begin
        y_o     <= $signed(sram_ao_q_i);
        y_valid <= 1'b1;
    end else
        y_valid <= 1'b0;
end

// Done pulse
always @(posedge clk) begin
    if (reset)
        done <= 1'b0;
    else if (state == S_DONE_ST)
        done <= 1'b1;
    else
        done <= 1'b0;
end

// -------------------------------------------------------------------------
// SRAM mux (combinational; one if/else-if chain per macro)
//   sram_k/v/qkm : FC1 SAT write | FC2 MAC scratch read
//   sram_ao      : FC2 SAT write | S_OUT read
//
// FC1 SAT write address (per bank):
//   addr = subtok * MLP_DIM + neuron
//        = {subtok[6:0], group_cnt[6:0], sat_wr_lane[2:0]}   (17-bit for k/v)
//        = {1'b0, subtok[5:0], group_cnt[6:0], sat_wr_lane[2:0]} (17-bit qkm)
// FC2 MAC scratch read address:
//   addr = subtok * MLP_DIM + feat
//        = {subtok[6:0], feat[9:0]}   (17-bit for k/v)
//        = {1'b0, subtok[5:0], feat[9:0]} (17-bit qkm)
// -------------------------------------------------------------------------

// sram_k: FC1 SAT write (tokens 0..127) or FC2 scratch read
always @(*) begin
    sram_k_ceb_o  = 1'b1;
    sram_k_web_o  = 1'b1;
    sram_k_addr_o = 17'd0;
    sram_k_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && layer == 1'b0 &&
        tok_cnt < 9'd128) begin
        sram_k_ceb_o  = 1'b0;
        sram_k_web_o  = 1'b0;
        sram_k_addr_o = {tok_cnt[6:0], group_cnt[6:0], sat_wr_lane[2:0]};
        sram_k_din_o  = sat_val_r;
    end else if ((state == S_BLOAD || state == S_MAC) &&
                 layer == 1'b1 && fc2_sram_rd_fire &&
                 tok_cnt < 9'd128) begin
        sram_k_ceb_o  = 1'b0;
        sram_k_web_o  = 1'b1;
        sram_k_addr_o = {tok_cnt[6:0], fc2_sram_rd_feat_mux[9:0]};
    end
end

// sram_v: FC1 SAT write (tokens 128..255) or FC2 scratch read
always @(*) begin
    sram_v_ceb_o  = 1'b1;
    sram_v_web_o  = 1'b1;
    sram_v_addr_o = 17'd0;
    sram_v_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && layer == 1'b0 &&
        tok_cnt >= 9'd128 && tok_cnt < 9'd256) begin
        sram_v_ceb_o  = 1'b0;
        sram_v_web_o  = 1'b0;
        sram_v_addr_o = {tok_cnt[6:0], group_cnt[6:0], sat_wr_lane[2:0]};
        sram_v_din_o  = sat_val_r;
    end else if ((state == S_BLOAD || state == S_MAC) &&
                 layer == 1'b1 && fc2_sram_rd_fire &&
                 tok_cnt >= 9'd128 && tok_cnt < 9'd256) begin
        sram_v_ceb_o  = 1'b0;
        sram_v_web_o  = 1'b1;
        sram_v_addr_o = {tok_cnt[6:0], fc2_sram_rd_feat_mux[9:0]};
    end
end

// sram_qkm: FC1 SAT write (tokens 256..319) or FC2 scratch read
always @(*) begin
    sram_qkm_ceb_o  = 1'b1;
    sram_qkm_web_o  = 1'b1;
    sram_qkm_addr_o = 17'd0;
    sram_qkm_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && layer == 1'b0 &&
        tok_cnt >= 9'd256) begin
        sram_qkm_ceb_o  = 1'b0;
        sram_qkm_web_o  = 1'b0;
        sram_qkm_addr_o = {1'b0, tok_cnt[5:0], group_cnt[6:0], sat_wr_lane[2:0]};
        sram_qkm_din_o  = sat_val_r;
    end else if ((state == S_BLOAD || state == S_MAC) &&
                 layer == 1'b1 && fc2_sram_rd_fire &&
                 tok_cnt >= 9'd256) begin
        sram_qkm_ceb_o  = 1'b0;
        sram_qkm_web_o  = 1'b1;
        sram_qkm_addr_o = {1'b0, tok_cnt[5:0], fc2_sram_rd_feat_mux[9:0]};
    end
end

// sram_ao: FC2 SAT write or S_OUT read
//   FC2 SAT write addr = tok * EMBED_DIM + neuron
//                      = {tok_cnt[8:0], group_cnt[4:0], sat_wr_lane[2:0]} (17-bit)
//   S_OUT read addr = out_cnt (linear over N_TOKENS * EMBED_DIM = 81920)
always @(*) begin
    sram_ao_ceb_o  = 1'b1;
    sram_ao_web_o  = 1'b1;
    sram_ao_addr_o = 17'd0;
    sram_ao_din_o  = 16'd0;
    if (state == S_SAT && sat_wr_pending && layer == 1'b1) begin
        sram_ao_ceb_o  = 1'b0;
        sram_ao_web_o  = 1'b0;
        sram_ao_addr_o = {tok_cnt[8:0], group_cnt[4:0], sat_wr_lane[2:0]};
        sram_ao_din_o  = sat_val_r;
    end else if (state == S_OUT && out_phase == 1'b0) begin
        sram_ao_ceb_o  = 1'b0;
        sram_ao_web_o  = 1'b1;
        sram_ao_addr_o = out_cnt;
    end
end

endmodule
