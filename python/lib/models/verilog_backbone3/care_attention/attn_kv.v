// =============================================================================
// attn_kv.v
//
// CARE attention S_KV: parallel read split k (Sram_k) + raw v (Sram_v),
// mean over tokens -> kv_buf[NUM_HEADS*HEAD_DIM*HEAD_DIM].
// Bit-accurate vs run_backbone_numpy_shared_trunk.py _care_kv_q88 and
// verilog_backbone2/care_attention.v S_KV.
//
// kv[h, d_out, d_k] = sat16_48((sum_n k[h,n,d_out]*v[h,n,d_k] * RCP_N_NUM
//                               + KV_Q88_ROUND) >> (RCP_N_SHIFT+8))
//
// SRAM read: 1-cycle latency (phase-0 ADDR, phase-1 USE accumulates sram_*_q_i).
// Flat (H,N,d): ptr = h*(N_TOKENS*HEAD_DIM) + n*HEAD_DIM + d
// kv_oidx: h=[7:6], d_out=[5:3], d_k=[2:0]
// Golden: numpy _care_kv_q88 (k must be post-S_SPLIT; v is raw QKV v)
// =============================================================================

module attn_kv #(
    parameter NUM_HEADS    = 4,
    parameter HEAD_DIM     = 8,
    parameter N_TOKENS     = 320,
    parameter RCP_N_NUM    = 205,
    parameter RCP_N_SHIFT  = 16,
    parameter KV_Q88_ROUND = 8388608   // 2^23, >>>24 round
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output wire        busy,
    output reg         done,

    output reg         sram_k_ceb_o,
    output reg         sram_k_web_o,
    output reg [13:0]  sram_k_addr_o,
    input  wire [15:0] sram_k_q_i,

    output reg         sram_v_ceb_o,
    output reg         sram_v_web_o,
    output reg [13:0]  sram_v_addr_o,
    input  wire [15:0] sram_v_q_i
);

localparam KV_ELEMS = NUM_HEADS * HEAD_DIM * HEAD_DIM;

reg signed [15:0] kv_buf [0:KV_ELEMS-1];

parameter S_IDLE    = 2'd0;
parameter S_KV      = 2'd1;
parameter S_DONE_ST = 2'd2;

reg [1:0] state;
reg [1:0] next_state;

reg [7:0]         kv_oidx;
reg [8:0]         kv_n;
reg signed [47:0] kv_acc;
reg               kv_phase;

wire [1:0]        kv_h;
wire [2:0]        kv_d1;
wire [2:0]        kv_d2;
wire [13:0]       kv_k_flat;
wire [13:0]       kv_v_flat;

wire signed [15:0] kv_k_op;
wire signed [15:0] kv_v_op;
wire signed [31:0] kv_term;
wire signed [48:0] kv_acc_next;
wire signed [63:0] kv_scaled;
wire signed [63:0] kv_shr_w;
wire signed [15:0] kv_wr_val;

assign kv_h      = kv_oidx[7:6];
assign kv_d1     = kv_oidx[5:3];
assign kv_d2     = kv_oidx[2:0];
assign kv_k_flat =
    {12'd0, kv_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, kv_n}) * HEAD_DIM
  + {11'd0, kv_d1};
assign kv_v_flat =
    {12'd0, kv_h} * (N_TOKENS * HEAD_DIM)
  + ({5'd0, kv_n}) * HEAD_DIM
  + {11'd0, kv_d2};

assign kv_k_op = $signed(sram_k_q_i);
assign kv_v_op = $signed(sram_v_q_i);
assign kv_term = $signed(kv_k_op) * $signed(kv_v_op);
assign kv_acc_next =
    (kv_n == 9'd0) ? $signed({{17{kv_term[31]}}, kv_term})
                   : kv_acc + $signed({{17{kv_term[31]}}, kv_term});
assign kv_scaled =
    $signed({{15{kv_acc_next[48]}}, kv_acc_next}) * $signed({48'd0, RCP_N_NUM[15:0]});
assign kv_shr_w =
    (kv_scaled + $signed({32'd0, KV_Q88_ROUND[31:0]})) >>> (RCP_N_SHIFT + 8);
assign kv_wr_val =
    (kv_shr_w > 64'sd32767) ? 16'sh7FFF :
    (kv_shr_w < -64'sd32768) ? 16'sh8000 : kv_shr_w[15:0];

assign busy = (state != S_IDLE);

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
        S_IDLE: next_state = start ? S_KV : S_IDLE;
        S_KV:   next_state = (kv_oidx == KV_ELEMS[7:0] - 8'd1 &&
                              kv_n == N_TOKENS[8:0] - 9'd1 &&
                              kv_phase == 1'b1) ? S_DONE_ST : S_KV;
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

// S_KV datapath
always @(posedge clk) begin
    if (reset) begin
        kv_oidx  <= 8'd0;
        kv_n     <= 9'd0;
        kv_acc   <= 48'sd0;
        kv_phase <= 1'b0;
    end else begin
        case (state)
            S_IDLE: begin
                kv_oidx  <= 8'd0;
                kv_n     <= 9'd0;
                kv_acc   <= 48'sd0;
                kv_phase <= 1'b0;
            end

            S_KV: begin
                if (kv_phase == 1'b0)
                    kv_phase <= 1'b1;
                else begin
                    kv_acc <= kv_acc_next;
                    if (kv_n == N_TOKENS[8:0] - 9'd1) begin
                        kv_buf[kv_oidx] <= kv_wr_val;
                        kv_n <= 9'd0;
                        if (kv_oidx == KV_ELEMS[7:0] - 8'd1)
                            kv_oidx <= 8'd0;
                        else
                            kv_oidx <= kv_oidx + 8'd1;
                    end else begin
                        kv_n <= kv_n + 9'd1;
                    end
                    kv_phase <= 1'b0;
                end
            end

            default: ;
        endcase
    end
end

// SRAM port mux: parallel read k + v (both phases)
always @(*) begin
    sram_k_ceb_o  = 1'b1;
    sram_k_web_o  = 1'b1;
    sram_k_addr_o = 14'd0;
    sram_v_ceb_o  = 1'b1;
    sram_v_web_o  = 1'b1;
    sram_v_addr_o = 14'd0;

    if (state == S_KV) begin
        sram_k_ceb_o  = 1'b0;
        sram_k_web_o  = 1'b1;
        sram_k_addr_o = kv_k_flat;
        sram_v_ceb_o  = 1'b0;
        sram_v_web_o  = 1'b1;
        sram_v_addr_o = kv_v_flat;
    end
end

endmodule
