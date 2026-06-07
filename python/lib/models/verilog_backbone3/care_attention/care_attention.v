// =============================================================================
// care_attention.v -- verilog_backbone3 CARE attention orchestrator
//
// Thin FSM wiring: attn_qkv -> preprocess -> z_recip -> kv -> core -> proj.
// Bit-accurate vs run_backbone_numpy_shared_trunk.py attention_forward() and
// verilog_backbone2/care_attention.v stage order.
//
// SRAM macros (parent provides behavioral / macro models):
//   Sram_q   : q capture (QKV), split/QK_MEAN/ATTN read-write (pre/core)
//   Sram_k   : k capture (QKV), split/K_MEAN/KV read-write (pre/kv)
//   Sram_v   : v capture (QKV), KV read (kv)
//   Sram_qkm : qkm write (pre), zr read-write (zr), zr read (core)
//   Sram_ao  : ao write (core), ao read (proj)
//
// kv_buf[256]: internal in attn_kv / attn_core; parent copies u_kv -> u_core
// in S_KV_COPY (Verilog-2005 has no unpacked-array module ports).
//
// wgt_addr_o (13-bit local):
//   S_QKV  : 0..3071   (parent may add block ROM base)
//   S_PROJ : 3072..4095 (local proj 0..1023 + 3072)
//
// Golden: Activation/backbone_blocks_<b>_after_attn_attn_out_bi.txt (proj out)
// =============================================================================

module care_attention #(
    parameter EMBED_DIM    = 32,
    parameter NUM_HEADS    = 4,
    parameter HEAD_DIM     = 8,
    parameter N_TOKENS     = 320,
    parameter S_Q88        = 152,
    parameter RELU6_MAX    = 1536,
    parameter RCP_N_NUM    = 205,
    parameter RCP_N_SHIFT  = 16,
    parameter KV_Q88_ROUND = 8388608
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    output reg         norm_rd_en,
    output reg [13:0]  norm_rd_flat,
    input  wire signed [15:0] norm_x,

    input  wire signed [15:0] wgt_i,
    input  wire signed [15:0] bias_i,
    output wire [12:0] wgt_addr_o,

    output wire        busy,
    output reg         done,

    output wire signed [15:0] y_o,
    output wire               y_valid,
    output wire [6:0]         y_neu_o,
    output wire [8:0]         px_tok_o,

    output reg         sram_q_ceb_o,
    output reg         sram_q_web_o,
    output reg [13:0]  sram_q_addr_o,
    output reg [15:0]  sram_q_din_o,
    input  wire [15:0] sram_q_q_i,

    output reg         sram_k_ceb_o,
    output reg         sram_k_web_o,
    output reg [13:0]  sram_k_addr_o,
    output reg [15:0]  sram_k_din_o,
    input  wire [15:0] sram_k_q_i,

    output reg         sram_v_ceb_o,
    output reg         sram_v_web_o,
    output reg [13:0]  sram_v_addr_o,
    output reg [15:0]  sram_v_din_o,
    input  wire [15:0] sram_v_q_i,

    output reg         sram_qkm_ceb_o,
    output reg         sram_qkm_web_o,
    output reg [13:0]  sram_qkm_addr_o,
    output reg [15:0]  sram_qkm_din_o,
    input  wire [15:0] sram_qkm_q_i,

    output reg         sram_ao_ceb_o,
    output reg         sram_ao_web_o,
    output reg [13:0]  sram_ao_addr_o,
    output reg [15:0]  sram_ao_din_o,
    input  wire [15:0] sram_ao_q_i
);

localparam KV_ELEMS = NUM_HEADS * HEAD_DIM * HEAD_DIM;

parameter S_IDLE    = 4'd0;
parameter S_QKV     = 4'd1;
parameter S_PRE     = 4'd2;
parameter S_ZR      = 4'd3;
parameter S_KV      = 4'd4;
parameter S_KV_COPY = 4'd5;
parameter S_CORE    = 4'd6;
parameter S_PROJ    = 4'd7;
parameter S_DONE_ST = 4'd8;

reg [3:0] state;
reg [3:0] next_state;

reg qkv_start;
reg pre_start;
reg zr_start;
reg kv_start;
reg core_start;
reg proj_start;

integer kv_ci;

// --- submodule SRAM / control wires ---
wire        qkv_busy;
wire        qkv_done;
wire        qkv_norm_rd_en;
wire [13:0] qkv_norm_rd_flat;
wire [12:0] qkv_wgt_addr_o;
wire        qkv_sram_q_ceb_o;
wire        qkv_sram_q_web_o;
wire [13:0] qkv_sram_q_addr_o;
wire [15:0] qkv_sram_q_din_o;
wire        qkv_sram_k_ceb_o;
wire        qkv_sram_k_web_o;
wire [13:0] qkv_sram_k_addr_o;
wire [15:0] qkv_sram_k_din_o;
wire        qkv_sram_v_ceb_o;
wire        qkv_sram_v_web_o;
wire [13:0] qkv_sram_v_addr_o;
wire [15:0] qkv_sram_v_din_o;

wire        pre_busy;
wire        pre_done;
wire        pre_sram_q_ceb_o;
wire        pre_sram_q_web_o;
wire [13:0] pre_sram_q_addr_o;
wire [15:0] pre_sram_q_din_o;
wire        pre_sram_k_ceb_o;
wire        pre_sram_k_web_o;
wire [13:0] pre_sram_k_addr_o;
wire [15:0] pre_sram_k_din_o;
wire        pre_sram_qkm_ceb_o;
wire        pre_sram_qkm_web_o;
wire [13:0] pre_sram_qkm_addr_o;
wire [15:0] pre_sram_qkm_din_o;

wire        zr_busy;
wire        zr_done;
wire        zr_sram_qkm_ceb_o;
wire        zr_sram_qkm_web_o;
wire [13:0] zr_sram_qkm_addr_o;
wire [15:0] zr_sram_qkm_din_o;

wire        kv_busy;
wire        kv_done;
wire        kv_sram_k_ceb_o;
wire        kv_sram_k_web_o;
wire [13:0] kv_sram_k_addr_o;
wire        kv_sram_v_ceb_o;
wire        kv_sram_v_web_o;
wire [13:0] kv_sram_v_addr_o;

wire        core_busy;
wire        core_done;
wire        core_sram_q_ceb_o;
wire        core_sram_q_web_o;
wire [13:0] core_sram_q_addr_o;
wire        core_sram_qkm_ceb_o;
wire        core_sram_qkm_web_o;
wire [13:0] core_sram_qkm_addr_o;
wire        core_sram_ao_ceb_o;
wire        core_sram_ao_web_o;
wire [13:0] core_sram_ao_addr_o;
wire [15:0] core_sram_ao_din_o;

wire        proj_busy;
wire        proj_done;
wire [12:0] proj_wgt_addr_o;
wire signed [15:0] proj_y_o;
wire               proj_y_valid;
wire [6:0]         proj_y_neu_o;
wire [8:0]         proj_px_tok_o;
wire        proj_sram_ao_ceb_o;
wire        proj_sram_ao_web_o;
wire [13:0] proj_sram_ao_addr_o;

assign busy       = (state != S_IDLE);
assign wgt_addr_o = (state == S_PROJ) ? (13'd3072 + proj_wgt_addr_o) : qkv_wgt_addr_o;
assign y_o        = proj_y_o;
assign y_valid    = proj_y_valid;
assign y_neu_o    = proj_y_neu_o;
assign px_tok_o   = proj_px_tok_o;

attn_qkv #(
    .EMBED_DIM (EMBED_DIM),
    .NUM_HEADS (NUM_HEADS),
    .HEAD_DIM  (HEAD_DIM),
    .N_TOKENS  (N_TOKENS),
    .QKV_OUT   (96)
) u_qkv (
    .clk            (clk),
    .reset          (reset),
    .start          (qkv_start),
    .norm_rd_en     (qkv_norm_rd_en),
    .norm_rd_flat   (qkv_norm_rd_flat),
    .norm_x         (norm_x),
    .wgt_i          (wgt_i),
    .bias_i         (bias_i),
    .wgt_addr_o     (qkv_wgt_addr_o),
    .busy           (qkv_busy),
    .done           (qkv_done),
    .sram_q_ceb_o   (qkv_sram_q_ceb_o),
    .sram_q_web_o   (qkv_sram_q_web_o),
    .sram_q_addr_o  (qkv_sram_q_addr_o),
    .sram_q_din_o   (qkv_sram_q_din_o),
    .sram_k_ceb_o   (qkv_sram_k_ceb_o),
    .sram_k_web_o   (qkv_sram_k_web_o),
    .sram_k_addr_o  (qkv_sram_k_addr_o),
    .sram_k_din_o   (qkv_sram_k_din_o),
    .sram_v_ceb_o   (qkv_sram_v_ceb_o),
    .sram_v_web_o   (qkv_sram_v_web_o),
    .sram_v_addr_o  (qkv_sram_v_addr_o),
    .sram_v_din_o   (qkv_sram_v_din_o)
);

attn_preprocess #(
    .NUM_HEADS     (NUM_HEADS),
    .HEAD_DIM      (HEAD_DIM),
    .N_TOKENS      (N_TOKENS),
    .S_Q88         (S_Q88),
    .RELU6_MAX     (RELU6_MAX),
    .RCP_N_NUM     (RCP_N_NUM),
    .RCP_N_SHIFT   (RCP_N_SHIFT)
) u_pre (
    .clk             (clk),
    .reset           (reset),
    .start           (pre_start),
    .busy            (pre_busy),
    .done            (pre_done),
    .sram_q_ceb_o    (pre_sram_q_ceb_o),
    .sram_q_web_o    (pre_sram_q_web_o),
    .sram_q_addr_o   (pre_sram_q_addr_o),
    .sram_q_din_o    (pre_sram_q_din_o),
    .sram_q_q_i      (sram_q_q_i),
    .sram_k_ceb_o    (pre_sram_k_ceb_o),
    .sram_k_web_o    (pre_sram_k_web_o),
    .sram_k_addr_o   (pre_sram_k_addr_o),
    .sram_k_din_o    (pre_sram_k_din_o),
    .sram_k_q_i      (sram_k_q_i),
    .sram_qkm_ceb_o  (pre_sram_qkm_ceb_o),
    .sram_qkm_web_o  (pre_sram_qkm_web_o),
    .sram_qkm_addr_o (pre_sram_qkm_addr_o),
    .sram_qkm_din_o  (pre_sram_qkm_din_o)
);

attn_z_recip #(
    .NUM_HEADS (NUM_HEADS),
    .N_TOKENS  (N_TOKENS)
) u_zr (
    .clk             (clk),
    .reset           (reset),
    .start           (zr_start),
    .busy            (zr_busy),
    .done            (zr_done),
    .sram_qkm_ceb_o  (zr_sram_qkm_ceb_o),
    .sram_qkm_web_o  (zr_sram_qkm_web_o),
    .sram_qkm_addr_o (zr_sram_qkm_addr_o),
    .sram_qkm_din_o  (zr_sram_qkm_din_o),
    .sram_qkm_q_i    (sram_qkm_q_i)
);

attn_kv #(
    .NUM_HEADS      (NUM_HEADS),
    .HEAD_DIM       (HEAD_DIM),
    .N_TOKENS       (N_TOKENS),
    .RCP_N_NUM      (RCP_N_NUM),
    .RCP_N_SHIFT    (RCP_N_SHIFT),
    .KV_Q88_ROUND   (KV_Q88_ROUND)
) u_kv (
    .clk            (clk),
    .reset          (reset),
    .start          (kv_start),
    .busy           (kv_busy),
    .done           (kv_done),
    .sram_k_ceb_o   (kv_sram_k_ceb_o),
    .sram_k_web_o   (kv_sram_k_web_o),
    .sram_k_addr_o  (kv_sram_k_addr_o),
    .sram_k_q_i     (sram_k_q_i),
    .sram_v_ceb_o   (kv_sram_v_ceb_o),
    .sram_v_web_o   (kv_sram_v_web_o),
    .sram_v_addr_o  (kv_sram_v_addr_o),
    .sram_v_q_i     (sram_v_q_i)
);

attn_core #(
    .EMBED_DIM (EMBED_DIM),
    .NUM_HEADS (NUM_HEADS),
    .HEAD_DIM  (HEAD_DIM),
    .N_TOKENS  (N_TOKENS)
) u_core (
    .clk             (clk),
    .reset           (reset),
    .start           (core_start),
    .busy            (core_busy),
    .done            (core_done),
    .sram_q_ceb_o    (core_sram_q_ceb_o),
    .sram_q_web_o    (core_sram_q_web_o),
    .sram_q_addr_o   (core_sram_q_addr_o),
    .sram_q_q_i      (sram_q_q_i),
    .sram_qkm_ceb_o  (core_sram_qkm_ceb_o),
    .sram_qkm_web_o  (core_sram_qkm_web_o),
    .sram_qkm_addr_o (core_sram_qkm_addr_o),
    .sram_qkm_q_i    (sram_qkm_q_i),
    .sram_ao_ceb_o   (core_sram_ao_ceb_o),
    .sram_ao_web_o   (core_sram_ao_web_o),
    .sram_ao_addr_o  (core_sram_ao_addr_o),
    .sram_ao_din_o   (core_sram_ao_din_o)
);

attn_proj #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_proj (
    .clk             (clk),
    .reset           (reset),
    .start           (proj_start),
    .wgt_i           (wgt_i),
    .bias_i          (bias_i),
    .wgt_addr_o      (proj_wgt_addr_o),
    .busy            (proj_busy),
    .done            (proj_done),
    .y_o             (proj_y_o),
    .y_valid         (proj_y_valid),
    .y_neu_o         (proj_y_neu_o),
    .px_tok_o        (proj_px_tok_o),
    .sram_ao_ceb_o   (proj_sram_ao_ceb_o),
    .sram_ao_web_o   (proj_sram_ao_web_o),
    .sram_ao_addr_o  (proj_sram_ao_addr_o),
    .sram_ao_q_i     (sram_ao_q_i)
);

// FSM segment 1: state register
always @(posedge clk) begin
    if (reset)
        state <= S_IDLE;
    else
        state <= next_state;
end

// FSM segment 2: next-state
always @(*) begin
    case (state)
        S_IDLE:    next_state = start ? S_QKV : S_IDLE;
        S_QKV:     next_state = qkv_done ? S_PRE : S_QKV;
        S_PRE:     next_state = pre_done ? S_ZR : S_PRE;
        S_ZR:      next_state = zr_done ? S_KV : S_ZR;
        S_KV:      next_state = kv_done ? S_KV_COPY : S_KV;
        S_KV_COPY: next_state = S_CORE;
        S_CORE:    next_state = core_done ? S_PROJ : S_CORE;
        S_PROJ:    next_state = proj_done ? S_DONE_ST : S_PROJ;
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

// Copy kv_buf u_kv -> u_core in S_KV_COPY (integration only; not synthesizable as-is)
`ifndef SYNTHESIS
always @(posedge clk) begin
    if (state == S_KV_COPY) begin
        for (kv_ci = 0; kv_ci < KV_ELEMS; kv_ci = kv_ci + 1)
            u_core.kv_buf[kv_ci] <= u_kv.kv_buf[kv_ci];
    end
end
`endif

// One-cycle submodule start on stage entry
always @(posedge clk) begin
    if (reset) begin
        qkv_start  <= 1'b0;
        pre_start  <= 1'b0;
        zr_start   <= 1'b0;
        kv_start   <= 1'b0;
        core_start <= 1'b0;
        proj_start <= 1'b0;
    end else begin
        qkv_start  <= 1'b0;
        pre_start  <= 1'b0;
        zr_start   <= 1'b0;
        kv_start   <= 1'b0;
        core_start <= 1'b0;
        proj_start <= 1'b0;

        if (state != next_state) begin
            case (next_state)
                S_QKV:  qkv_start  <= 1'b1;
                S_PRE:  pre_start  <= 1'b1;
                S_ZR:   zr_start   <= 1'b1;
                S_KV:   kv_start   <= 1'b1;
                S_CORE: core_start <= 1'b1;
                S_PROJ: proj_start <= 1'b1;
                default: ;
            endcase
        end
    end
end

// norm1 read mux (QKV only)
always @(*) begin
    if (state == S_QKV) begin
        norm_rd_en   = qkv_norm_rd_en;
        norm_rd_flat = qkv_norm_rd_flat;
    end else begin
        norm_rd_en   = 1'b0;
        norm_rd_flat = 14'd0;
    end
end

// SRAM mux by active stage
always @(*) begin
    sram_q_ceb_o    = 1'b1;
    sram_q_web_o    = 1'b1;
    sram_q_addr_o   = 14'd0;
    sram_q_din_o    = 16'd0;
    sram_k_ceb_o    = 1'b1;
    sram_k_web_o    = 1'b1;
    sram_k_addr_o   = 14'd0;
    sram_k_din_o    = 16'd0;
    sram_v_ceb_o    = 1'b1;
    sram_v_web_o    = 1'b1;
    sram_v_addr_o   = 14'd0;
    sram_v_din_o    = 16'd0;
    sram_qkm_ceb_o  = 1'b1;
    sram_qkm_web_o  = 1'b1;
    sram_qkm_addr_o = 14'd0;
    sram_qkm_din_o  = 16'd0;
    sram_ao_ceb_o   = 1'b1;
    sram_ao_web_o   = 1'b1;
    sram_ao_addr_o  = 14'd0;
    sram_ao_din_o   = 16'd0;

    case (state)
        S_QKV: begin
            sram_q_ceb_o  = qkv_sram_q_ceb_o;
            sram_q_web_o  = qkv_sram_q_web_o;
            sram_q_addr_o = qkv_sram_q_addr_o;
            sram_q_din_o  = qkv_sram_q_din_o;
            sram_k_ceb_o  = qkv_sram_k_ceb_o;
            sram_k_web_o  = qkv_sram_k_web_o;
            sram_k_addr_o = qkv_sram_k_addr_o;
            sram_k_din_o  = qkv_sram_k_din_o;
            sram_v_ceb_o  = qkv_sram_v_ceb_o;
            sram_v_web_o  = qkv_sram_v_web_o;
            sram_v_addr_o = qkv_sram_v_addr_o;
            sram_v_din_o  = qkv_sram_v_din_o;
        end

        S_PRE: begin
            sram_q_ceb_o    = pre_sram_q_ceb_o;
            sram_q_web_o    = pre_sram_q_web_o;
            sram_q_addr_o   = pre_sram_q_addr_o;
            sram_q_din_o    = pre_sram_q_din_o;
            sram_k_ceb_o    = pre_sram_k_ceb_o;
            sram_k_web_o    = pre_sram_k_web_o;
            sram_k_addr_o   = pre_sram_k_addr_o;
            sram_k_din_o    = pre_sram_k_din_o;
            sram_qkm_ceb_o  = pre_sram_qkm_ceb_o;
            sram_qkm_web_o  = pre_sram_qkm_web_o;
            sram_qkm_addr_o = pre_sram_qkm_addr_o;
            sram_qkm_din_o  = pre_sram_qkm_din_o;
        end

        S_ZR: begin
            sram_qkm_ceb_o  = zr_sram_qkm_ceb_o;
            sram_qkm_web_o  = zr_sram_qkm_web_o;
            sram_qkm_addr_o = zr_sram_qkm_addr_o;
            sram_qkm_din_o  = zr_sram_qkm_din_o;
        end

        S_KV: begin
            sram_k_ceb_o  = kv_sram_k_ceb_o;
            sram_k_web_o  = kv_sram_k_web_o;
            sram_k_addr_o = kv_sram_k_addr_o;
            sram_v_ceb_o  = kv_sram_v_ceb_o;
            sram_v_web_o  = kv_sram_v_web_o;
            sram_v_addr_o = kv_sram_v_addr_o;
        end

        S_CORE: begin
            sram_q_ceb_o    = core_sram_q_ceb_o;
            sram_q_web_o    = core_sram_q_web_o;
            sram_q_addr_o   = core_sram_q_addr_o;
            sram_qkm_ceb_o  = core_sram_qkm_ceb_o;
            sram_qkm_web_o  = core_sram_qkm_web_o;
            sram_qkm_addr_o = core_sram_qkm_addr_o;
            sram_ao_ceb_o   = core_sram_ao_ceb_o;
            sram_ao_web_o   = core_sram_ao_web_o;
            sram_ao_addr_o  = core_sram_ao_addr_o;
            sram_ao_din_o   = core_sram_ao_din_o;
        end

        S_PROJ: begin
            sram_ao_ceb_o  = proj_sram_ao_ceb_o;
            sram_ao_web_o  = proj_sram_ao_web_o;
            sram_ao_addr_o = proj_sram_ao_addr_o;
        end

        default: ;
    endcase
end

endmodule
