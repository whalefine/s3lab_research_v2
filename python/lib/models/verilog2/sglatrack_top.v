// =============================================================================
// sglatrack_top.v -- verilog2 merged top (backbone + head, 6 shared SRAMs)
//
// Plan B: backbone skips S_OUT; head S_FILL reads Sram_tok1 directly.
// Global MUX (phase_sel): 0=backbone, 1=head. Registered, 1-cycle guard.
//
// SRAM macro sharing (6 macros total):
//   Physical     | Backbone use                 | Head use
//   -------------|------------------------------|---------------------------
//   Sram_tok1    | inter-block tok + bb norm    | S_FILL read / wgt_buf
//   Sram_tok2    | transformer x_buf            | sh2 (conv2 out / tail in)
//   Sram_q       | care attn q + tmp-on-q       | sh1_lo (conv1 out lo)
//   Sram_k       | care attn k                  | sh1_hi (conv1 out hi)
//   Sram_v       | care attn v + ao             | x_buf (head input)
//   Sram_qkm     | care attn qkm + zr           | cal_bbox size/off
//
// Top FSM: S_IDLE -> S_BACKBONE -> S_GUARD -> S_HEAD -> S_DONE
//
// Compile (from project root):
//   vcs python/lib/models/verilog2/sglatrack_top.v \
//       python/lib/models/verilog2/TEST.v \
//       memory/Sram_*.v memory/rom_*.v \
//       +lint=TFIPC-L +define+TSMC_CM_NO_WARNING
// =============================================================================

`include "inv_sqrt_lut_seed.v"
`include "inv_sqrt_nr.v"
`include "recip_lut_seed.v"
`include "recip_nr.v"
`include "linear.v"
`include "linear_wide.v"
`include "residual.v"
`include "layer_norm.v"
`include "care_attention.v"
`include "mlp.v"
`include "transformer_block.v"
`include "backbone_top.v"
`include "sigmoid_lut.v"
`include "cal_bbox.v"
`include "conv.v"
`include "tail.v"
`include "head_top.v"

module sglatrack_top #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320,
    parameter DATA_W    = 16,
    parameter FEAT_H    = 16,
    parameter FEAT_W    = 16
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    start,
    input  wire [3:0]              sel_block_i,
    input  wire signed [DATA_W-1:0] data_in,
    input  wire                    data_valid,
    output wire                    busy,
    output wire                    done,
    output wire [DATA_W-1:0]       cx_o,
    output wire [DATA_W-1:0]       cy_o,
    output wire [DATA_W-1:0]       w_o,
    output wire [DATA_W-1:0]       h_o
);

// =========================================================================
// Top-level FSM
// =========================================================================
parameter TS_IDLE     = 3'd0;
parameter TS_BACKBONE = 3'd1;
parameter TS_GUARD    = 3'd2;
parameter TS_HEAD     = 3'd3;
parameter TS_DONE     = 3'd4;

reg [2:0] top_state, top_next;

reg        phase_sel;      // 0=backbone, 1=head (registered)
reg        bb_start_r;
reg        hd_start_r;
wire       bb_done;
wire       bb_busy;
wire       hd_done;
wire       hd_busy;

// FSM state register
always @(posedge clk) begin
    if (reset) top_state <= TS_IDLE;
    else       top_state <= top_next;
end

// FSM next-state
always @(*) begin
    top_next = top_state;
    case (top_state)
        TS_IDLE:     top_next = start ? TS_BACKBONE : TS_IDLE;
        TS_BACKBONE: top_next = bb_done ? TS_GUARD : TS_BACKBONE;
        TS_GUARD:    top_next = TS_HEAD;
        TS_HEAD:     top_next = hd_done ? TS_DONE : TS_HEAD;
        TS_DONE:     top_next = TS_IDLE;
        default:     top_next = TS_IDLE;
    endcase
end

// FSM datapath: phase_sel, sub-module start pulses
always @(posedge clk) begin
    bb_start_r <= 1'b0;
    hd_start_r <= 1'b0;

    if (reset) begin
        phase_sel  <= 1'b0;
    end else begin
        case (top_state)
            TS_IDLE: begin
                phase_sel <= 1'b0;
                if (start)
                    bb_start_r <= 1'b1;
            end
            TS_GUARD: begin
                phase_sel  <= 1'b1;
                hd_start_r <= 1'b1;
            end
            TS_DONE: begin
                phase_sel <= 1'b0;
            end
            default: ;
        endcase
    end
end

assign busy = (top_state != TS_IDLE);
assign done = (top_state == TS_DONE);

// =========================================================================
// Backbone port wires
// =========================================================================
wire              bb_tok1_ceb,  bb_tok1_web;
wire [13:0]       bb_tok1_addr;
wire [DATA_W-1:0] bb_tok1_din;

wire              bb_tok2_ceb,  bb_tok2_web;
wire [13:0]       bb_tok2_addr;
wire [DATA_W-1:0] bb_tok2_din;

wire              bb_q_ceb,  bb_q_web;
wire [13:0]       bb_q_addr;
wire [DATA_W-1:0] bb_q_din;

wire              bb_k_ceb,  bb_k_web;
wire [13:0]       bb_k_addr;
wire [DATA_W-1:0] bb_k_din;

wire              bb_v_ceb,  bb_v_web;
wire [13:0]       bb_v_addr;
wire [DATA_W-1:0] bb_v_din;

wire              bb_qkm_ceb,  bb_qkm_web;
wire [13:0]       bb_qkm_addr;
wire [DATA_W-1:0] bb_qkm_din;

// =========================================================================
// Head port wires
// =========================================================================
wire              hd_x_ceb,  hd_x_web;
wire [13:0]       hd_x_addr;
wire [DATA_W-1:0] hd_x_din;

wire              hd_sh1lo_ceb,  hd_sh1lo_web;
wire [13:0]       hd_sh1lo_addr;
wire [DATA_W-1:0] hd_sh1lo_din;

wire              hd_sh1hi_ceb,  hd_sh1hi_web;
wire [13:0]       hd_sh1hi_addr;
wire [DATA_W-1:0] hd_sh1hi_din;

wire              hd_tok2_ceb,  hd_tok2_web;
wire [13:0]       hd_tok2_addr;
wire [DATA_W-1:0] hd_tok2_din;

wire              hd_tok1_ceb,  hd_tok1_web;
wire [13:0]       hd_tok1_addr;
wire [DATA_W-1:0] hd_tok1_din;

wire              hd_bbox_ceb,  hd_bbox_web;
wire [10:0]       hd_bbox_addr;
wire [DATA_W-1:0] hd_bbox_din;

// =========================================================================
// SRAM macro port wires (after MUX)
// =========================================================================
wire              m_tok1_ceb, m_tok1_web;
wire [13:0]       m_tok1_addr;
wire [DATA_W-1:0] m_tok1_din;
wire [DATA_W-1:0] m_tok1_q;

wire              m_tok2_ceb, m_tok2_web;
wire [13:0]       m_tok2_addr;
wire [DATA_W-1:0] m_tok2_din;
wire [DATA_W-1:0] m_tok2_q;

wire              m_q_ceb, m_q_web;
wire [13:0]       m_q_addr;
wire [DATA_W-1:0] m_q_din;
wire [DATA_W-1:0] m_q_q;

wire              m_k_ceb, m_k_web;
wire [13:0]       m_k_addr;
wire [DATA_W-1:0] m_k_din;
wire [DATA_W-1:0] m_k_q;

wire              m_v_ceb, m_v_web;
wire [13:0]       m_v_addr;
wire [DATA_W-1:0] m_v_din;
wire [DATA_W-1:0] m_v_q;

wire              m_qkm_ceb, m_qkm_web;
wire [13:0]       m_qkm_addr;
wire [DATA_W-1:0] m_qkm_din;
wire [DATA_W-1:0] m_qkm_q;

// =========================================================================
// Global MUX: phase_sel = 0 -> backbone, 1 -> head
//   Backbone mapping:  tok1=inter-block/norm, tok2=x_buf, q=q, k=k, v=v, qkm=qkm
//   Head mapping:      tok1=S_FILL/wgt, tok2=sh2, q=sh1_lo, k=sh1_hi, v=x_buf, qkm=bbox
// =========================================================================

// Sram_tok1: backbone inter-block/norm / head wgt_buf (+ S_FILL read)
assign m_tok1_ceb  = phase_sel ? hd_tok1_ceb      : bb_tok1_ceb;
assign m_tok1_web  = phase_sel ? hd_tok1_web       : bb_tok1_web;
assign m_tok1_addr = phase_sel ? hd_tok1_addr      : bb_tok1_addr;
assign m_tok1_din  = phase_sel ? hd_tok1_din       : bb_tok1_din;

// Sram_tok2: backbone x_buf / head sh2
assign m_tok2_ceb  = phase_sel ? hd_tok2_ceb       : bb_tok2_ceb;
assign m_tok2_web  = phase_sel ? hd_tok2_web        : bb_tok2_web;
assign m_tok2_addr = phase_sel ? hd_tok2_addr       : bb_tok2_addr;
assign m_tok2_din  = phase_sel ? hd_tok2_din        : bb_tok2_din;

// Sram_q: backbone q / head sh1_lo
assign m_q_ceb     = phase_sel ? hd_sh1lo_ceb     : bb_q_ceb;
assign m_q_web     = phase_sel ? hd_sh1lo_web      : bb_q_web;
assign m_q_addr    = phase_sel ? hd_sh1lo_addr     : bb_q_addr;
assign m_q_din     = phase_sel ? hd_sh1lo_din      : bb_q_din;

// Sram_k: backbone k / head sh1_hi
assign m_k_ceb     = phase_sel ? hd_sh1hi_ceb     : bb_k_ceb;
assign m_k_web     = phase_sel ? hd_sh1hi_web      : bb_k_web;
assign m_k_addr    = phase_sel ? hd_sh1hi_addr     : bb_k_addr;
assign m_k_din     = phase_sel ? hd_sh1hi_din      : bb_k_din;

// Sram_v: backbone v / head x_buf
assign m_v_ceb     = phase_sel ? hd_x_ceb         : bb_v_ceb;
assign m_v_web     = phase_sel ? hd_x_web          : bb_v_web;
assign m_v_addr    = phase_sel ? hd_x_addr         : bb_v_addr;
assign m_v_din     = phase_sel ? hd_x_din          : bb_v_din;

// Sram_qkm: backbone qkm / head bbox (head 11-bit addr, pad to 14-bit)
assign m_qkm_ceb   = phase_sel ? hd_bbox_ceb      : bb_qkm_ceb;
assign m_qkm_web   = phase_sel ? hd_bbox_web       : bb_qkm_web;
assign m_qkm_addr  = phase_sel ? {3'b0, hd_bbox_addr} : bb_qkm_addr;
assign m_qkm_din   = phase_sel ? hd_bbox_din       : bb_qkm_din;

// =========================================================================
// Backbone instance
// =========================================================================
backbone_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_backbone (
    .clk            (clk),
    .reset          (reset),
    .start          (bb_start_r),
    .sel_block_i    (sel_block_i),
    .x_i            (data_in),
    .x_valid        (data_valid),
    .busy           (bb_busy),
    .x_ready        (),
    .done           (bb_done),
    .y_o            (),
    .y_valid        (),
    .sram_tok1_ceb_o    (bb_tok1_ceb),
    .sram_tok1_web_o    (bb_tok1_web),
    .sram_tok1_addr_o   (bb_tok1_addr),
    .sram_tok1_din_o    (bb_tok1_din),
    .sram_tok1_q_i      (m_tok1_q),
    .sram_tok2_ceb_o   (bb_tok2_ceb),
    .sram_tok2_web_o   (bb_tok2_web),
    .sram_tok2_addr_o  (bb_tok2_addr),
    .sram_tok2_din_o   (bb_tok2_din),
    .sram_tok2_q_i     (m_tok2_q),
    .sram_q_ceb_o       (bb_q_ceb),
    .sram_q_web_o       (bb_q_web),
    .sram_q_addr_o      (bb_q_addr),
    .sram_q_din_o       (bb_q_din),
    .sram_q_q_i         (m_q_q),
    .sram_k_ceb_o       (bb_k_ceb),
    .sram_k_web_o       (bb_k_web),
    .sram_k_addr_o      (bb_k_addr),
    .sram_k_din_o       (bb_k_din),
    .sram_k_q_i         (m_k_q),
    .sram_v_ceb_o       (bb_v_ceb),
    .sram_v_web_o       (bb_v_web),
    .sram_v_addr_o      (bb_v_addr),
    .sram_v_din_o       (bb_v_din),
    .sram_v_q_i         (m_v_q),
    .sram_qkm_ceb_o     (bb_qkm_ceb),
    .sram_qkm_web_o     (bb_qkm_web),
    .sram_qkm_addr_o    (bb_qkm_addr),
    .sram_qkm_din_o     (bb_qkm_din),
    .sram_qkm_q_i       (m_qkm_q)
);

// =========================================================================
// Head instance (Plan B: no a_i/a_valid; S_FILL reads Sram_tok1 via sram_tok1 port)
// =========================================================================
head_top #(
    .IN_CH    (EMBED_DIM),
    .C_SH1    (96),
    .C_SH2    (48),
    .FEAT_H   (FEAT_H),
    .FEAT_W   (FEAT_W),
    .N_TOKENS (N_TOKENS),
    .LENS_Z   (64),
    .DATA_W   (DATA_W)
) u_head (
    .clk     (clk),
    .reset   (reset),
    .start   (hd_start_r),
    .busy    (hd_busy),
    .done    (hd_done),
    .cx_o    (cx_o),
    .cy_o    (cy_o),
    .w_o     (w_o),
    .h_o     (h_o),
    .sram_x_ceb_o      (hd_x_ceb),
    .sram_x_web_o      (hd_x_web),
    .sram_x_addr_o     (hd_x_addr),
    .sram_x_din_o      (hd_x_din),
    .sram_x_q_i        (m_v_q),
    .sram_sh1_lo_ceb_o (hd_sh1lo_ceb),
    .sram_sh1_lo_web_o (hd_sh1lo_web),
    .sram_sh1_lo_addr_o(hd_sh1lo_addr),
    .sram_sh1_lo_din_o (hd_sh1lo_din),
    .sram_sh1_lo_q_i   (m_q_q),
    .sram_sh1_hi_ceb_o (hd_sh1hi_ceb),
    .sram_sh1_hi_web_o (hd_sh1hi_web),
    .sram_sh1_hi_addr_o(hd_sh1hi_addr),
    .sram_sh1_hi_din_o (hd_sh1hi_din),
    .sram_sh1_hi_q_i   (m_k_q),
    .sram_tok2_ceb_o    (hd_tok2_ceb),
    .sram_tok2_web_o    (hd_tok2_web),
    .sram_tok2_addr_o   (hd_tok2_addr),
    .sram_tok2_din_o    (hd_tok2_din),
    .sram_tok2_q_i      (m_tok2_q),
    .sram_tok1_ceb_o    (hd_tok1_ceb),
    .sram_tok1_web_o    (hd_tok1_web),
    .sram_tok1_addr_o   (hd_tok1_addr),
    .sram_tok1_din_o    (hd_tok1_din),
    .sram_tok1_q_i      (m_tok1_q),
    .sram_bbox_ceb_o   (hd_bbox_ceb),
    .sram_bbox_web_o   (hd_bbox_web),
    .sram_bbox_addr_o  (hd_bbox_addr),
    .sram_bbox_din_o   (hd_bbox_din),
    .sram_bbox_q_i     (m_qkm_q)
);

// =========================================================================
// 6 shared SRAM macro instances (CLK = ~clk)
// =========================================================================

Sram_tok1 u_sram_tok1 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (m_tok1_ceb),
    .WEB   (m_tok1_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (m_tok1_addr),
    .D     (m_tok1_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (m_tok1_q)
);

Sram_tok2 u_sram_tok2 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (m_tok2_ceb),
    .WEB   (m_tok2_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (m_tok2_addr),
    .D     (m_tok2_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (m_tok2_q)
);

Sram_q u_sram_q (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (m_q_ceb),
    .WEB   (m_q_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (m_q_addr),
    .D     (m_q_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (m_q_q)
);

Sram_k u_sram_k (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (m_k_ceb),
    .WEB   (m_k_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (m_k_addr),
    .D     (m_k_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (m_k_q)
);

Sram_v u_sram_v (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (m_v_ceb),
    .WEB   (m_v_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (m_v_addr),
    .D     (m_v_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (m_v_q)
);

Sram_qkm u_sram_qkm (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (m_qkm_ceb),
    .WEB   (m_qkm_web),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (m_qkm_addr[10:0]),
    .D     (m_qkm_din),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (m_qkm_q)
);

endmodule
