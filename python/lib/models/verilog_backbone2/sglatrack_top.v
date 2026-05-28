// =============================================================================
// sglatrack_top.v -- verilog_backbone2 wrapper (backbone-only)
//
// SRAM macros (port mux in child; head2-style sram_*_ceb / sram_*_q wires):
//   Sram_tok1  u_sram_tok1    backbone norm out + S_OUT stream
//   Sram_tok2  u_sram_tok2    inter-block tok_buf
//   Sram_tok3  u_sram_x       transformer_block x_buf
//   Sram_tok4  u_sram_tmp     transformer_block tmp_buf
//   Sram_x     u_sram_snap_x  care_attention norm1 x_snap (not head sram_x)
//   Sram_q/k/v/qkm           care_attention q/k/v/ao/qkm buffers
//
// Input: template/search post-embed (Q8.8) -> backbone_top output stream.
// =============================================================================

module sglatrack_top #(
    parameter EMBED_DIM = 32,
    parameter N_TOKENS  = 320,
    parameter DATA_W    = 16
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    start,
    output wire                    busy,
    output wire                    done,
    input  wire [3:0]              sel_block_i,
    input  wire signed [DATA_W-1:0] data_in,
    input  wire                    data_valid,
    output wire signed [DATA_W-1:0] data_o,
    output wire                    data_o_valid
);

// ---- backbone_top tok2 / tok1 ----
wire              sram_tok2_ceb;
wire              sram_tok2_web;
wire [13:0]       sram_tok2_addr;
wire [DATA_W-1:0] sram_tok2_din;
wire [DATA_W-1:0] sram_tok2_q;

wire              sram_tok1_ceb;
wire              sram_tok1_web;
wire [13:0]       sram_tok1_addr;
wire [DATA_W-1:0] sram_tok1_din;
wire [DATA_W-1:0] sram_tok1_q;

// ---- transformer_block x_buf / tmp_buf ----
wire              sram_x_ceb;
wire              sram_x_web;
wire [13:0]       sram_x_addr;
wire [DATA_W-1:0] sram_x_din;
wire [DATA_W-1:0] sram_x_q;

wire              sram_tmp_ceb;
wire              sram_tmp_web;
wire [13:0]       sram_tmp_addr;
wire [DATA_W-1:0] sram_tmp_din;
wire [DATA_W-1:0] sram_tmp_q;

// ---- care_attention ----
wire              sram_snap_x_ceb;
wire              sram_snap_x_web;
wire [13:0]       sram_snap_x_addr;
wire [DATA_W-1:0] sram_snap_x_din;
wire [DATA_W-1:0] sram_snap_x_q;

wire              sram_q_ceb;
wire              sram_q_web;
wire [13:0]       sram_q_addr;
wire [DATA_W-1:0] sram_q_din;
wire [DATA_W-1:0] sram_q_q;

wire              sram_k_ceb;
wire              sram_k_web;
wire [13:0]       sram_k_addr;
wire [DATA_W-1:0] sram_k_din;
wire [DATA_W-1:0] sram_k_q;

wire              sram_v_ceb;
wire              sram_v_web;
wire [13:0]       sram_v_addr;
wire [DATA_W-1:0] sram_v_din;
wire [DATA_W-1:0] sram_v_q;

wire              sram_qkm_ceb;
wire              sram_qkm_web;
wire [13:0]       sram_qkm_addr;
wire [DATA_W-1:0] sram_qkm_din;
wire [DATA_W-1:0] sram_qkm_q;

// Macro pins: no skew between A and CEB/WEB/D (see verilog_rule.mdc SS8)
wire              sram_tok2_ceb_mac;
wire              sram_tok2_web_mac;
wire [13:0]       sram_tok2_addr_mac;
wire [DATA_W-1:0] sram_tok2_din_mac;

wire              sram_tok1_ceb_mac;
wire              sram_tok1_web_mac;
wire [13:0]       sram_tok1_addr_mac;
wire [DATA_W-1:0] sram_tok1_din_mac;

wire              sram_x_ceb_mac;
wire              sram_x_web_mac;
wire [13:0]       sram_x_addr_mac;
wire [DATA_W-1:0] sram_x_din_mac;

wire              sram_tmp_ceb_mac;
wire              sram_tmp_web_mac;
wire [13:0]       sram_tmp_addr_mac;
wire [DATA_W-1:0] sram_tmp_din_mac;

wire              sram_snap_x_ceb_mac;
wire              sram_snap_x_web_mac;
wire [13:0]       sram_snap_x_addr_mac;
wire [DATA_W-1:0] sram_snap_x_din_mac;

wire              sram_q_ceb_mac;
wire              sram_q_web_mac;
wire [13:0]       sram_q_addr_mac;
wire [DATA_W-1:0] sram_q_din_mac;

wire              sram_k_ceb_mac;
wire              sram_k_web_mac;
wire [13:0]       sram_k_addr_mac;
wire [DATA_W-1:0] sram_k_din_mac;

wire              sram_v_ceb_mac;
wire              sram_v_web_mac;
wire [13:0]       sram_v_addr_mac;
wire [DATA_W-1:0] sram_v_din_mac;

wire              sram_qkm_ceb_mac;
wire              sram_qkm_web_mac;
wire [13:0]       sram_qkm_addr_mac;
wire [DATA_W-1:0] sram_qkm_din_mac;

assign sram_tok2_ceb_mac    = sram_tok2_ceb;
assign sram_tok2_web_mac    = sram_tok2_web;
assign sram_tok2_addr_mac   = sram_tok2_addr;
assign sram_tok2_din_mac    = sram_tok2_din;

assign sram_tok1_ceb_mac    = sram_tok1_ceb;
assign sram_tok1_web_mac    = sram_tok1_web;
assign sram_tok1_addr_mac   = sram_tok1_addr;
assign sram_tok1_din_mac    = sram_tok1_din;

assign sram_x_ceb_mac       = sram_x_ceb;
assign sram_x_web_mac       = sram_x_web;
assign sram_x_addr_mac      = sram_x_addr;
assign sram_x_din_mac       = sram_x_din;

assign sram_tmp_ceb_mac     = sram_tmp_ceb;
assign sram_tmp_web_mac     = sram_tmp_web;
assign sram_tmp_addr_mac    = sram_tmp_addr;
assign sram_tmp_din_mac     = sram_tmp_din;

assign sram_snap_x_ceb_mac  = sram_snap_x_ceb;
assign sram_snap_x_web_mac  = sram_snap_x_web;
assign sram_snap_x_addr_mac = sram_snap_x_addr;
assign sram_snap_x_din_mac  = sram_snap_x_din;

assign sram_q_ceb_mac       = sram_q_ceb;
assign sram_q_web_mac       = sram_q_web;
assign sram_q_addr_mac      = sram_q_addr;
assign sram_q_din_mac       = sram_q_din;

assign sram_k_ceb_mac       = sram_k_ceb;
assign sram_k_web_mac       = sram_k_web;
assign sram_k_addr_mac      = sram_k_addr;
assign sram_k_din_mac       = sram_k_din;

assign sram_v_ceb_mac       = sram_v_ceb;
assign sram_v_web_mac       = sram_v_web;
assign sram_v_addr_mac      = sram_v_addr;
assign sram_v_din_mac       = sram_v_din;

assign sram_qkm_ceb_mac     = sram_qkm_ceb;
assign sram_qkm_web_mac     = sram_qkm_web;
assign sram_qkm_addr_mac    = sram_qkm_addr;
assign sram_qkm_din_mac     = sram_qkm_din;

backbone_top #(
    .EMBED_DIM (EMBED_DIM),
    .N_TOKENS  (N_TOKENS)
) u_backbone (
    .clk            (clk),
    .reset          (reset),
    .start          (start),
    .sel_block_i    (sel_block_i),
    .x_i            (data_in),
    .x_valid        (data_valid),
    .busy           (busy),
    .done           (done),
    .y_o            (data_o),
    .y_valid        (data_o_valid),
    .sram_tok2_ceb_o    (sram_tok2_ceb),
    .sram_tok2_web_o    (sram_tok2_web),
    .sram_tok2_addr_o   (sram_tok2_addr),
    .sram_tok2_din_o    (sram_tok2_din),
    .sram_tok2_q_i      (sram_tok2_q),
    .sram_tok1_ceb_o    (sram_tok1_ceb),
    .sram_tok1_web_o    (sram_tok1_web),
    .sram_tok1_addr_o   (sram_tok1_addr),
    .sram_tok1_din_o    (sram_tok1_din),
    .sram_tok1_q_i      (sram_tok1_q),
    .sram_x_ceb_o   (sram_x_ceb),
    .sram_x_web_o   (sram_x_web),
    .sram_x_addr_o  (sram_x_addr),
    .sram_x_din_o   (sram_x_din),
    .sram_x_q_i     (sram_x_q),
    .sram_tmp_ceb_o (sram_tmp_ceb),
    .sram_tmp_web_o (sram_tmp_web),
    .sram_tmp_addr_o(sram_tmp_addr),
    .sram_tmp_din_o (sram_tmp_din),
    .sram_tmp_q_i   (sram_tmp_q),
    .sram_snap_x_ceb_o  (sram_snap_x_ceb),
    .sram_snap_x_web_o  (sram_snap_x_web),
    .sram_snap_x_addr_o (sram_snap_x_addr),
    .sram_snap_x_din_o  (sram_snap_x_din),
    .sram_snap_x_q_i    (sram_snap_x_q),
    .sram_q_ceb_o       (sram_q_ceb),
    .sram_q_web_o       (sram_q_web),
    .sram_q_addr_o      (sram_q_addr),
    .sram_q_din_o       (sram_q_din),
    .sram_q_q_i         (sram_q_q),
    .sram_k_ceb_o       (sram_k_ceb),
    .sram_k_web_o       (sram_k_web),
    .sram_k_addr_o      (sram_k_addr),
    .sram_k_din_o       (sram_k_din),
    .sram_k_q_i         (sram_k_q),
    .sram_v_ceb_o       (sram_v_ceb),
    .sram_v_web_o       (sram_v_web),
    .sram_v_addr_o      (sram_v_addr),
    .sram_v_din_o       (sram_v_din),
    .sram_v_q_i         (sram_v_q),
    .sram_qkm_ceb_o     (sram_qkm_ceb),
    .sram_qkm_web_o     (sram_qkm_web),
    .sram_qkm_addr_o    (sram_qkm_addr),
    .sram_qkm_din_o     (sram_qkm_din),
    .sram_qkm_q_i       (sram_qkm_q)
);

Sram_tok2 u_sram_tok2 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tok2_ceb_mac),
    .WEB   (sram_tok2_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tok2_addr_mac),
    .D     (sram_tok2_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tok2_q)
);

Sram_tok1 u_sram_tok1 (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tok1_ceb_mac),
    .WEB   (sram_tok1_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tok1_addr_mac),
    .D     (sram_tok1_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tok1_q)
);

Sram_tok3 u_sram_x (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_x_ceb_mac),
    .WEB   (sram_x_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_x_addr_mac),
    .D     (sram_x_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_x_q)
);

Sram_tok4 u_sram_tmp (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_tmp_ceb_mac),
    .WEB   (sram_tmp_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_tmp_addr_mac),
    .D     (sram_tmp_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_tmp_q)
);

Sram_x u_sram_snap_x (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_snap_x_ceb_mac),
    .WEB   (sram_snap_x_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_snap_x_addr_mac),
    .D     (sram_snap_x_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_snap_x_q)
);

Sram_q u_sram_q (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_q_ceb_mac),
    .WEB   (sram_q_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_q_addr_mac),
    .D     (sram_q_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_q_q)
);

Sram_k u_sram_k (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_k_ceb_mac),
    .WEB   (sram_k_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_k_addr_mac),
    .D     (sram_k_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_k_q)
);

Sram_v u_sram_v (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_v_ceb_mac),
    .WEB   (sram_v_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_v_addr_mac),
    .D     (sram_v_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_v_q)
);

Sram_qkm u_sram_qkm (
    .SLP   (1'b0),
    .DSLP  (1'b0),
    .SD    (1'b0),
    .PUDELAY(),
    .CLK   (~clk),
    .CEB   (sram_qkm_ceb_mac),
    .WEB   (sram_qkm_web_mac),
    .BIST  (1'b0),
    .CEBM  (),
    .WEBM  (),
    .A     (sram_qkm_addr_mac),
    .D     (sram_qkm_din_mac),
    .BWEB  (16'b0),
    .AM    (),
    .DM    (),
    .BWEBM (16'b0),
    .RTSEL (2'b01),
    .WTSEL (2'b00),
    .Q     (sram_qkm_q)
);

endmodule
