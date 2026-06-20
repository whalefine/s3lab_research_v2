# =============================================================================
# sglatrack_top.sdc -- verilog3 merged top (backbone3 + head3)
#
# Ports: clk reset start sel_block_i data_in data_valid
#        busy done cx_o cy_o w_o h_o
#
# Removed vs head-only / backbone-only SDC:
#   - tok1_preload*   (head-only TB; not in verilog3 netlist)
#   - tok1_readback*  (backbone-only GLS; not in verilog3 netlist)
#
# Genus: read_sdc synthesis/sglatrack_top.sdc  (after read_hdl sglatrack_top.v)
# =============================================================================

set CLK_PERIOD      3.0
set CLK_RISING_EDGE 0
set CLK_FALLING_EDGE [expr {$CLK_PERIOD / 2.0}]
set IO_MAX           [expr {$CLK_PERIOD * 0.4}]
set IO_MIN           0.0
set OUTPUT_DELAY     1.2
set PORT_LOADING     0.05

# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
create_clock -name clk -period $CLK_PERIOD -waveform [list $CLK_RISING_EDGE $CLK_FALLING_EDGE] [get_ports clk]

set_clock_uncertainty -setup 0.05 [get_clocks clk]
set_clock_uncertainty -hold  0.02 [get_clocks clk]

# SRAM/ROM macros now use CLK(clk) (same edge as the core clock); no inverted
# generated clock is needed. The macro CLK pins are driven directly by clk.

# ---------------------------------------------------------------------------
# Reset (async control; not a synchronous data input)
# ---------------------------------------------------------------------------
set_ideal_network [get_ports reset]
set_false_path -from [get_ports reset]

# ---------------------------------------------------------------------------
# Functional I/O (no tok1_preload / tok1_readback)
# ---------------------------------------------------------------------------
set_input_delay  -clock clk -max $IO_MAX [get_ports {start sel_block_i data_in data_valid}]
set_input_delay  -clock clk -min $IO_MIN [get_ports {start sel_block_i data_in data_valid}]

set_output_delay -clock clk -max $OUTPUT_DELAY [get_ports {busy done x_ready cx_o cy_o w_o h_o}]
set_output_delay -clock clk -min 0.0           [get_ports {busy done x_ready cx_o cy_o w_o h_o}]

set_load $PORT_LOADING [get_ports {busy done x_ready cx_o cy_o w_o h_o}]
