#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate chip_top.v: an IO-pad ring wrapper around the synthesized
sglatrack_top (verilog3-2), to be placed *before APR*.

Mirrors the reference project's TSMC pad usage:
  - GPIO pad : PDDWUWSWCDGS_H / PDDWUWSWCDGS_V  (configurable in/out)
  - Corner   : PCBRTE_V                          (RTE ring source)

Flow:
  1. synthesis -> sglatrack_top_syn.v (module name kept as sglatrack_top)
  2. python gen_chip_io.py            -> chip_top.v   (top cell = CHIP)
  3. APR uses CHIP as the top, with chip_top.v + sglatrack_top_syn.v + IO/std libs

Pad pin convention (taken from the reference .v in the request):
  INPUT  pad : .C(core_in),  .I(1'b0),     .IE(1'b1), .OEN(1'b1), .PAD(chip_pin),
               .DS0/1/2(1), .DS3(0), .PU(0), .PD(0), .ST(0), .RTE(io_rte)
  OUTPUT pad : .C(),         .I(core_out), .IE(1'b0), .OEN(1'b0), .PAD(chip_pin),
               .DS0/1/2/3(1),         .PU(0), .PD(0), .ST(0), .RTE(io_rte)

Edit PORTS / pad-cell names / drive-strength / side distribution below to
match your actual IO library and floorplan.
"""

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORE_MODULE = "sglatrack_top"   # synthesized core (post-synth keeps this name)
CHIP_MODULE = "CHIP"            # APR top cell
OUT_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chip_top.v")

GPIO_PAD_H = "PDDWUWSWCDGS_H"   # horizontal GPIO pad (top/bottom side)
GPIO_PAD_V = "PDDWUWSWCDGS_V"   # vertical   GPIO pad (left/right side)
CORNER_PAD = "PCBRTE_V"         # corner / RTE source pad

# Core port list: (name, direction, msb)  -- msb=None means 1-bit scalar.
# Keep this in sync with sglatrack_top's port declaration.
PORTS = [
    ("clk",         "in",  None),
    ("reset",       "in",  None),
    ("start",       "in",  None),
    ("sel_block_i", "in",  3),
    ("data_in",     "in",  15),
    ("data_valid",  "in",  None),
    ("busy",        "out", None),
    ("done",        "out", None),
    ("x_ready",     "out", None),
    ("cx_o",        "out", 15),
    ("cy_o",        "out", 15),
    ("w_o",         "out", 15),
    ("h_o",         "out", 15),
]

IO_OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHIP.io")

# Balanced 4-side pad assignment (~23 pads/side) for a near-square die.
# Each 16-bit OUTPUT bus stays whole on one side; only the data_in INPUT bus
# is split. left/right -> _V, top/bottom -> _H.
#   top   (23): cx_o[15:0]            + data_in[15:9]
#   bottom(23): h_o[15:0]             + data_in[8:2]
#   left  (23): cy_o[15:0]            + data_in[1:0] + sel_block_i[3:0] + clk
#   right (22): w_o[15:0]             + reset/start/data_valid + busy/done/x_ready
def pad_side(name, i):
    if name == "cx_o":
        return "top"
    if name == "h_o":
        return "bottom"
    if name == "cy_o":
        return "left"
    if name == "w_o":
        return "right"
    if name == "data_in":
        if i >= 9:
            return "top"
        if i >= 2:
            return "bottom"
        return "left"
    if name in ("sel_block_i", "clk"):
        return "left"
    if name in ("reset", "start", "data_valid", "busy", "done", "x_ready"):
        return "right"
    return "top"


# Cell-shape convention (per IO kit): _V cells are "tall"  -> top/bottom edges;
# _H cells are "wide" -> left/right edges. Picking the correctly-shaped variant
# per side lets each pad sit at the side's default orientation, so CHIP.io does
# NOT need a per-pad orientation= (only the corner cells carry MX/MY).
SIDE_ORIENT = {"top": GPIO_PAD_V, "bottom": GPIO_PAD_V,
               "left": GPIO_PAD_H, "right": GPIO_PAD_H}


def pad_orient(name, i):
    return SIDE_ORIENT[pad_side(name, i)]


# ---- CHIP.io physical placement helpers --------------------------------------
# Power/ground + POC instance names (must also exist in chip_top.v netlist).
# Spread across sides for even power; tune per floorplan.
SIDE_POWER = {
    "top":    ["CORE_PG1", "IO_PG1"],
    "bottom": ["CORE_PG0", "IO_PG0", "POC"],
    "left":   ["CORE_PG3", "IO_PG3"],
    "right":  ["CORE_PG2", "IO_PG2", "IO_PG4"],
}
# Non-corner ring source pad(s) per side.
SIDE_EXTRA = {"top": ["pad_RTE"]}
# Corner cells: (.io section name, inst name, orientation-or-None).
CORNERS = [
    ("topright",    "CORNERTR", None),
    ("topleft",     "CORNERTL", "MX"),
    ("bottomleft",  "CORNERBL", None),
    ("bottomright", "CORNERBR", "MY"),
]
IO_SKIP = 15        # spacing of first pad from corner (tune per die)
IO_ENDGAP = 15      # trailing endspace per side (tune per die)

# Signal pads carry no explicit orientation: the correctly-shaped _V/_H variant
# (see SIDE_ORIENT) already matches each side, so the tool's per-side default
# orientation applies. Only corner cells keep MX/MY (see CORNERS).


def bit_list(name, msb):
    """Expand a port into per-bit (pin_name, core_wire, index) tuples, MSB..0."""
    if msb is None:
        return [(name, "c_%s" % name, None)]
    out = []
    for i in range(msb, -1, -1):
        out.append(("%s[%d]" % (name, i), "c_%s[%d]" % (name, i), i))
    return out


def decl_width(msb):
    return "" if msb is None else "[%d:0] " % msb


def side_signal_insts():
    """Ordered signal-pad instance names per side (PORTS order, MSB..0)."""
    sides = {"top": [], "bottom": [], "left": [], "right": []}
    for name, d, msb in PORTS:
        prefix = "ipad_" if d == "in" else "opad_"
        for pin, cwire, i in bit_list(name, msb):
            inst = prefix + name + ("" if i is None else "_%d" % i)
            sides[pad_side(name, i)].append(inst)
    return sides


def side_full_order(side, sig):
    """Signal pads with power/extra inserted near the middle of the side."""
    extra = list(SIDE_POWER.get(side, [])) + list(SIDE_EXTRA.get(side, []))
    if not extra:
        return list(sig)
    mid = len(sig) // 2
    return list(sig[:mid]) + extra + list(sig[mid:])


def emit_chip_io():
    """Write a balanced CHIP.io. skip/endgap/power spread are tunable hints."""
    sig = side_signal_insts()
    L = []
    a = L.append
    a("# AUTO-GENERATED by gen_chip_io.py (balanced ~23 pads/side).")
    a("# skip/endspace gap and power-pad spread are floorplan hints -- tune freely.")
    a("(globals")
    a("\tversion = 3")
    a("\tio_order = default")
    a(")")
    a("(iopad")
    corner = {c[0]: c for c in CORNERS}

    def emit_corner(secname):
        _sec, inst, orient = corner[secname]
        o = (' orientation=%s' % orient) if orient else ''
        a("\t(%s" % secname)
        a('\t\t(inst name="%s"%s )' % (inst, o))
        a("\t)")

    def emit_side(secname, side):
        order = side_full_order(side, sig[side])
        a("\t(%s" % secname)
        for k, inst in enumerate(order):
            sk = (' skip=%d' % IO_SKIP) if k == 0 else ''
            a('\t\t(inst name="%s"%s )' % (inst, sk))
        a("\t\t(endspace gap=%d )" % IO_ENDGAP)
        a("\t)")

    emit_corner("topright")
    emit_side("top", "top")
    emit_corner("topleft")
    emit_side("left", "left")
    emit_corner("bottomleft")
    emit_side("bottom", "bottom")
    emit_corner("bottomright")
    emit_side("right", "right")
    a(")")

    with open(IO_OUT_FILE, "w") as f:
        f.write("\n".join(L) + "\n")
    return {s: len(side_full_order(s, sig[s])) for s in sig}


def main():
    lines = []
    w = lines.append

    w("// =============================================================================")
    w("// chip_top.v -- IO-pad ring wrapper for %s (verilog3-2)" % CORE_MODULE)
    w("// AUTO-GENERATED by gen_chip_io.py. Do not hand-edit; edit the generator.")
    w("//")
    w("// Top cell for APR = %s. Compile with the synthesized netlist + IO/std libs:" % CHIP_MODULE)
    w("//   vcs chip_top.v sglatrack_top_syn.v -v <io_lib>.v -v <std_lib>.v ...")
    w("//")
    w("// Power pads / extra corner cells (VDD/VSS/VDDIO/VSSIO + 4 corners) are")
    w("// library-specific and intentionally left as a TODO block near the bottom.")
    w("// =============================================================================")
    w("`timescale 1ns/1ps")
    w("")

    # ---- module header (named ports) ----
    port_names = []
    for name, _d, _m in PORTS:
        port_names.append(name)
    w("module %s (" % CHIP_MODULE)
    w("    " + ",\n    ".join(port_names))
    w(");")
    w("")

    # ---- chip-level port directions ----
    w("    // ---- chip-level (pad-facing) ports ----")
    for name, d, msb in PORTS:
        kw = "input " if d == "in" else "output"
        w("    %s %s%s;" % (kw, decl_width(msb), name))
    w("")

    # ---- core-side wires ----
    w("    // ---- core-side nets (CORE <-> pad .C/.I) ----")
    for name, d, msb in PORTS:
        w("    wire %sc_%s;" % (decl_width(msb), name))
    w("")

    # ---- IO ring net ----
    w("    // ---- IO ring routing net ----")
    w("    wire io_rte;")
    w("")

    # ---- core instance ----
    w("    // ---- synthesized core ----")
    w("    %s CORE (" % CORE_MODULE)
    conn = []
    for name, _d, _m in PORTS:
        conn.append("        .%s(c_%s)" % (name, name))
    w(",\n".join(conn))
    w("    );")
    w("")

    # ---- corner / RTE pad ----
    w("    // ---- corner pad: drives the IO ring (RTE) ----")
    w("    %s pad_RTE (.IRTE(1'b0), .RTE(io_rte));" % CORNER_PAD)
    w("")

    # ---- signal pads ----
    pad_idx = 0
    w("    // ---- input pads (IE=1, OEN=1, I=1'b0) ----")
    for name, d, msb in PORTS:
        if d != "in":
            continue
        for pin, cwire, i in bit_list(name, msb):
            orient = pad_orient(name, i)
            inst = "ipad_%s%s" % (name, "" if i is None else "_%d" % i)
            w("    %s %s" % (orient, inst))
            w("    (.C(%s), .I(1'b0), .IE(1'b1), .OEN(1'b1), .PAD(%s),"
              % (cwire, pin))
            w("     .DS0(1'b1), .DS1(1'b1), .DS2(1'b1), .DS3(1'b0),"
              " .PU(1'b0), .PD(1'b0), .ST(1'b0), .RTE(io_rte));")
            pad_idx += 1
    w("")

    w("    // ---- output pads (IE=0, OEN=0, I=core) ----")
    for name, d, msb in PORTS:
        if d != "out":
            continue
        for pin, cwire, i in bit_list(name, msb):
            orient = pad_orient(name, i)
            inst = "opad_%s%s" % (name, "" if i is None else "_%d" % i)
            w("    %s %s" % (orient, inst))
            w("    (.C(), .I(%s), .IE(1'b0), .OEN(1'b0), .PAD(%s),"
              % (cwire, pin))
            w("     .DS0(1'b1), .DS1(1'b1), .DS2(1'b1), .DS3(1'b1),"
              " .PU(1'b0), .PD(1'b0), .ST(1'b0), .RTE(io_rte));")
            pad_idx += 1
    w("")

    # ---- power pad TODO ----
    w("    // -------------------------------------------------------------------------")
    w("    // TODO: add power/ground pads + remaining 3 corner cells per your IO lib,")
    w("    // e.g. PVDD1CDG / PVSS1CDG (core), PVDD2CDG / PVSS2CDG (IO), and corner")
    w("    // cells on the other three corners. Names depend on the actual pad library.")
    w("    // -------------------------------------------------------------------------")
    w("")
    w("endmodule")
    w("")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))

    per_side = emit_chip_io()

    n_in = sum((1 if m is None else m + 1) for _n, d, m in PORTS if d == "in")
    n_out = sum((1 if m is None else m + 1) for _n, d, m in PORTS if d == "out")
    print("wrote %s" % OUT_FILE)
    print("  input pads : %d" % n_in)
    print("  output pads: %d" % n_out)
    print("  signal pads: %d" % (n_in + n_out))
    print("wrote %s" % IO_OUT_FILE)
    print("  per-side total (signal + power/extra):")
    for s in ("top", "bottom", "left", "right"):
        print("    %-6s : %d" % (s, per_side[s]))


if __name__ == "__main__":
    main()
