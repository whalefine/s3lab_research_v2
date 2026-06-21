#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect LEF / GDSII files from each rom* folder and create symbolic links in lef/ and gds/.

Expected directory layout (under --root):

    <root>/
      rom_backbone_blocks_0_3_attn_qkv_weight/
        LEF/    *.lef
        GDSII/  *.gds
      rom_box_head_tail_ctr_offset_size_weight/
        LEF/    *.lef
        GDSII/  *.gds
      ...

After running, this creates under --root:

    <root>/lef/   <- symbolic links to all rom*/LEF/*.lef (pointing to source absolute path)
    <root>/gds/   <- symbolic links to all rom*/GDSII/*.gds (pointing to source absolute path)

Default behavior:
  - Only process folders whose name starts with --prefix (default "rom").
  - Subfolder name matching is case-insensitive (LEF/lef, GDSII/gdsii).
  - File extension matching is case-insensitive (.lef/.LEF, .gds/.GDS).
  - Create symbolic links; the link target is always the source file's absolute path.
  - If a link/file with the same name already exists at the destination, skip it
    (use --overwrite to replace).

Usage examples:
    python3 collect_lef_gds.py                 # create symlinks in the current directory
    python3 collect_lef_gds.py --root /path/to/Netlist3
    python3 collect_lef_gds.py --overwrite     # overwrite existing link/file with the same name
"""

import argparse
import os
import sys


def find_subdir(parent, target_name):
    """Find a subfolder under parent whose name equals target_name (case-insensitive)."""
    target_lower = target_name.lower()
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    for name in entries:
        full = os.path.join(parent, name)
        if os.path.isdir(full) and name.lower() == target_lower:
            return full
    return None


def list_files_with_ext(folder, ext):
    """List full paths of files in folder with extension ext (case-insensitive, e.g. ".lef")."""
    ext_lower = ext.lower()
    result = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() == ext_lower:
            result.append(full)
    return result


def transfer(src, dst_dir, overwrite=False):
    """Create a symbolic link in dst_dir pointing to the absolute path of src.

    Returns ("linked"/"skipped", dst_path).
    """
    src_abs = os.path.abspath(src)
    dst = os.path.join(dst_dir, os.path.basename(src))
    # os.path.exists returns False for a symlink pointing to a missing target, so use lexists
    if os.path.lexists(dst):
        if not overwrite:
            return ("skipped", dst)
        os.remove(dst)
    os.symlink(src_abs, dst)
    return ("linked", dst)


def collect(root, prefix, lef_subdir, gds_subdir, out_lef, out_gds,
            overwrite=False):
    out_lef_dir = os.path.join(root, out_lef)
    out_gds_dir = os.path.join(root, out_gds)
    os.makedirs(out_lef_dir, exist_ok=True)
    os.makedirs(out_gds_dir, exist_ok=True)

    prefix_lower = prefix.lower()
    rom_dirs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.lower().startswith(prefix_lower):
            # avoid scanning the output folders themselves
            if os.path.abspath(full) in (os.path.abspath(out_lef_dir),
                                         os.path.abspath(out_gds_dir)):
                continue
            rom_dirs.append(full)

    if not rom_dirs:
        print("[WARN] no folder starting with '%s' found under %s" % (prefix, root))
        return

    stats = {"lef": 0, "gds": 0, "lef_skip": 0, "gds_skip": 0, "no_lef": 0, "no_gds": 0}

    for rom in rom_dirs:
        rom_name = os.path.basename(rom)

        lef_dir = find_subdir(rom, lef_subdir)
        if lef_dir is None:
            print("[--] %-55s no %s/ subfolder" % (rom_name, lef_subdir))
            stats["no_lef"] += 1
        else:
            lef_files = list_files_with_ext(lef_dir, ".lef")
            if not lef_files:
                print("[--] %-55s no *.lef in %s/" % (rom_name, lef_subdir))
            for f in lef_files:
                action, dst = transfer(f, out_lef_dir, overwrite)
                if action == "skipped":
                    stats["lef_skip"] += 1
                    print("[skip] %s already exists, skipping %s" % (os.path.basename(dst), rom_name))
                else:
                    stats["lef"] += 1
                    print("[lef ] %s -> %s" % (os.path.basename(dst), os.path.abspath(f)))

        gds_dir = find_subdir(rom, gds_subdir)
        if gds_dir is None:
            print("[--] %-55s no %s/ subfolder" % (rom_name, gds_subdir))
            stats["no_gds"] += 1
        else:
            gds_files = list_files_with_ext(gds_dir, ".gds")
            if not gds_files:
                print("[--] %-55s no *.gds in %s/" % (rom_name, gds_subdir))
            for f in gds_files:
                action, dst = transfer(f, out_gds_dir, overwrite)
                if action == "skipped":
                    stats["gds_skip"] += 1
                    print("[skip] %s already exists, skipping %s" % (os.path.basename(dst), rom_name))
                else:
                    stats["gds"] += 1
                    print("[gds ] %s -> %s" % (os.path.basename(dst), os.path.abspath(f)))

    print("\n===== done (symbolic link) =====")
    print("folders scanned   : %d" % len(rom_dirs))
    print("LEF links         : %d (skipped %d) -> %s" % (stats["lef"], stats["lef_skip"], out_lef_dir))
    print("GDS links         : %d (skipped %d) -> %s" % (stats["gds"], stats["gds_skip"], out_gds_dir))
    if stats["no_lef"] or stats["no_gds"]:
        print("missing LEF dir   : %d; missing GDSII dir : %d" % (stats["no_lef"], stats["no_gds"]))


def main():
    parser = argparse.ArgumentParser(
        description="Create symbolic links (absolute path) in lef/ and gds/ for each rom* folder's LEF/*.lef and GDSII/*.gds")
    parser.add_argument("--root", default=".",
                        help="Root directory containing the rom* folders (default: current directory)")
    parser.add_argument("--prefix", default="rom",
                        help="Folder name prefix to process (default: rom)")
    parser.add_argument("--lef-subdir", default="LEF",
                        help="Subfolder name holding LEF files (default: LEF)")
    parser.add_argument("--gds-subdir", default="GDSII",
                        help="Subfolder name holding GDS files (default: GDSII)")
    parser.add_argument("--out-lef", default="lef",
                        help="Output folder name for LEF (default: lef)")
    parser.add_argument("--out-gds", default="gds",
                        help="Output folder name for GDS (default: gds)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite when a link/file with the same name exists (default: skip)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print("[ERROR] root is not a directory: %s" % root)
        sys.exit(1)

    collect(root, args.prefix, args.lef_subdir, args.gds_subdir,
            args.out_lef, args.out_gds, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
