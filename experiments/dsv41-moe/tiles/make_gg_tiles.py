"""Write the gfx950 grouped MXFP8 logic file with extra macro tiles taken from the regular
(non-grouped) MXFP8 Origami library.

Each new solution is the shipped grouped MT64x64x256 solution with the fields in which the
regular target tile differs from the regular MT64x64x256 solution copied over. The grouped
and regular MT64x64x256 solutions differ only in Stream-K, GlobalSplitU, the workspace check
and the accumulation mode, and no tile-specific field overlaps those, so every clone keeps
the grouped settings (StreamK 0, GSU 1, MultipleBuffer).

The shipped solution stays index 0 with its grid entry, and each clone gets a grid entry at
a far-off size, so the heuristic's default choice is unchanged and a clone is used only
when it is picked by kernel name (GG_KERNEL_MATCH in gg_shim.cpp)."""

import argparse
import copy
import os

import yaml

LOGIC = ("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950")
GG = "GridBased/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_GG_HA_S_SAB_SAV_UserArgs.yaml"
REG = "Origami/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs.yaml"
TILES = [(32, 128), (128, 32), (256, 32), (128, 64), (256, 64), (128, 128), (256, 128)]
NAME_KEYS = {"KernelNameMin", "SolutionNameMin", "BaseName", "SolutionUID", "SolutionIndex", "ProblemType"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="rocm-libraries checkout")
    ap.add_argument("--check", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()
    base = os.path.join(args.repo, LOGIC)
    gg_path = os.path.join(base, GG)
    with open(gg_path) as f:
        gg = yaml.load(f, Loader=yaml.CSafeLoader)
    with open(os.path.join(base, REG)) as f:
        reg = yaml.load(f, Loader=yaml.CSafeLoader)

    sols, grid = gg[5], gg[7]
    assert len(sols) == 1 and len(grid) == 1, "expected the single shipped grouped solution"
    g64 = sols[0]
    default = reg["DefaultSolution"]
    by_tile = {(s["MacroTile0"], s["MacroTile1"]): {**default, **s} for s in reg["Solutions"]
               if s["DepthU"] == 256}
    r64 = by_tile[(64, 64)]
    grouped_keys = {k for k in set(g64) | set(r64) if k not in NAME_KEYS and g64.get(k) != r64.get(k)}

    for i, tile in enumerate(TILES, start=1):
        rt = by_tile[tile]
        tile_keys = {k for k in set(rt) | set(r64) if k not in NAME_KEYS and rt.get(k) != r64.get(k)}
        clash = tile_keys & grouped_keys
        if clash:
            raise SystemExit(f"MT{tile}: fields both tile- and grouped-specific: {sorted(clash)}")
        s = copy.deepcopy(g64)
        for k in tile_keys:
            if k in rt:
                s[k] = copy.deepcopy(rt[k])
            else:
                s.pop(k, None)
        s["SolutionIndex"] = i
        s["SolutionNameMin"] = (rt["SolutionNameMin"].replace("_HA_S_SAB_SAV_UserArgs", "_GG_HA_S_SAB_SAV_UserArgs")
                                .replace("_SK3_", "_SK0_").replace("_GSU0_", "_GSU1_"))
        s["BaseName"] = rt["BaseName"]
        sols.append(s)
        grid.append([[65536 * i, 65536, 1, 65536], [i, 0.0]])
        print(f"index {i}: MT{tile[0]}x{tile[1]}x256, {len(tile_keys)} tile fields, "
              f"WorkGroup {s['WorkGroup']}, MIWaveGroup {s['MIWaveGroup']}, LdsNumBytes {s['LdsNumBytes']}")

    if args.check:
        return
    with open(gg_path, "w") as f:
        yaml.dump(gg, f, Dumper=yaml.CSafeDumper, default_flow_style=None, width=120)
    print("wrote", gg_path)


if __name__ == "__main__":
    main()
