"""Append geko-tuned MX kernels to a gfx950 grouped MX logic file as extra grouped solutions
(the MXFP8 file, or with --types f4f8 the mixed MXFP4 x MXFP8 one).

geko tunes batched (non-grouped) proxies of the MoE GEMMs with Stream-K off, so its winners already
carry the grouped settings (StreamK 0, GlobalSplitU 1, MultipleBuffer, WorkspaceCheck [4, 0, 1]).
Each clone is the shipped grouped MT64x64x256 solution with every field it shares with the winner
taken from the winner. The winners come from the TensileLite in the geko tree, so fields that only
one of the two versions has keep the grouped solution's value (or are absent), as do the fields in
VERSION_KEYS, which differ only in representation.

Clones are appended after the existing solutions with grid entries at far-off sizes, so the
heuristic's default choice is unchanged; pick them by kernel name (GG_KERNEL_MATCH in gg_shim.cpp)."""

import argparse
import copy
import json
import os

import yaml

LOGIC = "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950"
GG = {"mxfp8": "GridBased/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_GG_HA_S_SAB_SAV_UserArgs.yaml",
      "f4f8": "GridBased/gfx950_Cijk_Alik_Bljk_F4F8_F4BS_MXAE8B32_MXBE8B32_BH_Bias_GG_HA_S_SAB_SAV_UserArgs.yaml"}
NAME_KEYS = {"KernelNameMin", "SolutionNameMin", "BaseName", "SolutionUID", "SolutionIndex", "ProblemType"}
VERSION_KEYS = {"CodeObjectVersion", "InternalSupportParams", "CustomKernelName"}
GROUPED_KEYS = ["StreamK", "GlobalSplitU", "GlobalSplitUAlgorithm", "WorkspaceCheck", "_GlobalAccumulation"]


def grouped_value(sol, key):
    """sol[key], reading a newer Tensile's data-parallel TileProcessingStrategy as StreamK 0."""
    if key == "StreamK" and key not in sol and sol.get("TileProcessingStrategy") == "None":
        return 0
    return sol.get(key)


def winners(paths):
    """(size, solution) for every exact-logic entry of the geko libraries, one per distinct solution."""
    out, seen = [], {}
    for p in paths:
        with open(p) as f:
            lib = yaml.load(f, Loader=yaml.CSafeLoader)
        sols = [{**lib.get("DefaultSolution", {}), **s} for s in lib["Solutions"]]
        for size, (idx, gflops) in lib["ExactLogic"]:
            s = sols[idx]
            key = json.dumps({k: v for k, v in s.items() if k not in NAME_KEYS}, sort_keys=True, default=str)
            if key in seen:
                seen[key]["sizes"].append((size[:4], gflops))
                continue
            seen[key] = {"sol": s, "sizes": [(size[:4], gflops)]}
            out.append(seen[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="rocm-libraries checkout with the grouped logic file")
    ap.add_argument("--libs", nargs="+", required=True, help="geko logic files (full_libs or libs)")
    ap.add_argument("--types", choices=GG, default="mxfp8",
                    help="grouped logic file: MXFP8 x MXFP8 or MXFP4 weights x MXFP8 tokens")
    ap.add_argument("--clones", default="geko_clones.json", help="plan output, next to this script")
    ap.add_argument("--check", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()
    gg_path = os.path.join(args.repo, LOGIC, GG[args.types])
    with open(gg_path) as f:
        gg = yaml.load(f, Loader=yaml.CSafeLoader)
    sols, grid = gg[5], gg[7]
    g64 = sols[0]
    prefix = g64["SolutionNameMin"].split("_MT")[0]

    plan = []
    for w in winners(args.libs):
        ws = w["sol"]
        clash = [k for k in GROUPED_KEYS if grouped_value(ws, k) != g64.get(k)]
        tile = f"MT{ws['MacroTile0']}x{ws['MacroTile1']}x{ws['DepthU']}"
        if clash:
            print(f"skip {tile} for {w['sizes']}: grouped fields differ: "
                  + ", ".join(f"{k}={grouped_value(ws, k)!r} (grouped {g64.get(k)!r})" for k in clash))
            continue
        s = copy.deepcopy(g64)
        changed = []
        for k in s:
            if k in NAME_KEYS or k in VERSION_KEYS or k not in ws or ws[k] == s[k]:
                continue
            s[k] = copy.deepcopy(ws[k])
            changed.append(k)
        i = len(sols)
        s["SolutionIndex"] = i
        s["SolutionNameMin"] = prefix + "_MT" + ws["SolutionNameMin"].split("_MT", 1)[1]
        s["BaseName"] = ws["BaseName"]
        sols.append(s)
        grid.append([[65536 * i, 65536, 1, 65536], [i, 0.0]])
        plan.append({"index": i, "tile": tile, "sizes": w["sizes"], "fields": len(changed),
                     "WorkGroup": s["WorkGroup"], "MIWaveGroup": s["MIWaveGroup"],
                     "name": s["SolutionNameMin"]})
        print(f"index {i}: {tile}, {len(changed)} fields from geko, WorkGroup {s['WorkGroup']}, "
              f"MIWaveGroup {s['MIWaveGroup']}, sizes {[sz for sz, _ in w['sizes']]}")

    if args.check:
        return
    with open(gg_path, "w") as f:
        yaml.dump(gg, f, Dumper=yaml.CSafeDumper, default_flow_style=None, width=120)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.clones), "w") as f:
        json.dump(plan, f, indent=1)
    print("wrote", gg_path)


if __name__ == "__main__":
    main()
