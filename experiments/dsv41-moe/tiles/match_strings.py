"""Build tile_sweep.py --tiles arguments (label=substring) for the kernels in a gg_shim algos.txt.

Shipped and experiment-3 tiles are labeled by macro tile; geko clones (geko_clones.json) are labeled
geko_g{1,2}_b{batch}n{N}_{tile}, listing every proxy they won. A clone is matched to its built kernel
by macro tile and the largest overlap of name tokens, because the build regenerates kernel names.
Clones that differ only in fields kernel names omit (e.g. WorkGroupMapping) build to one kernel and
share its label. Each substring is the shortest run of name tokens that occurs in exactly one kernel."""

import argparse
import json
import re


def tokens(name):
    return name.split("_")


def shortest_unique(name, names):
    toks = tokens(name)
    for width in range(1, len(toks) + 1):
        for i in range(len(toks) - width + 1):
            sub = "_" + "_".join(toks[i:i + width]) + "_"
            if sum(sub in n + "_" for n in names) == 1:
                return sub
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", required=True)
    ap.add_argument("--clones", required=True)
    ap.add_argument("--tiles", nargs="*", default=[], help="shipped/experiment-3 tiles to include, e.g. 64x64")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    names = list(dict.fromkeys(m.group(1) for m in re.finditer(r"algo \d+\*?\s+(\S+)", open(args.algos).read())))
    mt = {n: re.search(r"_MT(\d+x\d+)x\d+_", n).group(1) for n in names}
    clones = json.load(open(args.clones))
    by_kernel = {}
    for c in clones:
        tile = c["tile"][2:].rsplit("x", 1)[0]
        want = set(tokens(c["name"]))
        cand = [n for n in names if mt[n] == tile]
        best = max(cand, key=lambda n: (len(want & set(tokens(n))), n not in by_kernel))
        entry = by_kernel.setdefault(best, {"tile": tile, "gemms": [], "proxies": []})
        gemm = "g1" if c["sizes"][0][0][0] == 1152 else "g2"
        if gemm not in entry["gemms"]:
            entry["gemms"].append(gemm)
        entry["proxies"] += [f"b{s[0][2]}n{s[0][1]}" for s in c["sizes"]]
    labels = {f"geko_{''.join(e['gemms'])}_{'+'.join(e['proxies'])}_{e['tile']}": n for n, e in by_kernel.items()}
    for tile in args.tiles:
        cand = [n for n in names if mt[n] == tile and n not in by_kernel]
        assert len(cand) == 1, (tile, len(cand))
        labels[tile] = cand[0]
    match = {label: shortest_unique(n, names) for label, n in labels.items()}
    order = args.tiles + [k for k in match if k not in args.tiles]
    if args.out_json:
        json.dump({k: match[k] for k in order}, open(args.out_json, "w"), indent=1)
    print(",".join(f"{k}={match[k]}" for k in order))


if __name__ == "__main__":
    main()
