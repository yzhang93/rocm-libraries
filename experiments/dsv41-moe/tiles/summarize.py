import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tile_sweep.json"))
key = sys.argv[2] if len(sys.argv) > 2 else "us"
T = sorted({r["T"] for r in d["timing"]}); tiles = []
for r in d["timing"]:
    if r["tile"] not in tiles: tiles.append(r["tile"])
short = lambda t: t.replace("MT", "").replace("x256", "")
for shape in ("gemm1", "gemm2"):
    print(f"\n{shape} ({key}, µs per launch)")
    print("T".rjust(6), *[short(t).rjust(9) for t in tiles], "  best vs 64x64")
    for t in T:
        rs = {r["tile"]: r for r in d["timing"] if r["shape"] == shape and r["T"] == t}
        base = rs["MT64x64x256"][key]; best = min(rs.values(), key=lambda r: r[key])
        print(str(t).rjust(6), *[f"{rs[x][key]:9.1f}" for x in tiles], f"  {short(best['tile'])} {base / best[key]:.2f}x")
print(f"\nGEMM1 + GEMM2 ({key}): 64x64 for both vs best tile per shape")
for t in T:
    s64 = sum(r[key] for r in d["timing"] if r["T"] == t and r["tile"] == "MT64x64x256")
    picks = [min((r for r in d["timing"] if r["T"] == t and r["shape"] == sh), key=lambda r: r[key]) for sh in ("gemm1", "gemm2")]
    sb = sum(p[key] for p in picks)
    print(f"  T={t:5d}: {s64:7.1f} -> {sb:7.1f} ({s64 / sb:.2f}x)  [{short(picks[0]['tile'])} / {short(picks[1]['tile'])}]")
