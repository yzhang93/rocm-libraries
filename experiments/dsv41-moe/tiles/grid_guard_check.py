"""Which grouped kernel the heuristic picks (no GG_KERNEL_MATCH) for shapes on and off the
tuned DSv4.1 MoE shapes, with gg_micro's correctness check on each."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gg_micro import check_case  # noqa: E402
from moe_gg_bench import Shim  # noqa: E402

os.environ.pop("GG_KERNEL_MATCH", None)
shim = Shim()
print("hipblaslt:", shim.hipblaslt_lib, flush=True)
names = []
create = shim.create


def capture(*a, **k):
    p, n = create(*a, **k)
    names.append(n)
    return p, n


shim.create = capture
shapes = [(64, 512), (1024, 5120), (1152, 5120), (1152, 576), (2048, 2048), (5120, 576), (5120, 5120),
          (7168, 576)]
counts = {"N<=32": [30, 0, 17, 32, 5, 1, 30, 30], "N>32": [100, 70, 0, 128, 64, 96, 33, 90]}
ok = True
for ab in (0, 3):
    for M, K in shapes:
        for label, c in counts.items():
            r = check_case(shim, M, K, c, ab=ab)
            mt = re.search(r"_MT(\d+x\d+x\d+)_", names[-1]).group(1)
            ok &= r["ok"]
            print(json.dumps(dict(ab=ab, M=M, K=K, cols=label, tile=mt, ok=r["ok"],
                                  err=round(r["worst_rel_err"], 5), pad=r["padding_untouched"])), flush=True)
print("ALL OK" if ok else "FAILURES")
