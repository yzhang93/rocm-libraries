import json, sys
for l in sys.stdin:
    if l.startswith("check "):
        r = json.loads(l[6:])
        print("%-5s %-20s M=%4d K=%4d groups=%3d active=%3d err=%.4f pad_ok=%s ok=%s" % (
            r.get("ab", "mxfp8"), r["case"], r["M"], r["K"], r["groups"], r["active"], r["worst_rel_err"], r["padding_untouched"], r["ok"]))
    elif not l.startswith("[aiter]"):
        print(l.rstrip())
