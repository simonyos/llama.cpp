"""Compare greedy outputs of two configs stored by cache_bench.py.

Usage: python compare_outputs.py A.json:labelA B.json:labelB
Prints, per set, how many prompts give the same tokens and the first differing token position.
"""
import json
import sys


def load(spec):
    path, label = spec.rsplit(":", 1)
    for r in json.load(open(path)):
        if r["label"] == label:
            return r
    raise SystemExit(f"{label} not in {path}")


a, b = load(sys.argv[1]), load(sys.argv[2])
total_same, total = 0, 0
for s, ea in a["sets"].items():
    eb = b["sets"].get(s)
    if eb is None:
        continue
    firsts = []
    for ra, rb in zip(ea["rows"], eb["rows"]):
        ta, tb = ra.get("tokens") or [], rb.get("tokens") or []
        if not ta or not tb:
            ta, tb = list(ra["text"]), list(rb["text"])  # fall back to characters
        n = min(len(ta), len(tb))
        first = next((i for i in range(n) if ta[i] != tb[i]), None if len(ta) == len(tb) else n)
        firsts.append(first)
    same = sum(f is None for f in firsts)
    total_same += same
    total += len(firsts)
    diverged = ", ".join(str(f) for f in firsts if f is not None)
    print(f"{s:8s} identical {same}/{len(firsts)}" + (f" | first differing token: {diverged}" if diverged else ""))
print(f"ALL      identical {total_same}/{total}  ({a['label']} vs {b['label']})")
