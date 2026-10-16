"""Summary table of cache_bench.py results: decode tok/s per set, cache stats per set, prefill speed."""
import json
import sys

SETS = ["en_chat", "code", "thai", "math"]
ZERO = {"tokens": 0, "hits": 0, "misses": 0, "swaps": 0, "cpu_layers": 0}

for path in sys.argv[1:]:
    for r in json.load(open(path)):
        if "sets" not in r or not r["sets"]:
            print(f"{r['label']}: {r.get('error', 'no data')}")
            continue
        tps = " / ".join(f"{r['sets'][s]['tps']:5.1f}" for s in SETS if s in r["sets"])
        pp = [x["pp_tps"] for s in SETS if s in r["sets"] for x in r["sets"][s]["rows"]]
        line = f"{r['label']:18s} ALL {r['tps']:5.1f} tok/s | en/code/th/math {tps} | prefill {sum(pp)/len(pp):5.0f} tok/s | VRAM {r.get('vram_mb')} MB"
        stats = []
        for i, s in enumerate(SETS):
            e = r["sets"].get(s, {})
            a, b = e.get("stats_before"), e.get("stats_after")
            if b is None and i == 0 and "stats_before" in r["sets"].get(SETS[1], {}):
                a, b = ZERO, r["sets"][SETS[1]]["stats_before"]  # older runs: the first set ends where the second starts
            if a is None or b is None:
                continue
            t = b["tokens"] - a["tokens"]
            h, m = b["hits"] - a["hits"], b["misses"] - a["misses"]
            ru = ""
            if "reuse_n" in b and b["reuse_n"] - a.get("reuse_n", 0) > 0:
                ru = f" reuse={(b['reuse']-a.get('reuse', 0))/(b['reuse_n']-a.get('reuse_n', 0)):.2f}"
            stats.append(f"{s} h={h/(h+m):.3f} miss/tok={m/t:.1f} swap/tok={(b['swaps']-a['swaps'])/t:.1f} cpuL/tok={(b['cpu_layers']-a['cpu_layers'])/t:.1f}{ru}")
        print(line)
        if stats:
            print("    " + " | ".join(stats))
