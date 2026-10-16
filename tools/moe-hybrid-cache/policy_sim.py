"""Replay a routing trace (LLAMA_MOE_CACHE_TRACE) through expert-cache policies and print hit rates.

Usage: python policy_sim.py TRACE N_LAYER N_USED "layers of group 0" ... --slots SLOTS
Each group is a comma-separated list of layer indices (layers whose experts share one bank).
A miss can be admitted at once and the slot is ready for the next token, like the real cache.
"""
import argparse
import heapq
import math
import struct
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("n_layer", type=int)
ap.add_argument("n_used", type=int)
ap.add_argument("groups", nargs="+")
ap.add_argument("--slots", type=int, required=True)
ap.add_argument("--n-expert", type=int, required=True)
ap.add_argument("--tokens-per-set", type=int, default=0, help="split the hit rate in sets of this many tokens")
ap.add_argument("--no-lfu", action="store_true")
args = ap.parse_args()

groups = [[int(x) for x in g.split(",")] for g in args.groups]
group_of = {}
for ig, g in enumerate(groups):
    for il in g:
        group_of[il] = ig

# read the trace: records of (il, n_used ids), tokens start at layer 0
data = open(args.trace, "rb").read()
rec = 4 * (1 + args.n_used)
tokens = []
cur = None
for off in range(0, len(data) - rec + 1, rec):
    vals = struct.unpack_from(f"<{1 + args.n_used}i", data, off)
    il, ids = vals[0], vals[1:]
    if il == min(group_of) or cur is None:
        cur = {}
        tokens.append(cur)
    cur[il] = ids
print(f"{len(tokens)} tokens")


def split_last_first(total):
    # slots to the last layers first, like the current cache
    n = [0] * len(groups)
    left = total
    for il in range(args.n_layer - 1, -1, -1):
        if il in group_of and left > 0:
            k = min(left, args.n_expert)
            n[group_of[il]] += k
            left -= k
    return n


def split_by_layers(total):
    n_layers = sum(len(g) for g in groups)
    return [total * len(g) // n_layers for g in groups]


def warm(n_slots):
    # start with the last layers of each group, like the cache
    res = []
    for ig, g in enumerate(groups):
        keys = []
        for il in sorted(g, reverse=True):
            for e in range(args.n_expert):
                if len(keys) < n_slots[ig]:
                    keys.append((il, e))
        res.append(keys)
    return res


def run(policy, n_slots, tau=64.0, admit_filter=False):
    hits = [0] * len(tokens)
    uses = [0] * len(tokens)
    swaps = 0
    init = warm(n_slots)
    if policy == "lru":
        caches = [OrderedDict((k, True) for k in init[ig]) for ig in range(len(groups))]
    else:
        # decayed LFU in the log domain: L = log(sum exp(t_i/tau)), the order of the keys does not change between uses
        caches = [dict((k, -1e30) for k in init[ig]) for ig in range(len(groups))]
        heaps = [[(-1e30, k) for k in init[ig]] for ig in range(len(groups))]
        for h in heaps:
            heapq.heapify(h)
        score = {}
    for t, tok in enumerate(tokens):
        pending = []
        for il, ids in tok.items():
            ig = group_of[il]
            c = caches[ig]
            for e in ids:
                k = (il, e)
                uses[t] += 1
                if policy != "lru":
                    x = t / tau
                    s = score.get(k, -1e30)
                    score[k] = max(s, x) + math.log1p(math.exp(-abs(s - x)))
                if k in c:
                    hits[t] += 1
                    if policy == "lru":
                        c.move_to_end(k)
                    else:
                        c[k] = score[k]
                        heapq.heappush(heaps[ig], (score[k], k))
                else:
                    pending.append((ig, k))
        # admissions after the token, ready for the next one
        for ig, k in pending:
            c = caches[ig]
            if k in c or n_slots[ig] == 0:
                continue
            if policy == "lru":
                c.popitem(last=False)
                c[k] = True
            else:
                h = heaps[ig]
                while h[0][1] not in c or c[h[0][1]] != h[0][0]:
                    heapq.heappop(h)  # stale entry
                victim = h[0][1]
                if admit_filter and c[victim] >= score[k]:
                    continue
                heapq.heappop(h)
                del c[victim]
                c[k] = score[k]
                heapq.heappush(h, (score[k], k))
            swaps += 1
    return hits, uses, swaps


def run_slru(n_slots, frac):
    # segmented LRU: a hit in probation moves the expert to the protected segment, misses enter probation
    hits = [0] * len(tokens)
    uses = [0] * len(tokens)
    swaps = 0
    init = warm(n_slots)
    prob = [OrderedDict((k, True) for k in init[ig]) for ig in range(len(groups))]
    prot = [OrderedDict() for _ in groups]
    cap = [int(frac * n) for n in n_slots]
    for t, tok in enumerate(tokens):
        pending = []
        for il, ids in tok.items():
            ig = group_of[il]
            for e in ids:
                k = (il, e)
                uses[t] += 1
                if k in prot[ig]:
                    hits[t] += 1
                    prot[ig].move_to_end(k)
                elif k in prob[ig]:
                    hits[t] += 1
                    del prob[ig][k]
                    prot[ig][k] = True
                    if len(prot[ig]) > cap[ig]:
                        old, _ = prot[ig].popitem(last=False)
                        prob[ig][old] = True
                else:
                    pending.append((ig, k))
        for ig, k in pending:
            if k in prob[ig] or k in prot[ig] or n_slots[ig] == 0:
                continue
            if prob[ig]:
                prob[ig].popitem(last=False)
            else:
                prot[ig].popitem(last=False)
            prob[ig][k] = True
            swaps += 1
    return hits, uses, swaps


def report(name, res):
    hits, uses, swaps = res
    line = f"{name:34s} hit {sum(hits) / sum(uses):.4f}  swaps/token {swaps / len(tokens):5.1f}"
    if args.tokens_per_set:
        per = []
        for s in range(0, len(tokens), args.tokens_per_set):
            h, u = sum(hits[s:s + args.tokens_per_set]), sum(uses[s:s + args.tokens_per_set])
            per.append(f"{h / u:.3f}")
        line += "  per set " + " / ".join(per)
    print(line, flush=True)


for split_name, split in [("last-first", split_last_first(args.slots)), ("by-layers", split_by_layers(args.slots))]:
    print(f"-- slots per group ({split_name}): {split}")
    report(f"LRU [{split_name}]", run("lru", split))
    for frac in (0.5, 0.7, 0.85):
        report(f"SLRU protected={frac} [{split_name}]", run_slru(split, frac))
    if args.no_lfu:
        continue
    for tau in (32.0, 128.0, 512.0):
        report(f"decayed LFU tau={tau:.0f} [{split_name}]", run("lfu", split, tau))
        report(f"decayed LFU tau={tau:.0f} + filter [{split_name}]", run("lfu", split, tau, admit_filter=True))
