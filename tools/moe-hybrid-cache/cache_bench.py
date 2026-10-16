"""Decode-speed and greedy-output benchmark for llama-server configs on gpt-oss-20b.

Usage: python cache_bench.py OUT.json BIN_DIR "label:extra args" ...
Each config starts a fresh llama-server, runs the chat-templated prompts of every set in
prompts/ (en_chat, code, thai, math) greedily, and stores per-prompt decode speed and text.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

MODEL = os.path.expanduser(os.environ.get("MODEL", "gpt-oss-20b-MXFP4.gguf"))
PROMPTS = os.path.expanduser(os.environ.get("PROMPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")))
SETS = os.environ.get("SETS", "en_chat,code,thai,math").split(",")
PORT = 8091
N_PREDICT = int(os.environ.get("N_PREDICT", "256"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "8"))
BASE = ["-c", "8192", "-fa", "on", "-t", "6", "-ngl", "99", "-np", "1", "--cache-ram", "0", "--no-webui"]


def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(proc, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def vram_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def cache_stats(log_path):
    # last "moe_cache stats" line printed by the hybrid cache, if any
    last = None
    with open(log_path, errors="replace") as f:
        for line in f:
            if "moe_cache_stats:" in line:
                last = line
    if last is None:
        return None
    return {k: float(v) for k, v in re.findall(r"(\w+)=([-0-9.]+)", last.split("moe_cache_stats:", 1)[1])}


def save_partial(res):
    # keep what is done so far, in case the machine goes down mid-run
    path = os.environ.get("PARTIAL_OUT")
    if path:
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)


def run(bin_dir, label, extra):
    log_path = f"server-{label}.log"
    cmd = [f"{bin_dir}/llama-server", "-m", MODEL, "--port", str(PORT), *BASE, *shlex.split(extra)]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    res = {"label": label, "extra": extra, "bin": bin_dir, "sets": {}}
    try:
        if not wait_ready(proc):
            tail = open(log_path, errors="replace").read().strip().splitlines()[-5:]
            res["error"] = " | ".join(tail)
            print(f"{label}: FAILED to start: {res['error']}", flush=True)
            return res
        res["vram_mb"] = vram_used_mb()
        post("/completion", {"prompt": "Hello", "n_predict": 8, "cache_prompt": False})  # warm-up
        for s in SETS:
            prompts = [p for p in open(f"{PROMPTS}/{s}.txt").read().splitlines() if p.strip()][:N_PROMPTS]
            rows = []
            st0 = cache_stats(log_path)
            for p in prompts:
                text = post("/apply-template", {"messages": [{"role": "user", "content": p}]})["prompt"]
                r = post("/completion", {"prompt": text, "n_predict": N_PREDICT, "temperature": 0, "top_k": 1,
                                         "cache_prompt": False, "ignore_eos": True, "return_tokens": True})
                t = r["timings"]
                rows.append({"tps": t["predicted_per_second"], "n": t["predicted_n"],
                             "pp_tps": t["prompt_per_second"], "pp_n": t["prompt_n"], "text": r["content"],
                             "tokens": r.get("tokens", [])})
            st1 = cache_stats(log_path)
            tok = sum(x["n"] for x in rows)
            ms = sum(x["n"] / x["tps"] * 1000 for x in rows)
            entry = {"rows": rows, "tps": tok / ms * 1000, "n_tokens": tok}
            if st0 is None and st1 is not None:
                st0 = {k: 0.0 for k in st1}  # no stats line yet: the counters start at zero
            if st0 is not None and st1 is not None:
                entry["stats_before"], entry["stats_after"] = st0, st1
            res["sets"][s] = entry
            save_partial(res)
            print(f"{label:16s} {s:8s} decode {entry['tps']:6.2f} tok/s over {tok} tok", flush=True)
        tok = sum(e["n_tokens"] for e in res["sets"].values())
        ms = sum(x["n"] / x["tps"] * 1000 for e in res["sets"].values() for x in e["rows"])
        res["tps"] = tok / ms * 1000
        print(f"{label:16s} ALL      decode {res['tps']:6.2f} tok/s | VRAM {res['vram_mb']} MB | {extra}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)
        res["final_stats"] = cache_stats(log_path)
    return res


if __name__ == "__main__":
    out, bin_dir = sys.argv[1], os.path.expanduser(sys.argv[2])
    results = []
    if os.path.exists(out):
        results = json.load(open(out))
    for spec in sys.argv[3:]:
        label, extra = spec.split(":", 1)
        results = [r for r in results if r["label"] != label] + [run(bin_dir, label, extra)]
        json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
