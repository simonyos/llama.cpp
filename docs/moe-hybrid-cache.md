# Hybrid VRAM expert cache for llama.cpp

An experimental patch for [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs MoE models whose experts do not fit in VRAM about **2x faster** than the usual `--n-cpu-moe` split on an 8 GB GPU.

The idea: MoE routing is sticky, so about half of the experts a token uses were also used by the previous token. The patch keeps every expert in pinned RAM and keeps the most recently used ones in a VRAM cache.
- **Hits** run on the GPU from the cache.
- **Misses** run on the CPU from RAM, while the GPU runs the hits.
- Missed experts are copied into the cache in the background, so later tokens hit.

> **Status:** experimental, single GPU, CUDA only, tested on one machine (RTX 3060 Ti 8 GB). Not proposed upstream.
>
> **Who did what:** [@simonyos](https://github.com/simonyos) set the hypothesis, ran every experiment on a home PC and decided what to try next.
> The code, benchmarks and analysis were written by Claude (Anthropic's AI coding agent) under @simonyos's direction.
> Commits carry a `Co-Authored-By: Claude` line. Please review the code before you rely on it.

## Results

Decode speed in tok/s, greedy, 4 sets (English chat, code, Thai, math) × 8 prompts × 256 tokens each, `-ngl 99 -c 8192 -fa on -t 6`:

| Model | Stock `--n-cpu-moe` (best that fits) | Upstream `--moe-cache-mib` | **This patch** | Speed-up vs stock |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 | 43.6 (`--n-cpu-moe 11`) | 64.9 (5600 MiB) | **87.1** (448 slots) | **2.00x** |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M | 37.3 (`--n-cpu-moe 33`) | 56.9 (5450 MiB) | **75.9** (1984 slots) | **2.03x** |

Per prompt set (English / code / Thai / math):

| Model | Stock | Upstream `--moe-cache-mib` | This patch | Cache hit rate |
|---|---|---|---|---|
| gpt-oss-20b | 43.6 / 43.6 / 43.5 / 43.5 | 74.6 / 58.5 / 70.4 / 59.2 | **94.1 / 81.7 / 90.9 / 82.9** | .970 / .934 / .963 / .937 |
| Qwen3-30B-A3B | 37.3 / 37.3 / 37.2 / 37.3 | 66.0 / 59.0 / 69.1 / 42.2 | **83.9 / 77.1 / 86.8 / 61.5** | .910 / .898 / .928 / .821 |

- The VRAM use is the same as the stock baseline (7.2–7.7 GB of 8 GB).
- Upstream `--moe-cache-mib` already exists in llama.cpp: it uploads every missed expert over PCIe before computing it on the GPU (hit rate 0.948 on gpt-oss and 0.890 on Qwen3 here).
  This patch computes misses on the CPU instead and fills the cache in the background, which is about 1.34x faster than upstream's cache on both models.
- Prefill of the short chat prompts (20–90 tokens): gpt-oss 208 → 263 tok/s, Qwen3 74 → 99 tok/s.
- The GPU was power-limited to 150 W for all runs (see Limitations).
- The speed-ups are measured against the best stock placement that fits (`--n-cpu-moe`), not against all experts on the CPU.
  With all experts on the CPU (`--cpu-moe`) as the baseline, gpt-oss-20b would show about 3.5x (24.9 tok/s with a stock build). Keep this in mind when you compare with numbers that use `--cpu-moe` or `-ncmoe <all layers>` as the baseline.

### Hardware and software

- CPU AMD Ryzen 5 5500 (6 cores / 12 threads), RAM 4 × 8 GB DDR4-2667 (dual channel)
- GPU NVIDIA RTX 3060 Ti 8 GB, PCIe 3.0 x16 (13.4 GB/s pinned H2D), power limit 150 W
- Pop!_OS 24.04, Linux 7.0.11, NVIDIA driver 580.159.03, CUDA 13.4
- llama.cpp commit `ceefd16` (2026-10-16) + `llama-moe-hybrid-cache.patch`

## Build

```bash
git clone --branch moe-hybrid-cache https://github.com/simonyos/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build -j --target llama-server llama-completion llama-perplexity
```

- Set `CMAKE_CUDA_ARCHITECTURES` for your GPU, for example `86-real` for RTX 30xx or `89-real` for RTX 40xx. Only `86` has been tested.
- The branch is upstream llama.cpp `ceefd16` (2026-10-16) plus one patch commit. To apply it to your own checkout instead, download `llama-moe-hybrid-cache.patch` from the [release](https://github.com/simonyos/llama.cpp/releases) and run `git checkout ceefd16 && git apply llama-moe-hybrid-cache.patch`.
- Our build used the CUDA 13.4 toolkit from pip wheels with CUDA 13.0 runtime headers. That combination also needed `-DCMAKE_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`. A normal CUDA toolkit install does not need it.

## Run

```bash
# gpt-oss-20b
./build/bin/llama-server -m gpt-oss-20b-MXFP4.gguf -ngl 99 -c 8192 -fa on -t 6 \
    --cpu-moe --load-mode none --moe-cache-slots 448 --moe-cache-admit 8

# Qwen3-30B-A3B
./build/bin/llama-server -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -ngl 99 -c 8192 -fa on -t 6 \
    --cpu-moe --load-mode none --moe-cache-slots 1984 --moe-cache-admit 16
```

- `--cpu-moe --load-mode none` keeps all experts in pinned host RAM. This is required: the background copies read pinned memory.
- `--moe-cache-slots N` is the VRAM cache size in experts. Take the largest N that still leaves about 0.5 GB of VRAM free. One expert is 12.65 MiB for gpt-oss-20b and 2.5–2.9 MiB for Qwen3-30B-A3B Q4_K_M.
- `--moe-cache-admit K` is the number of experts that may be copied at the same time. 8–16 works well; more does not help.
- `--moe-cache-admit 0` gives a static cache. It holds exactly the experts that `--n-cpu-moe` would keep on the GPU, and it is useful for checking correctness.

## Correctness

| Check | gpt-oss-20b | Qwen3-30B-A3B |
|---|---|---|
| Static cache (`--moe-cache-admit 0`) vs stock `--n-cpu-moe`, greedy tokens identical | 32/32 prompts | 32/32 prompts |
| Static cache, teacher-forced KL divergence / PPL vs stock | KLD ≈ 0, PPL 2.9027 = 2.9027 | KLD ≈ 0, PPL 1.8222 = 1.8222 |
| Dynamic cache, KL divergence / top-1 agreement / PPL vs stock | KLD 0.0016, top-1 98.5%, PPL 2.897 vs 2.903 | KLD 0.0018, top-1 99.0%, PPL 1.825 vs 1.822 |
| Stock placement change as a control (early layers moved to the GPU with `-ot`) | KLD 0.0011, top-1 99.2% | KLD 0.0023, top-1 99.4% |
| VRAM slot contents vs RAM, byte by byte (`LLAMA_MOE_CACHE_VERIFY`) | 4 checks × 416 slots, 0 differ | 4 checks × ~1,980 slots, 0 differ |

- The static cache runs every expert on the same device as stock does, and its output is bit-identical. That checks the routing, the skipped experts and the merge.
- With the dynamic cache, an expert runs on the GPU or the CPU depending on the cache state. CPU and GPU round slightly differently, so greedy output drifts from stock after about 100 tokens on average.
  A stock `--n-cpu-moe` change does the same, and the KL divergence is in the same range as the control. Perplexity is not worse.
- KL divergence and PPL come from `llama-perplexity --kl-divergence` over 1,024 scored tokens with `-ub 1`, so they test the decode path.

## How it works

- **Cache**: one bank per expert tensor type, in VRAM, holding N expert slots. Eviction is a segmented LRU: experts that were hit again are protected from experts that were used only once.
  The cache starts with the experts of the last layers, like `--n-cpu-moe`.
- **Per layer and token**, with no CPU split in the graph:
  1. The router runs on the GPU. A scheduler copy callback starts the readback of the selected expert ids and the layer input, then lets the GPU continue.
  2. The GPU looks the ids up in a slot map. Hits run from the cache, and misses get id −1, which the CUDA kernels skip.
  3. A second callback waits for the readback. It runs the misses on the CPU with the same ggml ops as stock `--n-cpu-moe`, while the GPU computes the hits. Then it uploads the CPU result, and the GPU adds the two halves.
- **Admission**: a miss is copied into the least valuable slot right away, by a copy kernel on a second CUDA stream that reads pinned RAM.
  A DMA copy (`cudaMemcpyAsync`) would hold the copy engine for ~2 ms and stall the small per-layer uploads, so the patch avoids it.
  The copy waits for the GPU work already queued, and the slot is used only after an event says the copy is done.
- **Batches of 2–31 tokens** use a CPU split in the graph. Misses get a zero slot of the bank for batches of 5+ tokens. **Batches of 32+ tokens** (prefill) use upstream's path, which copies cached experts inside VRAM.
- **ggml changes**:
  - Negative expert ids are skipped in `mul_mat_id` and `add_id`, on the CPU and in the CUDA mmvq kernels.
  - New `ggml_backend_event_query`.
  - New `ggml_backend_cuda_set_tensor_async_kernel` for kernel copies from pinned memory.
  - The scheduler calls the copy callback before it syncs the backend.

Cost model measured on this machine, in ms per token, with misses per token from the cache stats:

| | Fixed part | Per miss | Fixed part with all experts in VRAM (extrapolated) |
|---|---|---|---|
| gpt-oss-20b, CPU split per layer (PCIe x8, 200 W) | 9.8 | 0.52 | 8.6 |
| gpt-oss-20b, this patch | 9.3 | 0.45 | 8.6 |
| Qwen3-30B-A3B, CPU split per layer | 11.4 | 0.11 | 7.0 |
| Qwen3-30B-A3B, this patch | 8.5 | 0.11 | 7.0 |

The async path removed most of the fixed per-layer cost. What is left is mostly the CPU work for the misses, which reads each missed expert from RAM at ~25 GB/s.

## Related work and how this differs

Caching MoE experts in VRAM is an active topic in llama.cpp. As of October 2026:

| | Where misses run | How experts reach VRAM | Slots | Per-layer host sync | New expert usable |
|---|---|---|---|---|---|
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) `--moe-cache-mib` (merged Oct 7) | GPU, after an upload | async copy on the compute stream, on the critical path | pooled per device group, LRU | yes, reads ids back to plan uploads | at once, after the upload |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) (draft) | CPU, from host RAM | worker thread, `ggml_backend_tensor_set` (DMA), at most 2 per layer and step | fixed per layer, LRU, starts empty | yes, a CPU split per layer | from the next decode step |
| [#29949](https://github.com/ggml-org/llama.cpp/issues/29949) (issue + branch `lru-upstream-v3`; CUDA and HIP, multi-GPU incl. `-sm tensor`, MTP) | GPU, after a fill | in-graph fill kernel that reads pinned RAM, on the critical path | fixed per layer, LRU decided on the GPU | none | at once, after the fill |
| [arXiv 2512.16473](https://arxiv.org/abs/2512.16473) (ASP-DAC 2026) | CPU | (research code, not llama.cpp) | GPU expert cache | – | – |
| **This patch** | **CPU, at the same time as the GPU hits** | **background copy kernel that reads pinned RAM, on a second stream** | **one pool per expert type, segmented LRU, starts like `--n-cpu-moe`** | **readback is async; waits only for the CPU misses** | **as soon as its copy event completes (checked every layer)** |

The core idea, GPU hits plus CPU misses plus async refill, is not new. The #29887 thread already suggests it as future work, #27861 implements it, and arXiv 2512.16473 describes it.
Fiddler (2024) and HybriMoE (2025) also run experts on the CPU next to a GPU, with static placement and with dynamic scheduling plus caching, respectively.
Earlier llama.cpp work on GPU expert caches includes [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) (closed as completed; it confirmed strong temporal locality), the RFC [#28248](https://github.com/ggml-org/llama.cpp/issues/28248), and the closed PRs [#21609](https://github.com/ggml-org/llama.cpp/pull/21609), [#21614](https://github.com/ggml-org/llama.cpp/pull/21614), [#21620](https://github.com/ggml-org/llama.cpp/pull/21620) and [#24524](https://github.com/ggml-org/llama.cpp/pull/24524).
What this patch adds:

1. **A measurement of why refill should not use the copy engine.** Reading pinned memory from a kernel instead of using DMA is not new: #29949's `LRU_FILL` already fills its pool that way.
   What we add is the measured reason it matters when the refill runs on a side stream: on this GPU, one expert-sized `cudaMemcpyAsync` held a small synchronous upload for 1989 µs, against 36 µs with a kernel copy.
   The scheduler needs such small uploads every layer, so DMA refill costs about as much as it saves (we measured 52 tok/s with DMA vs 77 tok/s with kernel copies on gpt-oss-20b at PCIe x8).
   #27861 refills through `ggml_backend_tensor_set`, which is the DMA path. That fits its author's note that unthrottled uploads "eat the entire win", and its limit of 2 uploads per layer and step.
2. **No CPU split per layer.** Two copy callbacks per layer replace the CPU split, so layers without misses cost little, and the CPU work for the misses overlaps the GPU work for the hits. This needed one scheduler change: the callback now runs before the backend sync.
3. **Exactness test.** With `--moe-cache-admit 0` the cache holds the same experts as `--n-cpu-moe`, and greedy output is bit-identical to stock (32/32 prompts on both models). Dynamic runs are compared against stock placement changes with KL divergence.
4. **Builds on master's cache.** It extends the `llama_moe_cache` that #29887 added, so it applies to current master. #27861 and #29949 were written before #29887 and conflict with it (see the #29949 thread).

Which design is faster depends on the machine. Computing a miss on the CPU reads the expert once from RAM; filling it on the GPU first moves it over PCIe.
On a PCIe 3.0 link a gpt-oss-20b expert (12.65 MiB) takes ~1.0 ms to move at 13.4 GB/s against ~0.33–0.45 ms to compute on this CPU, so CPU misses win here.
On PCIe 4.0/5.0 the two costs are close, and a design that keeps everything on the GPU (#29949) also avoids our per-layer host round trip (~1.5 ms/token on Qwen3-30B-A3B), so it may be faster there.

Measured on the same machine (Qwen3-30B-A3B, 4 prompts × 4 sets × 256 tokens, 150 W, PCIe x16):

| Build | tok/s (en / code / th / math) | overall |
|---|---|---|
| stock `--n-cpu-moe 33` | 37.3 / 37.3 / 37.2 / 37.3 | 37.3 |
| #29887 `--cpu-moe --moe-cache-mib 5450` (8 prompts per set) | 66.0 / 59.0 / 69.1 / 42.2 | 56.9 |
| #27861 at `bccbacd`, `--cpu-moe -lm none --moe-expert-cache 41` (1968 slots), 2 inserts | 68.0 / 65.0 / 69.9 / 50.2 | 62.2 |
| #27861, same, 8 inserts | 64.0 / 29.0 / 18.4 / 18.2 | 25.1 |
| **this patch**, 1984 slots, admit 16 | **84.0 / 78.3 / 86.4 / 62.0** | **76.4** |

- #27861 with 8 uploads per step slows down more the longer it runs, which is what a saturated copy engine would do. We did not profile it to prove the cause.
  That run also ended with a GPU copy-engine read fault (NVIDIA Xid 31) when the server shut down, so we stopped testing it there.
- #27861 only supports the SILU gate/up path without biases, so it could not run gpt-oss-20b.
- #29949 reports 9.7 → 37–50 t/s (3.5–4x) on a Radeon 7900 XTX and 47.6 t/s on 2x RTX 3090, both on PCIe 4.0 x16 (~26 GB/s), with a larger model (Qwen3.8-Flash-Next).
  Its baseline is all experts on the CPU (`-ncmoe 48`), not the best `--n-cpu-moe` split. Against #29887 on the same machine it reports 31.2 vs 20.1 t/s (~1.55x).
  Same-machine numbers for #29949 on this PC: being measured, will be added here.
- The SpecMD study ([arXiv 2602.03921](https://arxiv.org/abs/2602.03921)) argues that LRU/LFU locality assumptions do not fit MoE. Here decayed LFU was worse than LRU on real traces, and segmented LRU was a little better.

## Limitations

- Experimental and not proposed upstream. Single GPU, quantized experts, CUDA only, tested on Ampere (sm_86) only.
- Tested on one machine with PCIe 3.0 x16. On PCIe 4.0/5.0 the trade-off against GPU-side fill designs may change (see Related work).
- `--load-mode none` pins every expert in RAM (9.5 GiB for gpt-oss-20b, 16.4 GiB for Qwen3-30B-A3B), so RAM must hold the whole model plus the OS.
- VRAM headroom: with 448 slots, gpt-oss-20b leaves ~0.3 GB free. A raw (non-chat) prompt of 5–31 tokens then fails to create the cuBLAS handle. Chat prompts are longer and work. Use 416 slots for short raw prompts.
- Prefill is not the target. Prompts of 32+ tokens use upstream's path, which is faster than stock but still copies missing experts over PCIe.
- Some speed-ups are workload dependent. Math has the lowest hit rate on Qwen3 (0.82), and the cache gains least there.
- Our test PC reset twice under the combined CPU + GPU + RAM load with the GPU at its default 200 W limit, probably a power supply limit. All numbers here are at 150 W, which did not change stock speed or outputs.

## Debug and experiment switches (environment variables)

| Variable | Effect |
|---|---|
| `LLAMA_MOE_CACHE_SYNC=1` | use the older path with one CPU split per layer instead of the async callbacks |
| `LLAMA_MOE_CACHE_SLRU=f` | protected fraction of the segmented LRU (default 0.7, `0` = plain LRU) |
| `LLAMA_MOE_CACHE_SPLIT=last` | with several expert types, split slots by the last layers instead of by layer count |
| `LLAMA_MOE_CACHE_VERIFY=N` | every N tokens, compare all ready VRAM slots with RAM and log the result |
| `LLAMA_MOE_CACHE_TRACE=file` | append the selected experts of each layer and token to a binary file |

## Files

In this branch:
- `tools/moe-hybrid-cache/cache_bench.py` – benchmark: starts `llama-server` per config and runs the four prompt sets in `prompts/`; `summarize.py` prints the tables.
  Example: `MODEL=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf python tools/moe-hybrid-cache/cache_bench.py out.json build/bin "cache:--cpu-moe --load-mode none --moe-cache-slots 1984 --moe-cache-admit 16"`
- `tools/moe-hybrid-cache/compare_outputs.py` – compares greedy outputs of two configs token by token.
- `tools/moe-hybrid-cache/policy_sim.py` – replays a routing trace (`LLAMA_MOE_CACHE_TRACE`) through cache policies.
- `tools/moe-hybrid-cache/prompts/` – the prompt sets (English chat, code, Thai, math).
- `tools/moe-hybrid-cache/results/` – raw benchmark results (JSON and logs) for the tables above.
- The patch itself is attached to the [release](https://github.com/simonyos/llama.cpp/releases).

Results from your own hardware are welcome: open an issue on this fork with your GPU, PCIe link, RAM, model and the `cache_bench.py` output.
