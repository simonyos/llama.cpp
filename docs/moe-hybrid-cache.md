# MoE expert cache for llama.cpp (VRAM + CPU hybrid)

I wanted to run MoE models that don't fit in my 8 GB card (RTX 3060 Ti) without them crawling. This branch keeps all the experts in RAM and caches the ones that get used a lot in VRAM. On my PC it ends up about **2x faster** than the usual `--n-cpu-moe` split.

Why it works: MoE routing is pretty sticky. Around half the experts a token uses were also used by the token right before it. So:
- **hits** run on the GPU from the cache
- **misses** run on the CPU straight from RAM, at the same time as the GPU does the hits
- missed experts get copied into the cache in the background, so the next tokens hit

> **Heads up:** I didn't write this code myself. I had the idea, ran all the tests on my own PC and decided what to try next. The code, benchmarks and analysis were written by Claude (Anthropic's AI coding agent) while I steered it. The commits have a `Co-Authored-By: Claude` line.
>
> It's experimental: single GPU, CUDA only, tested on one machine. Not proposed upstream. Please read the code before you rely on it.

## Results

Decode speed in tok/s. Greedy, 4 prompt sets (English chat, code, Thai, math), 8 prompts each, 256 tokens per prompt, `-ngl 99 -c 8192 -fa on -t 6`:

| Model | Stock `--n-cpu-moe` (best that fits) | Upstream `--moe-cache-mib` | **This branch** | vs stock |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 | 43.6 (`--n-cpu-moe 11`) | 64.9 (5600 MiB) | **87.1** (448 slots) | **2.00x** |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M | 37.3 (`--n-cpu-moe 33`) | 56.9 (5450 MiB) | **75.9** (1984 slots) | **2.03x** |

Per prompt set (English / code / Thai / math):

| Model | Stock | Upstream `--moe-cache-mib` | This branch | Cache hit rate |
|---|---|---|---|---|
| gpt-oss-20b | 43.6 / 43.6 / 43.5 / 43.5 | 74.6 / 58.5 / 70.4 / 59.2 | **94.1 / 81.7 / 90.9 / 82.9** | .970 / .934 / .963 / .937 |
| Qwen3-30B-A3B | 37.3 / 37.3 / 37.2 / 37.3 | 66.0 / 59.0 / 69.1 / 42.2 | **83.9 / 77.1 / 86.8 / 61.5** | .910 / .898 / .928 / .821 |

A few notes:
- VRAM use is about the same as stock (7.2–7.7 GB out of 8).
- llama.cpp already has a cache, `--moe-cache-mib` (#29887). That one uploads every missed expert over PCIe before the GPU uses it (hit rate was 0.948 on gpt-oss and 0.890 on Qwen3 here).
  This branch just lets the CPU compute the misses instead. That came out about 1.34x faster than the upstream cache on both models.
- Prefill on short chat prompts (20–90 tokens) got a bit faster too: gpt-oss 208 → 263 tok/s, Qwen3 74 → 99 tok/s.
- The GPU was capped at 150 W for every run (more on that under Limitations).
- The 2x is against the best `--n-cpu-moe` split I could fit, not against all experts on the CPU.
  If you compare against `--cpu-moe` instead, gpt-oss would look like ~3.5x (24.9 tok/s on a stock build). Worth keeping in mind when you compare with other numbers out there.

### My setup

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

- Set `CMAKE_CUDA_ARCHITECTURES` for your card: `86-real` is RTX 30xx, `89-real` is RTX 40xx. I've only tested `86`.
- The branch is upstream `ceefd16` plus the patch. If you'd rather patch your own checkout, grab `llama-moe-hybrid-cache.patch` from the [release](https://github.com/simonyos/llama.cpp/releases) and run `git checkout ceefd16 && git apply llama-moe-hybrid-cache.patch`.
- I built with the CUDA 13.4 toolkit from pip wheels (with 13.0 runtime headers), which needed `-DCMAKE_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`. A normal CUDA install shouldn't need that.

## Run

```bash
# gpt-oss-20b
./build/bin/llama-server -m gpt-oss-20b-MXFP4.gguf -ngl 99 -c 8192 -fa on -t 6 \
    --cpu-moe --load-mode none --moe-cache-slots 448 --moe-cache-admit 8

# Qwen3-30B-A3B
./build/bin/llama-server -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -ngl 99 -c 8192 -fa on -t 6 \
    --cpu-moe --load-mode none --moe-cache-slots 1984 --moe-cache-admit 16
```

- `--cpu-moe --load-mode none` puts all the experts in pinned RAM. You need this, because the background copies read from pinned memory.
- `--moe-cache-slots N` is how many experts the VRAM cache holds. Go as high as you can while leaving ~0.5 GB of VRAM free. One expert is 12.65 MiB for gpt-oss-20b and 2.5–2.9 MiB for Qwen3-30B-A3B Q4_K_M.
- `--moe-cache-admit K` is how many experts can be copying at once. 8–16 was fine for me, more didn't help.
- `--moe-cache-admit 0` gives a static cache that holds exactly what `--n-cpu-moe` would put on the GPU. Handy for checking the output is right.

## Is the output still correct?

| Check | gpt-oss-20b | Qwen3-30B-A3B |
|---|---|---|
| Static cache (`--moe-cache-admit 0`) vs stock `--n-cpu-moe`, greedy tokens identical | 32/32 prompts | 32/32 prompts |
| Static cache, teacher-forced KL divergence / PPL vs stock | KLD ≈ 0, PPL 2.9027 = 2.9027 | KLD ≈ 0, PPL 1.8222 = 1.8222 |
| Dynamic cache, KL divergence / top-1 agreement / PPL vs stock | KLD 0.0016, top-1 98.5%, PPL 2.897 vs 2.903 | KLD 0.0018, top-1 99.0%, PPL 1.825 vs 1.822 |
| Stock placement change as a control (early layers moved to the GPU with `-ot`) | KLD 0.0011, top-1 99.2% | KLD 0.0023, top-1 99.4% |
| VRAM slot contents vs RAM, byte by byte (`LLAMA_MOE_CACHE_VERIFY`) | 4 checks × 416 slots, 0 differ | 4 checks × ~1,980 slots, 0 differ |

- The static cache runs every expert on the same device as stock, and the output is bit-identical. So the routing, skipping and merging are right.
- With the dynamic cache an expert can run on the GPU or the CPU depending on what's cached. CPU and GPU round a little differently, so greedy output drifts from stock after ~100 tokens on average.
  Changing `--n-cpu-moe` on stock does the same thing, and the KL divergence is in the same range. Perplexity isn't worse.
- KLD and PPL are from `llama-perplexity --kl-divergence`, 1,024 scored tokens with `-ub 1`, so it actually goes through the decode path.

## How it works

- **Cache:** one bank per expert tensor type in VRAM, holding N expert slots. Eviction is a segmented LRU: experts that got hit again are protected from ones that were only used once.
  It starts with the experts of the last layers, same as `--n-cpu-moe`.
- **Each layer, each token** (no CPU split in the graph):
  1. The router runs on the GPU. A scheduler copy callback starts reading back the chosen expert ids and the layer input, and the GPU keeps going.
  2. The GPU looks the ids up in a slot map. Hits run from the cache; misses get id −1, which the CUDA kernels skip.
  3. A second callback waits for the readback and runs the misses on the CPU with the same ggml ops as stock `--n-cpu-moe`, while the GPU does the hits. Then it uploads the CPU result and the GPU adds both halves.
- **Admission:** a miss gets copied into the least valuable slot right away, by a copy kernel on a second CUDA stream that reads pinned RAM.
  A DMA copy (`cudaMemcpyAsync`) holds the copy engine for ~2 ms and stalls the small uploads every layer needs, so the patch doesn't use it.
  The copy waits for the GPU work already queued, and a slot is only used once an event says the copy is done.
- **Batches of 2–31 tokens** use a CPU split in the graph (misses get a zero slot for batches of 5+). **Batches of 32+** (prefill) use upstream's path, which copies cached experts inside VRAM.
- **ggml changes:**
  - negative expert ids are skipped in `mul_mat_id` and `add_id` (CPU and CUDA mmvq)
  - new `ggml_backend_event_query`
  - new `ggml_backend_cuda_set_tensor_async_kernel` for kernel copies from pinned memory
  - the scheduler calls the copy callback before it syncs the backend

Rough cost model from my PC, ms per token, with misses per token from the cache stats:

| | Fixed part | Per miss | Fixed part with all experts in VRAM (extrapolated) |
|---|---|---|---|
| gpt-oss-20b, CPU split per layer (PCIe x8, 200 W) | 9.8 | 0.52 | 8.6 |
| gpt-oss-20b, this branch | 9.3 | 0.45 | 8.6 |
| Qwen3-30B-A3B, CPU split per layer | 11.4 | 0.11 | 7.0 |
| Qwen3-30B-A3B, this branch | 8.5 | 0.11 | 7.0 |

The async path got rid of most of the fixed per-layer cost. What's left is mostly the CPU reading the missed experts from RAM (~25 GB/s).

## Other work on this

Caching MoE experts in VRAM is a pretty active topic in llama.cpp right now (October 2026):

| | Where misses run | How experts reach VRAM | Slots | Per-layer host sync | New expert usable |
|---|---|---|---|---|---|
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) `--moe-cache-mib` (merged Oct 7) | GPU, after an upload | async copy on the compute stream, on the critical path | pooled per device group, LRU | yes, reads ids back to plan uploads | at once, after the upload |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) (draft) | CPU, from host RAM | worker thread, `ggml_backend_tensor_set` (DMA), at most 2 per layer and step | fixed per layer, LRU, starts empty | yes, a CPU split per layer | from the next decode step |
| [#29949](https://github.com/ggml-org/llama.cpp/issues/29949) (issue + branch `lru-upstream-v3`; CUDA and HIP, multi-GPU incl. `-sm tensor`, MTP) | GPU, after a fill | in-graph fill kernel that reads pinned RAM, on the critical path | fixed per layer, LRU decided on the GPU | none | at once, after the fill |
| [arXiv 2512.16473](https://arxiv.org/abs/2512.16473) (ASP-DAC 2026) | CPU | (research code, not llama.cpp) | GPU expert cache | – | – |
| **This branch** | **CPU, at the same time as the GPU hits** | **background copy kernel that reads pinned RAM, on a second stream** | **one pool per expert type, segmented LRU, starts like `--n-cpu-moe`** | **readback is async; waits only for the CPU misses** | **as soon as its copy event completes (checked every layer)** |

To be clear, the core idea (GPU hits + CPU misses + background refill) isn't new. The #29887 thread suggests it as future work, #27861 implements it, and arXiv 2512.16473 describes it.
Fiddler (2024) and HybriMoE (2025) also run experts on the CPU next to a GPU, with static placement and with dynamic scheduling plus caching.
Earlier llama.cpp work on this includes [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) (closed as completed; it confirmed strong temporal locality), the RFC [#28248](https://github.com/ggml-org/llama.cpp/issues/28248), and the closed PRs [#21609](https://github.com/ggml-org/llama.cpp/pull/21609), [#21614](https://github.com/ggml-org/llama.cpp/pull/21614), [#21620](https://github.com/ggml-org/llama.cpp/pull/21620) and [#24524](https://github.com/ggml-org/llama.cpp/pull/24524).

What's different here:

1. **Numbers on why the refill shouldn't use the copy engine.** Reading pinned memory from a kernel instead of DMA isn't new, #29949's `LRU_FILL` already fills its pool that way.
   What I can add is a measurement of why it matters when the refill runs on a side stream: on my GPU one expert-sized `cudaMemcpyAsync` held up a small synchronous upload for 1989 µs, vs 36 µs with a kernel copy.
   The scheduler needs uploads like that every layer, so a DMA refill eats about as much as it saves (52 tok/s with DMA vs 77 tok/s with kernel copies, gpt-oss-20b at PCIe x8).
   #27861 refills through `ggml_backend_tensor_set`, which is the DMA path. That would fit its author's note that unthrottled uploads "eat the entire win", and its limit of 2 uploads per layer and step.
2. **No CPU split per layer.** Two copy callbacks per layer replace it, so layers with no misses cost very little, and the CPU work on misses overlaps the GPU work on hits. It needed one scheduler change: the callback now runs before the backend sync.
3. **An exactness check.** With `--moe-cache-admit 0` the cache holds the same experts as `--n-cpu-moe` and greedy output is bit-identical to stock (32/32 prompts on both models). Dynamic runs are compared against stock placement changes with KL divergence.
4. **It sits on top of master's cache.** It extends the `llama_moe_cache` from #29887, so it applies to current master. #27861 and #29949 were written before #29887 and conflict with it (see the #29949 thread).

Which approach is faster really depends on the machine. Computing a miss on the CPU reads the expert once from RAM; filling it on the GPU first means moving it over PCIe.
On PCIe 3.0 a gpt-oss-20b expert (12.65 MiB) takes ~1.0 ms to move at 13.4 GB/s vs ~0.33–0.45 ms to compute on my CPU, so CPU misses win here.
On PCIe 4.0/5.0 the two costs are close, and a design that keeps everything on the GPU (#29949) also skips the per-layer host round trip I still have (~1.5 ms/token on Qwen3-30B-A3B), so it might be faster there.

Same-machine runs (Qwen3-30B-A3B, 4 prompts × 4 sets × 256 tokens unless noted, 150 W, PCIe x16):

| Build | tok/s (en / code / th / math) | overall |
|---|---|---|
| stock `--n-cpu-moe 33` | 37.3 / 37.3 / 37.2 / 37.3 | 37.3 |
| #29887 `--cpu-moe --moe-cache-mib 5450` (8 prompts per set) | 66.0 / 59.0 / 69.1 / 42.2 | 56.9 |
| #27861 at `bccbacd`, `--cpu-moe -lm none --moe-expert-cache 41` (1968 slots), 2 inserts | 68.0 / 65.0 / 69.9 / 50.2 | 62.2 |
| #27861, same, 8 inserts | 64.0 / 29.0 / 18.4 / 18.2 | 25.1 |
| stock `--n-cpu-moe 33` built from #29949's base (8 prompts per set) | 37.1 / 37.2 / 37.1 / 37.2 | 37.1 |
| #29949 at `094ea24`, `-ncmoe 48 --load-mode none --moe-lru-slots 41` (1968 slots, 7.5 GB VRAM, 8 prompts per set) | 62.8 / 58.1 / 67.0 / 42.8 | 56.0 |
| #29949, `--moe-lru-slots 42` (2016 slots, 7.7 GB VRAM, 8 prompts per set) | 64.5 / 59.3 / 68.5 / 43.9 | 57.3 |
| **this branch**, 1984 slots, admit 16 | **84.0 / 78.3 / 86.4 / 62.0** | **76.4** |
| **this branch**, same (8 prompts per set, 7.6 GB VRAM) | **83.9 / 77.1 / 86.8 / 61.5** | **75.9** |

- #27861 with 8 uploads per step kept getting slower the longer it ran, which is what a saturated copy engine would look like. I didn't profile it to prove that.
  That run also ended with a GPU copy-engine read fault (NVIDIA Xid 31) when the server shut down, so I stopped testing it there.
- #27861 only supports the SILU gate/up path without biases, so it couldn't run gpt-oss-20b.
- #29949 reports 9.7 → 37–50 t/s (3.5–4x) on a Radeon 7900 XTX and 47.6 t/s on 2x RTX 3090, both on PCIe 4.0 x16 (~26 GB/s), with a bigger model (Qwen3.8-Flash-Next).
  Its baseline is all experts on the CPU (`-ncmoe 48`), not the best `--n-cpu-moe` split. Against #29887 on the same machine it reports 31.2 vs 20.1 t/s (~1.55x).
  On my PC (PCIe 3.0 x16, 13.4 GB/s) I built its branch at `094ea24` and ran the same prompts. It got 56.0–57.3 tok/s on Qwen3-30B-A3B, about the same as #29887 (56.9), and this branch got 75.9 (~1.33x) with less VRAM.
  Stock speed on #29949's base (37.1) matches mine, so the base commit isn't the reason. On the author's PCIe 4.0 machines the ranking could be different.
- #29949 stops on an assertion with gpt-oss-20b (`GGML_ASSERT` in `llama-graph.cpp`, biased experts aren't supported), so I could only compare on Qwen3-30B-A3B.
- The SpecMD study ([arXiv 2602.03921](https://arxiv.org/abs/2602.03921)) argues that LRU/LFU locality assumptions don't fit MoE. On my real traces decayed LFU was worse than LRU, and segmented LRU was a little better.

## Limitations

- Experimental, not proposed upstream. Single GPU, quantized experts, CUDA only, only tested on Ampere (sm_86).
- Only tested on one machine with PCIe 3.0 x16. On PCIe 4.0/5.0 the trade-off vs GPU-side fill designs might change (see above).
- `--load-mode none` pins every expert in RAM (9.5 GiB for gpt-oss-20b, 16.4 GiB for Qwen3-30B-A3B), so your RAM has to hold the whole model plus the OS.
- VRAM is tight: with 448 slots, gpt-oss-20b leaves ~0.3 GB free, and a raw (non-chat) prompt of 5–31 tokens then fails to create the cuBLAS handle. Chat prompts are longer and work fine. Use 416 slots if you send short raw prompts.
- Prefill isn't the target. Prompts of 32+ tokens use upstream's path, which is faster than stock but still copies missing experts over PCIe.
- Gains depend on the workload. Math has the lowest hit rate on Qwen3 (0.82), and the cache helps least there.
- My PC reset twice under the combined CPU + GPU + RAM load with the GPU at its default 200 W limit, probably the power supply. All numbers here are at 150 W, which didn't change stock speed or outputs.

## Debug switches (env vars)

| Variable | Effect |
|---|---|
| `LLAMA_MOE_CACHE_SYNC=1` | use the older path with one CPU split per layer instead of the async callbacks |
| `LLAMA_MOE_CACHE_SLRU=f` | protected fraction of the segmented LRU (default 0.7, `0` = plain LRU) |
| `LLAMA_MOE_CACHE_SPLIT=last` | with several expert types, split slots by the last layers instead of by layer count |
| `LLAMA_MOE_CACHE_VERIFY=N` | every N tokens, compare all ready VRAM slots with RAM and log the result |
| `LLAMA_MOE_CACHE_TRACE=file` | append the selected experts of each layer and token to a binary file |

## Files

In this branch:
- `tools/moe-hybrid-cache/cache_bench.py` – the benchmark: starts `llama-server` per config and runs the four prompt sets in `prompts/`; `summarize.py` prints the tables.
  Example: `MODEL=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf python tools/moe-hybrid-cache/cache_bench.py out.json build/bin "cache:--cpu-moe --load-mode none --moe-cache-slots 1984 --moe-cache-admit 16"`
- `tools/moe-hybrid-cache/compare_outputs.py` – compares greedy outputs of two configs token by token
- `tools/moe-hybrid-cache/policy_sim.py` – replays a routing trace (`LLAMA_MOE_CACHE_TRACE`) through cache policies
- `tools/moe-hybrid-cache/prompts/` – the prompt sets (English chat, code, Thai, math)
- `tools/moe-hybrid-cache/results/` – raw results (JSON and logs) behind the tables above
- the patch itself is attached to the [release](https://github.com/simonyos/llama.cpp/releases)

## Tried it?

I'd really like to see numbers from other hardware, especially PCIe 4.0/5.0 and other cards. [Open an issue](https://github.com/simonyos/llama.cpp/issues/new?template=001-moe-cache-results.yml) with your GPU, PCIe link, RAM, model and the tok/s you got (stock vs this branch). Bugs and "it didn't work on my setup" reports are welcome too.
