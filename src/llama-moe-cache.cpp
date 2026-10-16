#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // hybrid mode, segmented LRU if prot_cap > 0: the list above is the probation segment, a slot that is hit there moves to the protected segment
    // so experts used once do not push out the experts that are used again
    int32_t prot_cap = 0;
    std::vector<int8_t> seg; // [n_slots] 1 if in the protected segment
    int32_t p_head = -1;
    int32_t p_tail = -1;
    int32_t p_size = 0;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
        seg.assign(n_slots, 0);
    }

    void unlink(int32_t s) {
        int32_t & h = seg[s] ? p_head : head;
        int32_t & t = seg[s] ? p_tail : tail;
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            h = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            t = prev[s];
        }
        p_size -= seg[s];
    }

    void push_tail(int8_t sg, int32_t s) {
        int32_t & h = sg ? p_head : head;
        int32_t & t = sg ? p_tail : tail;
        seg[s]  = sg;
        prev[s] = t;
        next[s] = -1;
        if (t >= 0) {
            next[t] = s;
        } else {
            h = s;
        }
        t = s;
        p_size += sg;
    }

    // hybrid mode: slot s was hit
    void hit(int32_t s) {
        if (prot_cap == 0) {
            touch(s);
            return;
        }
        unlink(s);
        push_tail(1, s);
        if (p_size > prot_cap) {
            const int32_t d = p_head;
            unlink(d);
            push_tail(0, d);
        }
    }

    // hybrid mode: slot s got a new expert
    void fill_slot(int32_t s) {
        if (prot_cap == 0) {
            touch(s);
            return;
        }
        unlink(s);
        push_tail(0, s);
    }

    int32_t victim() const {
        return prot_cap == 0 || head >= 0 ? head : p_head;
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down, then their biases if requested
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer, bool biases) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    if (biases && !res.empty()) {
        for (ggml_tensor * t : { layer.ffn_gate_up_exps_b, layer.ffn_gate_exps_b, layer.ffn_up_exps_b, layer.ffn_down_exps_b }) {
            if (t != nullptr) {
                res.push_back(t);
            }
        }
    }
    return res;
}

// bytes of one expert, the experts are the last dimension
static size_t llama_moe_cache_expert_size(const ggml_tensor * t) {
    return t->nb[ggml_n_dims(t) - 1];
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
        std::vector<bool> ready; // [n_slots] hybrid: the slot holds its expert, false while it is copied
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // hybrid mode: the hits run on the device, the misses run on the CPU and are copied to the cache on a second backend
    // CUDA skips negative expert ids for small batches only, larger batches give the misses zero slots after the slots of the banks
    // MMQ needs distinct experts per token, so the miss at position i of a token gets zero slot i
    // from 32 tokens the scheduler runs the host experts on the device (op offload)
    static constexpr int64_t max_batch_skip   = 4;
    static constexpr int64_t max_batch_hybrid = 31;
    static constexpr int64_t log_every = 64;       // tokens

    struct route_ctx {
        impl  * cache;
        int32_t il;
    };

    struct load {
        int32_t ig;
        int32_t slot;
        ggml_backend_event_t event;
    };

    struct hybrid_stats {
        int64_t tokens     = 0;
        int64_t hits       = 0;
        int64_t misses     = 0;
        int64_t swaps      = 0;
        int64_t cpu_layers = 0; // layers of a token with at least one miss
        int64_t t_admit_us = 0; // time spent to start the copies
        int64_t reuse      = 0; // experts also selected by the previous token in the same layer
        int64_t reuse_n    = 0;
    };

    bool    hybrid  = false;
    int32_t n_admit = 0;
    int32_t n_zero  = 0; // zero slots after the slots of each bank

    // copies with the SMs if the backend can, a DMA copy would delay the small copies of the scheduler until it ends
    using set_tensor_async_kernel_t = bool (*)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t);
    set_tensor_async_kernel_t set_tensor_async_kernel = nullptr;
    int32_t il_first = -1; // first cached layer, counts the tokens

    ggml_backend_ptr copy_backend;
    std::deque<load> loads; // in the order of the copies
    std::vector<ggml_backend_event_t> free_events;
    std::vector<route_ctx> route_ctxs;              // [n_layer]
    std::vector<std::vector<int32_t>> slot_maps;    // [n_layer][n_expert]
    std::vector<int32_t> hit_slots;
    std::vector<int32_t> missed;
    std::vector<std::vector<int32_t>> prev_sel;     // [n_layer] experts of the previous token
    std::vector<int64_t> reuse_layer;               // [n_layer]
    std::vector<int64_t> reuse_n_layer;             // [n_layer]
    hybrid_stats hst;
    int64_t next_log = log_every;

    // hybrid mode, one token: two copy callbacks per layer instead of a CPU split
    // the first one starts the readback of the selected experts and of the layer input, without waiting for the device
    // the second one runs the misses on the CPU while the device runs the hits, then uploads the CPU output
    struct async_layer {
        ggml_tensor * slot_map = nullptr; // host weight I32 [1, n_expert], the device looks up the slots in it
        ggml_tensor * cpu_out  = nullptr; // host weight F32 [n_embd, n_expert_used, 1], the device adds it to its output

        // tensors of the current graph
        ggml_tensor * ids     = nullptr; // device, I32 [n_expert_used]
        ggml_tensor * cur     = nullptr; // device, F32 [n_embd, 1, 1]
        ggml_tensor * sub_cur = nullptr; // CPU sub-graph of the misses, not in the graph of the scheduler
        ggml_tensor * sub_ids = nullptr;
        ggml_tensor * sub_out = nullptr;

        ggml_context_ptr ctx_graph;
        ggml_cgraph * sub_graph = nullptr;
        ggml_gallocr_t galloc = nullptr;

        std::vector<int32_t> snapshot; // [n_expert] the slots that the device looks up
        int32_t * ids_host = nullptr;  // [n_expert_used] pinned
        bool dirty = true;             // sub_out may hold the output of earlier misses
    };

    bool    async_ok = false;
    int64_t n_embd   = 0;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<async_layer> async_layers;                            // [n_layer]
    std::unordered_map<const ggml_tensor *, int32_t> async_of;        // trigger -> il*2 + (0 = slot map, 1 = CPU output)
    ggml_context_ptr ctx_async;
    ggml_backend_buffer_ptr buf_async;
    ggml_backend_buffer_ptr buf_async_ids;
    ggml_backend_event_t ev_readback = nullptr;
    ggml_backend_event_t ev_main     = nullptr; // the copies of the admitted experts wait for the device work before them

    // debug: LLAMA_MOE_CACHE_VERIFY=N compares the ready slots with the host experts every N tokens
    int64_t verify_every = 0;
    int64_t next_verify  = 0;

    // debug: LLAMA_MOE_CACHE_TRACE=file appends the experts selected for each layer of each token (I32 il, then n_expert_used ids)
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            int32_t n_slots_hybrid, int32_t n_admit_hybrid) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        hybrid  = n_slots_hybrid > 0;
        n_admit = std::max(n_admit_hybrid, 0);
        if (const char * env = getenv("LLAMA_MOE_CACHE_VERIFY")) {
            verify_every = std::max(1, atoi(env));
            next_verify  = verify_every;
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_TRACE")) {
            trace = fopen(env, "ab");
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                backend_cpu = backends[i];
            }
        }
        n_embd = model.hparams.n_embd;
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il], hybrid);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }
        if (hybrid) {
            check_hybrid(model);
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        // the hybrid mode adds n_expert_used zero slots before it
        n_zero = hybrid ? n_expert_used : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(llama_moe_cache_expert_size(t)*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // hybrid mode: the slots go to the last layers first like --n-cpu-moe, so the cache starts with all experts of the last layers
        std::vector<int32_t> slots_hybrid(groups.size(), 0);
        if (hybrid) {
            std::vector<int32_t> group_of(model.layers.size(), -1);
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                for (int32_t il : groups[ig].layers) {
                    group_of[il] = (int32_t) ig;
                }
            }
            int32_t n_left = n_slots_hybrid;
            for (int32_t il = (int32_t) model.layers.size() - 1; il >= 0 && n_left > 0; --il) {
                if (group_of[il] >= 0) {
                    const int32_t n = std::min<int32_t>(n_left, groups[group_of[il]].ref[0]->ne[2]);
                    slots_hybrid[group_of[il]] += n;
                    n_left -= n;
                }
            }
            // with copies, each group gets slots in proportion to its layers instead (LLAMA_MOE_CACHE_SPLIT=last keeps the split above)
            const char * env_split = getenv("LLAMA_MOE_CACHE_SPLIT");
            if (n_admit > 0 && !(env_split && strcmp(env_split, "last") == 0) && groups.size() > 1) {
                size_t n_layers = 0;
                for (const group & g : groups) {
                    n_layers += g.layers.size();
                }
                int32_t n_given = 0;
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    slots_hybrid[ig] = (int32_t) ((int64_t) n_slots_hybrid*groups[ig].layers.size()/n_layers);
                    n_given += slots_hybrid[ig];
                }
                slots_hybrid.back() += n_slots_hybrid - n_given;
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    slots_hybrid[ig] = std::min<int32_t>(slots_hybrid[ig], groups[ig].layers.size()*groups[ig].ref[0]->ne[2]);
                }
            }
        }

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            if (hybrid) {
                g.n_slots = slots_hybrid[&g - groups.data()];
            }
            while (!hybrid && g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            if (hybrid) {
                // segmented LRU: a fraction of the slots is kept for experts that were hit again (LLAMA_MOE_CACHE_SLRU=f, 0 = plain LRU)
                const char * env_slru = getenv("LLAMA_MOE_CACHE_SLRU");
                const double frac = env_slru ? std::min(std::max(atof(env_slru), 0.0), 0.95) : 0.7;
                g.lru.prot_cap = (int32_t) (frac*g.n_slots);
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                // the biases of the hybrid mode are 2D
                ggml_tensor * bank = ggml_n_dims(t) == 2 ?
                    ggml_new_tensor_2d(ctx, t->type, t->ne[0], g.n_slots + n_zero + 1) :
                    ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(llama_moe_cache_expert_size(bank) == llama_moe_cache_expert_size(t));
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il], hybrid);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    // the hybrid mode also reads the zero slots
                    ggml_tensor * cached = hybrid ? bank : ggml_n_dims(bank) == 2 ?
                        ggml_view_2d(ctx, bank, bank->ne[0], g.n_slots, bank->nb[1], 0) :
                        ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                // the hybrid mode keeps its slot maps in host vectors, they are not copied by the scheduler
                if (hybrid) {
                    continue;
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            if (!hybrid) {
                buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                    t->buffer = buf_host.get();
                }
            }
        } else if (!hybrid) {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            init_hybrid();
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        if (buf_host) {
            ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    // the hybrid mode reads the experts from host memory with async copies and skips the misses on the device by their negative slots
    void check_hybrid(const llama_model & model) const {
        for (const group & g : groups) {
            if (g.id != groups[0].id) {
                throw std::runtime_error("--moe-cache-slots supports one GPU only");
            }
            ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[g.id].backend));
            for (int32_t il : g.layers) {
                for (const ggml_tensor * t : llama_moe_cache_layer_experts(model.layers[il], true)) {
                    if (ggml_n_dims(t) == 3 && !ggml_is_quantized(t->type)) {
                        throw std::runtime_error("--moe-cache-slots needs quantized experts");
                    }
                    if (buft_pinned == nullptr || ggml_backend_buffer_get_type(t->buffer) != buft_pinned) {
                        throw std::runtime_error("--moe-cache-slots needs the experts in pinned host memory, use --load-mode none");
                    }
                    if (model.hdmd.rot.count(t) > 0) {
                        throw std::runtime_error("--moe-cache-slots does not support Hadamard-folded experts");
                    }
                }
            }
        }
    }

    void init_hybrid() {
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[groups[0].id].backend);

        copy_backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!copy_backend) {
            throw std::runtime_error("failed to create the MoE cache copy backend");
        }
        set_tensor_async_kernel = (set_tensor_async_kernel_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_set_tensor_async_kernel");
        for (int32_t i = 0; i < n_admit; ++i) {
            ggml_backend_event_t ev = ggml_backend_event_new(dev);
            if (ev == nullptr) {
                throw std::runtime_error("--moe-cache-slots needs a device with events");
            }
            free_events.push_back(ev);
        }

        route_ctxs.resize(layers.size());
        slot_maps.resize(layers.size());
        prev_sel.resize(layers.size());
        reuse_layer.assign(layers.size(), 0);
        reuse_n_layer.assign(layers.size(), 0);
        for (size_t il = 0; il < layers.size(); ++il) {
            route_ctxs[il] = { this, (int32_t) il };
        }

        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n_expert = g.lru.n_expert;
            g.ready.assign(g.n_slots, true);
            for (int32_t il : g.layers) {
                slot_maps[il].assign(n_expert, -1);
                g.lru.slot_map[il] = slot_maps[il].data();
                il_first = il_first < 0 ? il : std::min(il_first, il);
            }

            // start with all experts of the last layers, like --n-cpu-moe, the empty slots are evicted first
            int32_t s = 0;
            for (auto it = g.layers.rbegin(); it != g.layers.rend() && s < g.n_slots; ++it) {
                for (int32_t e = 0; e < n_expert && s < g.n_slots; ++e, ++s) {
                    upload(g, *it, e, s, nullptr);
                    g.lru.key_of[s] = *it*n_expert + e;
                    g.lru.slot_map[*it][e] = s;
                }
            }
            for (int32_t i = 0; i < s; ++i) {
                g.lru.fill_slot(i);
            }
        }

        init_async(dev);

        LLAMA_LOG_INFO("%s: hybrid mode: misses run on the CPU%s, up to %d experts are copied at the same time%s\n", __func__,
            async_ok ? " next to the device" : "", n_admit, set_tensor_async_kernel ? " by a kernel" : "");
    }

    void init_async(ggml_backend_dev_t dev) {
        const char * env = getenv("LLAMA_MOE_CACHE_SYNC");
        ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(dev);
        if ((env && atoi(env) != 0) || backend_cpu == nullptr || buft_pinned == nullptr) {
            return;
        }
        ev_readback = ggml_backend_event_new(dev);
        ev_main     = ggml_backend_event_new(dev);
        if (ev_readback == nullptr || ev_main == nullptr) {
            return;
        }

        async_layers.resize(layers.size());
        size_t n_cached = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            n_cached += layers[il].ig >= 0 ? 1 : 0;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ 2*n_cached*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_async.reset(ggml_init(params));
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            async_layer & a = async_layers[il];
            a.slot_map = ggml_new_tensor_2d(ctx_async.get(), GGML_TYPE_I32, 1, groups[layers[il].ig].lru.n_expert);
            a.cpu_out  = ggml_new_tensor_3d(ctx_async.get(), GGML_TYPE_F32, n_embd, n_expert_used, 1);
            ggml_format_name(a.slot_map, "moe_cache.async_slot_map-%zu", il);
            ggml_format_name(a.cpu_out,  "moe_cache.async_cpu_out-%zu", il);
            a.snapshot.assign(a.slot_map->ne[1], -1);
            async_of[a.slot_map] = 2*il;
            async_of[a.cpu_out]  = 2*il + 1;
        }
        // as host weights, the scheduler starts a split at their users and copies them with the copy callback
        buf_async.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_async.get(), ggml_backend_cpu_buffer_type()));
        buf_async_ids.reset(ggml_backend_buft_alloc_buffer(buft_pinned, layers.size()*n_expert_used*sizeof(int32_t)));
        if (!buf_async || !buf_async_ids) {
            throw std::runtime_error("failed to allocate the MoE cache buffers");
        }
        ggml_backend_buffer_set_usage(buf_async.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t il = 0; il < layers.size(); ++il) {
            async_layers[il].ids_host = (int32_t *) ggml_backend_buffer_get_base(buf_async_ids.get()) + il*n_expert_used;
        }
        async_ok = true;
    }

    ggml_tensor * get_async_slot_map(int32_t il) const {
        if (!async_ok || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        return async_layers[il].slot_map;
    }

    ggml_tensor * get_async_cpu_out(int32_t il) const {
        return get_async_slot_map(il) ? async_layers[il].cpu_out : nullptr;
    }

    void set_async_graph(int32_t il, ggml_tensor * ids, ggml_tensor * cur, ggml_tensor * sub_cur, ggml_tensor * sub_ids, ggml_tensor * sub_out) {
        async_layer & a = async_layers[il];
        GGML_ASSERT(ggml_nelements(ids) == n_expert_used && ggml_nbytes(cur) == ggml_nbytes(sub_cur));
        GGML_ASSERT(ggml_are_same_shape(sub_out, a.cpu_out));
        a.ids     = ids;
        a.cur     = cur;
        a.sub_cur = sub_cur;
        a.sub_ids = sub_ids;
        a.sub_out = sub_out;

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_graph_overhead_custom(256, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        a.ctx_graph.reset(ggml_init(params));
        a.sub_graph = ggml_new_graph_custom(a.ctx_graph.get(), 256, false);
        ggml_set_input(sub_cur);
        ggml_set_input(sub_ids);
        ggml_set_output(sub_out);
        ggml_build_forward_expand(a.sub_graph, sub_out);

        // the CPU reads the readback and the device reads the output, so they are in pinned memory
        if (a.galloc == nullptr) {
            a.galloc = ggml_gallocr_new(ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[groups[layers[il].ig].id].backend)));
        }
        if (!ggml_gallocr_alloc_graph(a.galloc, a.sub_graph)) {
            throw std::runtime_error("failed to allocate the MoE cache CPU graph");
        }
        a.dirty = true;
    }

    // first callback of a layer: give the device the slots, start the readback for the CPU
    void async_readback(int32_t il, ggml_backend_t backend, ggml_tensor * dst) {
        async_layer & a = async_layers[il];
        group & g = groups[layers[il].ig];
        GGML_ASSERT(backend == devices[g.id].backend && a.ids != nullptr);

        poll();

        const int32_t * slots = g.lru.slot_map[il];
        std::copy(slots, slots + a.snapshot.size(), a.snapshot.begin());
        ggml_backend_tensor_set_async(backend, dst, a.snapshot.data(), 0, ggml_nbytes(dst));
        ggml_backend_tensor_get_async(backend, a.ids, a.ids_host, 0, ggml_nbytes(a.ids));
        ggml_backend_tensor_get_async(backend, a.cur, a.sub_cur->data, 0, ggml_nbytes(a.cur));
        ggml_backend_event_record(ev_readback, backend);
    }

    // second callback of a layer: the device runs the hits, the CPU runs the misses, then the device gets the CPU output
    void async_misses(int32_t il, ggml_backend_t backend, ggml_tensor * dst) {
        async_layer & a = async_layers[il];
        const int32_t ig = layers[il].ig;
        group & g = groups[ig];

        ggml_backend_event_synchronize(ev_readback);

        const int32_t n_expert = g.lru.n_expert;
        int32_t * cpu_ids = (int32_t *) a.sub_ids->data;
        hit_slots.clear();
        missed.clear();
        for (int64_t iu = 0; iu < n_expert_used; ++iu) {
            const int32_t e = a.ids_host[iu];
            GGML_ASSERT(e >= 0 && e < n_expert);
            const int32_t s = a.snapshot[e];
            if (s >= 0) {
                g.lru.hit(s);
                hit_slots.push_back(s);
                cpu_ids[iu] = -1;
            } else {
                missed.push_back(e);
                cpu_ids[iu] = e;
            }
        }
        count_reuse(il, a.ids_host, n_expert_used);

        if (!missed.empty()) {
            ggml_backend_graph_compute(backend_cpu, a.sub_graph);
            a.dirty = true;
        } else if (a.dirty) {
            memset(a.sub_out->data, 0, ggml_nbytes(a.sub_out));
            a.dirty = false;
        }

        // the copies wait for the device work so far, so that no kernel reads an evicted slot
        const int64_t t_start_us = ggml_time_us();
        int64_t n_swaps = 0;
        for (int32_t e : missed) {
            if ((int32_t) loads.size() >= n_admit) {
                break;
            }
            if (is_loading(ig, il*n_expert + e)) {
                continue;
            }
            if (n_swaps == 0) {
                ggml_backend_event_record(ev_main, backend);
                ggml_backend_event_wait(copy_backend.get(), ev_main);
            }
            if (!admit(ig, il, e)) {
                break;
            }
            n_swaps++;
        }
        hst.t_admit_us += ggml_time_us() - t_start_us;

        ggml_backend_tensor_set_async(backend, dst, a.sub_out->data, 0, ggml_nbytes(dst));

        count_token(il, 1, (int64_t) missed.size(), n_swaps);
    }

    // copy expert e of layer il to slot s, synchronous if backend is null
    void upload(group & g, int32_t il, int32_t e, int32_t s, ggml_backend_t backend) {
        const layer & l = layers[il];
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            const size_t size = llama_moe_cache_expert_size(w);
            const uint8_t * src = (const uint8_t *) w->data + e*size;
            if (backend) {
                if (!set_tensor_async_kernel || !set_tensor_async_kernel(backend, g.banks[ip], src, s*size, size)) {
                    ggml_backend_tensor_set_async(backend, g.banks[ip], src, s*size, size);
                }
            } else {
                ggml_backend_tensor_set(g.banks[ip], src, s*size, size);
            }
        }
    }

    // the copies end in order, a slot can be used when its copy is done
    void poll() {
        while (!loads.empty() && ggml_backend_event_query(loads.front().event)) {
            const load & ld = loads.front();
            group & g = groups[ld.ig];
            const int32_t key = g.lru.key_of[ld.slot];
            g.ready[ld.slot] = true;
            g.lru.slot_map[key / g.lru.n_expert][key % g.lru.n_expert] = ld.slot;
            free_events.push_back(ld.event);
            loads.pop_front();
        }
    }

    bool is_loading(int32_t ig, int32_t key) const {
        for (const load & ld : loads) {
            if (ld.ig == ig && groups[ig].lru.key_of[ld.slot] == key) {
                return true;
            }
        }
        return false;
    }

    // give the least recently used slot to expert e of layer il, the slot can be used when the copy is done
    bool admit(int32_t ig, int32_t il, int32_t e) {
        group & g = groups[ig];
        const int32_t s = g.lru.victim();
        if (!g.ready[s] || std::find(hit_slots.begin(), hit_slots.end(), s) != hit_slots.end()) {
            return false;
        }
        const int32_t n_expert = g.lru.n_expert;
        if (g.lru.key_of[s] >= 0) {
            g.lru.slot_map[g.lru.key_of[s] / n_expert][g.lru.key_of[s] % n_expert] = -1;
        }
        g.lru.key_of[s] = il*n_expert + e;
        g.lru.fill_slot(s);
        g.ready[s] = false;

        upload(g, il, e, s, copy_backend.get());

        ggml_backend_event_t ev = free_events.back();
        free_events.pop_back();
        ggml_backend_event_record(ev, copy_backend.get());
        loads.push_back({ ig, s, ev });
        return true;
    }

    static void route_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        const route_ctx * rc = (const route_ctx *) userdata;
        rc->cache->route(rc->il, dst->src[0], dst);
    }

    void route(int32_t il, const ggml_tensor * sel, ggml_tensor * dst) {
        const int32_t ig = layers[il].ig;
        group & g = groups[ig];

        const int32_t n_expert = g.lru.n_expert;
        const int64_t n_used   = sel->ne[0];
        const int64_t n_tokens = sel->ne[1];
        GGML_ASSERT(ggml_is_contiguous(dst) && dst->ne[0] == n_used && dst->ne[1] == n_tokens);

        poll();

        int32_t * dev_ids = (int32_t *) dst->data;
        int32_t * cpu_ids = dev_ids + n_used*n_tokens;
        const int32_t * slots = g.lru.slot_map[il];
        const bool skip = n_tokens <= max_batch_skip;

        if (n_tokens == 1) {
            std::vector<int32_t> sel_ids(n_used);
            for (int64_t iu = 0; iu < n_used; ++iu) {
                sel_ids[iu] = *(const int32_t *) ((const char *) sel->data + iu*sel->nb[0]);
            }
            count_reuse(il, sel_ids.data(), n_used);
        }

        hit_slots.clear();
        missed.clear();
        for (int64_t it = 0; it < n_tokens; ++it) {
            for (int64_t iu = 0; iu < n_used; ++iu) {
                const int32_t e = *(const int32_t *) ((const char *) sel->data + it*sel->nb[1] + iu*sel->nb[0]);
                GGML_ASSERT(e >= 0 && e < n_expert);
                const int64_t i = it*n_used + iu;
                const int32_t s = slots[e];
                if (s >= 0) {
                    g.lru.hit(s);
                    hit_slots.push_back(s);
                    dev_ids[i] = s;
                    cpu_ids[i] = -1;
                } else {
                    missed.push_back(e);
                    dev_ids[i] = skip ? -1 : g.n_slots + (int32_t) iu; // the zero slots are never filled
                    cpu_ids[i] = e;
                }
            }
        }

        // the misses run on the CPU for this token and are copied to the cache for the next ones
        const int64_t t_start_us = ggml_time_us();
        int64_t n_swaps = 0;
        for (int32_t e : missed) {
            if ((int32_t) loads.size() >= n_admit) {
                break;
            }
            if (is_loading(ig, il*n_expert + e)) {
                continue;
            }
            if (n_swaps == 0) {
                // the scheduler waits for the device before this CPU split, wait again so that no kernel reads the evicted slot
                ggml_backend_synchronize(devices[g.id].backend);
            }
            if (!admit(ig, il, e)) {
                break;
            }
            n_swaps++;
        }
        hst.t_admit_us += ggml_time_us() - t_start_us;

        count_token(il, n_tokens, (int64_t) missed.size(), n_swaps);
    }

    // count the experts that the previous token selected too, this is what makes the cache work
    void count_reuse(int32_t il, const int32_t * sel_ids, int64_t n_used) {
        if (trace) {
            fwrite(&il, sizeof(il), 1, trace);
            fwrite(sel_ids, sizeof(int32_t), n_used, trace);
        }
        std::vector<int32_t> & prev = prev_sel[il];
        for (int64_t iu = 0; iu < n_used; ++iu) {
            reuse_layer[il] += std::find(prev.begin(), prev.end(), sel_ids[iu]) != prev.end() ? 1 : 0;
        }
        if (!prev.empty()) {
            reuse_n_layer[il] += n_used;
        }
        prev.assign(sel_ids, sel_ids + n_used);
    }

    void count_token(int32_t il, int64_t n_tokens, int64_t n_missed, int64_t n_swaps) {
        if (il == il_first) {
            hst.tokens += n_tokens;
            if (hst.tokens >= next_log) {
                log_hybrid();
                next_log += log_every;
            }
            if (verify_every > 0 && hst.tokens >= next_verify) {
                verify();
                next_verify += verify_every;
            }
        }
        hst.hits       += hit_slots.size();
        hst.misses     += n_missed;
        hst.swaps      += n_swaps;
        hst.cpu_layers += n_missed > 0 ? 1 : 0;
    }

    // read back the ready slots and compare them with their host experts
    void verify() {
        ggml_backend_synchronize(devices[groups[0].id].backend);
        std::vector<uint8_t> buf;
        int64_t n_checked = 0;
        int64_t n_bad     = 0;
        for (group & g : groups) {
            for (int32_t s = 0; s < g.n_slots; ++s) {
                const int32_t key = g.lru.key_of[s];
                if (key < 0 || !g.ready[s]) {
                    continue;
                }
                const layer & l = layers[key / g.lru.n_expert];
                const int32_t e = key % g.lru.n_expert;
                bool ok = g.lru.slot_map[key / g.lru.n_expert][e] == s;
                for (size_t ip = 0; ip < l.experts.size() && ok; ++ip) {
                    const size_t size = llama_moe_cache_expert_size(l.experts[ip]);
                    buf.resize(size);
                    ggml_backend_tensor_get(g.banks[ip], buf.data(), s*size, size);
                    ok = memcmp(buf.data(), (const uint8_t *) l.experts[ip]->data + e*size, size) == 0;
                }
                n_checked++;
                n_bad += ok ? 0 : 1;
            }
            // the zero slots must stay zero
            for (ggml_tensor * bank : g.banks) {
                const size_t size = llama_moe_cache_expert_size(bank);
                buf.resize(size*n_zero);
                ggml_backend_tensor_get(bank, buf.data(), g.n_slots*size, size*n_zero);
                n_bad += std::all_of(buf.begin(), buf.end(), [](uint8_t x) { return x == 0; }) ? 0 : 1;
            }
        }
        LLAMA_LOG_WARN("llama_moe_cache: verify: tokens=%lld slots_checked=%lld bad=%lld in_flight=%zu\n",
            (long long) hst.tokens, (long long) n_checked, (long long) n_bad, loads.size());
    }

    void log_hybrid() const {
        int64_t reuse = 0;
        int64_t reuse_n = 0;
        for (size_t il = 0; il < reuse_layer.size(); ++il) {
            reuse   += reuse_layer[il];
            reuse_n += reuse_n_layer[il];
        }
        const int64_t n = hst.hits + hst.misses;
        LLAMA_LOG_INFO("llama_moe_cache: moe_cache_stats: tokens=%lld hits=%lld misses=%lld swaps=%lld cpu_layers=%lld hit_rate=%.4f admit_ms=%.1f reuse=%lld reuse_n=%lld\n",
            (long long) hst.tokens, (long long) hst.hits, (long long) hst.misses, (long long) hst.swaps, (long long) hst.cpu_layers,
            n > 0 ? (double) hst.hits/n : 0.0, hst.t_admit_us/1000.0, (long long) reuse, (long long) reuse_n);
    }

    void log_reuse() const {
        std::string s;
        for (size_t il = 0; il < reuse_layer.size(); ++il) {
            if (reuse_n_layer[il] > 0) {
                s += format(" %zu:%.2f", il, (double) reuse_layer[il]/reuse_n_layer[il]);
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache: experts reused from the previous token per layer:%s\n", s.c_str());
    }

    ggml_tensor * build_route(ggml_context * ctx, ggml_tensor * selected_experts, int32_t il) {
        if (!hybrid || route_ctxs.empty() || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const int64_t n_tokens = selected_experts->ne[1];
        if (n_tokens == 0 || n_tokens > max_batch_hybrid) {
            return nullptr;
        }
        ggml_tensor * args[] = { selected_experts };
        return ggml_custom_4d(ctx, GGML_TYPE_I32, selected_experts->ne[0], n_tokens, 2, 1, args, 1, route_op, 1, &route_ctxs[il]);
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
        if (copy_backend) {
            ggml_backend_synchronize(copy_backend.get());
        }
        for (const load & ld : loads) {
            free_events.push_back(ld.event);
        }
        for (ggml_backend_event_t ev : free_events) {
            ggml_backend_event_free(ev);
        }
        for (async_layer & a : async_layers) {
            ggml_gallocr_free(a.galloc);
        }
        if (trace) {
            fclose(trace);
        }
        for (ggml_backend_event_t ev : { ev_readback, ev_main }) {
            if (ev) {
                ggml_backend_event_free(ev);
            }
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (hybrid || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        if (const auto it_async = async_of.find(src); it_async != async_of.end()) {
            const int32_t il = it_async->second / 2;
            if (it_async->second % 2 == 0) {
                async_readback(il, backend, dst);
            } else {
                async_misses(il, backend, dst);
            }
            return true;
        }

        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (hybrid && hst.tokens > 0) {
            log_hybrid();
            log_reuse();
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        int32_t n_slots, int32_t n_admit) :
    pimpl(new impl(model, backends, bufts, size, n_slots, n_admit)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::build_route(ggml_context * ctx, ggml_tensor * selected_experts, int32_t il) const {
    return pimpl->build_route(ctx, selected_experts, il);
}

ggml_tensor * llama_moe_cache::get_async_slot_map(int32_t il) const {
    return pimpl->get_async_slot_map(il);
}

ggml_tensor * llama_moe_cache::get_async_cpu_out(int32_t il) const {
    return pimpl->get_async_cpu_out(il);
}

void llama_moe_cache::set_async_graph(int32_t il, ggml_tensor * ids, ggml_tensor * cur, ggml_tensor * sub_cur, ggml_tensor * sub_ids, ggml_tensor * sub_out) const {
    pimpl->set_async_graph(il, ids, cur, sub_cur, sub_ids, sub_out);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
