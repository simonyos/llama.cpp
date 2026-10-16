#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
// in hybrid mode (n_slots > 0) the missing experts run on the CPU instead, and are copied to the cache in the background
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // n_slots > 0 selects the hybrid mode with n_slots experts on one GPU, at most n_admit of them are copied at the same time (0 = never)
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            int32_t n_slots = 0, int32_t n_admit = 0);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    // in hybrid mode the expert biases are cached too
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // hybrid mode: give the experts selected for layer il to the device or the CPU, nullptr if the layer is not cached for n_tokens tokens
    // I32 [n_expert_used, n_tokens, 2]: the slots for the device, then the experts for the CPU, -1 (or a zero slot of the device) where the other side computes the expert
    ggml_tensor * build_route(ggml_context * ctx, ggml_tensor * selected_experts, int32_t il) const;

    // hybrid mode, one token: the device looks up the cached experts in this host slot map, nullptr if not available for layer il
    // its copy callback starts the readback of ids and cur, the copy callback of get_async_cpu_out(il) runs the misses on the CPU and uploads their output
    ggml_tensor * get_async_slot_map(int32_t il) const;
    ggml_tensor * get_async_cpu_out(int32_t il) const;

    // the tensors of the current graph: ids and cur on the device, and the CPU sub-graph of the misses (not in the graph)
    void set_async_graph(int32_t il, ggml_tensor * ids, ggml_tensor * cur, ggml_tensor * sub_cur, ggml_tensor * sub_ids, ggml_tensor * sub_out) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
