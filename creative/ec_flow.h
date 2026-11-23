// ec_flow.h — Compiled Content-Addressed ComfyUI-Compatible DAG Engine ("Eureka Flow")
// Parses standard ComfyUI prompt JSON workflows into a statically scheduled,
// VRAM-liveness-planned, content-addressed native execution graph. Zero Python.

#pragma once

#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_FLOW_MAX_NODES   128
#define EC_FLOW_MAX_INPUTS  24
#define EC_FLOW_MAX_CACHE   256

typedef enum {
  EC_VAL_NONE   = 0,
  EC_VAL_LINK   = 1,  // ["node_id", slot_idx]
  EC_VAL_STRING = 2,
  EC_VAL_NUMBER = 3,
} ec_val_kind_t;

typedef struct {
  char          key[48];
  ec_val_kind_t kind;
  char          str_val[1024]; // or link target node id
  double        num_val;
  int32_t       link_slot;
  int32_t       link_node_idx; // resolved during compile (-1 if scalar)
} ec_flow_input_t;

typedef struct {
  char            id[32];
  char            class_type[64];
  ec_flow_input_t inputs[EC_FLOW_MAX_INPUTS];
  int32_t         n_inputs;
  uint64_t        merkle_hash;      // content-addressed execution identity (canonical key-sorted)
  char            merkle_cid[65];   // canonical 64-hex SHA-256 node identity (Cowrie SHA-NI)
  int32_t         last_consumer;    // topo step after which this output can be evicted from VRAM
} ec_flow_node_t;

typedef struct {
  uint64_t hash;
  uint64_t result_tag;   // payload checksum / handle id
  void    *payload;      // owned or referenced typed artifact
  size_t   payload_bytes;
} ec_flow_cache_entry_t;

typedef struct ec_flow_graph ec_flow_graph_t;

// Node executor callback: executes node using upstream cached outputs, writes result_tag.
typedef int (*ec_node_exec_fn)(
    const ec_flow_graph_t *g, int32_t node_idx,
    const ec_flow_cache_entry_t *upstream_vals[], int32_t n_upstream,
    ec_flow_cache_entry_t *out_entry, void *user_ctx);

struct ec_flow_graph {
  ec_flow_node_t        nodes[EC_FLOW_MAX_NODES];
  int32_t               n_nodes;
  int32_t               topo_order[EC_FLOW_MAX_NODES];
  int32_t               compiled;
  ec_flow_cache_entry_t cache[EC_FLOW_MAX_CACHE];
  int32_t               n_cache;
  // Telemetry from most recent execute()
  int32_t               last_executed;
  int32_t               last_cache_hits;
  int32_t               last_evictions;
};

void ec_flow_init(ec_flow_graph_t *g);
void ec_flow_free(ec_flow_graph_t *g);

// Parses ComfyUI prompt JSON: { "1": { "class_type": "...", "inputs": { ... } }, ... }
int ec_flow_parse_comfy_json(ec_flow_graph_t *g, const char *json, char *err, size_t err_cap);

// Mutates a scalar string or numeric input of a node (for testing surgical dirty invalidation)
int ec_flow_set_input_num(ec_flow_graph_t *g, const char *node_id, const char *key, double val);
int ec_flow_set_input_str(ec_flow_graph_t *g, const char *node_id, const char *key, const char *val);

// Topological sort + cycle check + static VRAM liveness planner + Merkle hash computation
int ec_flow_compile(ec_flow_graph_t *g, char *err, size_t err_cap);

// Live per-node DAG execution hook ("cached", "start", "done"); return non-zero to cancel
typedef int (*ec_flow_node_event_fn)(
    const ec_flow_graph_t *g, int32_t step, int32_t total_steps, int32_t node_idx,
    const char *status, const ec_flow_cache_entry_t *entry, double node_ms, void *user_ctx);

// Executes only cache-missed (dirty) nodes in topological order; skips content-addressed hits
int ec_flow_execute(ec_flow_graph_t *g, ec_node_exec_fn exec_fn, void *user_ctx, char *err, size_t err_cap);
int ec_flow_execute_ex(
    ec_flow_graph_t *g, ec_node_exec_fn exec_fn, ec_flow_node_event_fn event_fn,
    void *user_ctx, char *err, size_t err_cap);
void ec_flow_clear_cache(ec_flow_graph_t *g);

// Injects a verified persistent cache payload (e.g. from Cowrie StatePage on disk) into RAM cache.
int ec_flow_preload_cache_entry(
    ec_flow_graph_t *g, uint64_t hash, uint64_t result_tag,
    const void *payload, size_t payload_bytes);
int ec_flow_cache_has(const ec_flow_graph_t *g, uint64_t hash);

// Helper lookups
const ec_flow_input_t *ec_flow_get_input(const ec_flow_node_t *n, const char *key);

#ifdef __cplusplus
}
#endif
