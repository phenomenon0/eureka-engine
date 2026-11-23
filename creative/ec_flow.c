// ec_flow.c — Compiled Content-Addressed ComfyUI-Compatible DAG Engine ("Eureka Flow")
// Zero Python, zero third-party JSON parsers.

#define _POSIX_C_SOURCE 200809L
#include "ec_flow.h"
#include "ec_cowrie.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double flow_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static void flow_err(char *err, size_t cap, const char *msg) {
  if (err && cap > 0) snprintf(err, cap, "%s", msg);
}

// 64-bit FNV-1a content-addressing hash
static uint64_t fnv1a_init(void) { return 14695981039346656037ULL; }
static uint64_t fnv1a_mix(uint64_t h, const void *data, size_t len) {
  const uint8_t *p = (const uint8_t *)data;
  for (size_t i = 0; i < len; i++) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}
static uint64_t fnv1a_str(uint64_t h, const char *s) {
  return fnv1a_mix(h, s, strlen(s) + 1);
}

void ec_flow_init(ec_flow_graph_t *g) {
  memset(g, 0, sizeof *g);
}

void ec_flow_free(ec_flow_graph_t *g) {
  if (!g) return;
  for (int32_t i = 0; i < g->n_cache; i++) {
    free(g->cache[i].payload);
    g->cache[i].payload = NULL;
  }
  memset(g, 0, sizeof *g);
}

// Minimal hardened JSON cursor for ComfyUI prompt format
typedef struct {
  const char *p;
  const char *end;
} fj_t;

static void fj_ws(fj_t *c) {
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}

static int fj_str(fj_t *c, char *out, size_t cap) {
  fj_ws(c);
  if (c->p >= c->end || *c->p != '"') return -1;
  c->p++;
  size_t n = 0;
  while (c->p < c->end && *c->p != '"') {
    char ch = *c->p++;
    if (ch == '\\') {
      if (c->p >= c->end) return -1;
      ch = *c->p++;
    }
    if (out && n + 1 < cap) out[n++] = ch;
  }
  if (c->p >= c->end || *c->p != '"') return -1;
  c->p++;
  if (out && cap > 0) out[n] = '\0';
  return 0;
}

static int fj_num(fj_t *c, double *out) {
  fj_ws(c);
  if (c->p >= c->end) return -1;
  char *ep = NULL;
  double v = strtod(c->p, &ep);
  if (ep == c->p || ep > c->end) return -1;
  c->p = ep;
  *out = v;
  return 0;
}

static int fj_skip(fj_t *c, int depth) {
  if (depth > 16) return -1;
  fj_ws(c);
  if (c->p >= c->end) return -1;
  if (*c->p == '"') return fj_str(c, NULL, 0);
  if (*c->p == '{') {
    c->p++;
    fj_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    while (c->p < c->end) {
      if (fj_str(c, NULL, 0)) return -1;
      fj_ws(c);
      if (c->p >= c->end || *c->p != ':') return -1;
      c->p++;
      if (fj_skip(c, depth + 1)) return -1;
      fj_ws(c);
      if (c->p < c->end && *c->p == ',') { c->p++; continue; }
      if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
      return -1;
    }
    return -1;
  }
  if (*c->p == '[') {
    c->p++;
    fj_ws(c);
    if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
    while (c->p < c->end) {
      if (fj_skip(c, depth + 1)) return -1;
      fj_ws(c);
      if (c->p < c->end && *c->p == ',') { c->p++; continue; }
      if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
      return -1;
    }
    return -1;
  }
  while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' && *c->p != ' ' && *c->p != '\n' && *c->p != '\r' && *c->p != '\t')
    c->p++;
  return 0;
}

int ec_flow_parse_comfy_json(ec_flow_graph_t *g, const char *json, char *err, size_t err_cap) {
  // Preserve existing cache across graph reloads so edits hit cache cleanly!
  g->n_nodes = 0;
  g->compiled = 0;
  fj_t c = { json, json + strlen(json) };
  fj_ws(&c);
  if (c.p >= c.end || *c.p != '{') { flow_err(err, err_cap, "flow: root must be JSON object"); return -1; }
  c.p++;
  fj_ws(&c);
  if (c.p < c.end && *c.p == '}') return 0;

  while (c.p < c.end) {
    if (g->n_nodes >= EC_FLOW_MAX_NODES) { flow_err(err, err_cap, "flow: max nodes exceeded"); return -1; }
    ec_flow_node_t *nd = &g->nodes[g->n_nodes];
    memset(nd, 0, sizeof *nd);
    if (fj_str(&c, nd->id, sizeof nd->id)) { flow_err(err, err_cap, "flow: bad node id"); return -1; }
    fj_ws(&c);
    if (c.p >= c.end || *c.p != ':') { flow_err(err, err_cap, "flow: expected :"); return -1; }
    c.p++;
    fj_ws(&c);
    if (c.p >= c.end || *c.p != '{') { flow_err(err, err_cap, "flow: expected node obj"); return -1; }
    c.p++;
    fj_ws(&c);
    while (c.p < c.end && *c.p != '}') {
      char fk[48];
      if (fj_str(&c, fk, sizeof fk)) { flow_err(err, err_cap, "flow: bad node field"); return -1; }
      fj_ws(&c);
      if (c.p >= c.end || *c.p != ':') { flow_err(err, err_cap, "flow: field :"); return -1; }
      c.p++;
      if (!strcmp(fk, "class_type")) {
        if (fj_str(&c, nd->class_type, sizeof nd->class_type)) { flow_err(err, err_cap, "flow: bad class_type"); return -1; }
      } else if (!strcmp(fk, "inputs")) {
        fj_ws(&c);
        if (c.p >= c.end || *c.p != '{') { flow_err(err, err_cap, "flow: inputs must be obj"); return -1; }
        c.p++;
        fj_ws(&c);
        while (c.p < c.end && *c.p != '}') {
          if (nd->n_inputs >= EC_FLOW_MAX_INPUTS) { flow_err(err, err_cap, "flow: max inputs"); return -1; }
          ec_flow_input_t *in = &nd->inputs[nd->n_inputs];
          memset(in, 0, sizeof *in);
          in->link_node_idx = -1;
          if (fj_str(&c, in->key, sizeof in->key)) { flow_err(err, err_cap, "flow: bad input key"); return -1; }
          fj_ws(&c);
          if (c.p >= c.end || *c.p != ':') { flow_err(err, err_cap, "flow: input :"); return -1; }
          c.p++;
          fj_ws(&c);
          if (c.p >= c.end) { flow_err(err, err_cap, "flow: EOF in input"); return -1; }
          if (*c.p == '[') {
            // ComfyUI link format: ["src_node_id", output_slot_int]
            c.p++;
            if (fj_str(&c, in->str_val, sizeof in->str_val)) { flow_err(err, err_cap, "flow: link target id"); return -1; }
            fj_ws(&c);
            if (c.p >= c.end || *c.p != ',') { flow_err(err, err_cap, "flow: link comma"); return -1; }
            c.p++;
            double slot = 0;
            if (fj_num(&c, &slot)) { flow_err(err, err_cap, "flow: link slot"); return -1; }
            fj_ws(&c);
            if (c.p >= c.end || *c.p != ']') { flow_err(err, err_cap, "flow: link ]"); return -1; }
            c.p++;
            in->kind = EC_VAL_LINK;
            in->link_slot = (int32_t)slot;
          } else if (*c.p == '"') {
            if (fj_str(&c, in->str_val, sizeof in->str_val)) { flow_err(err, err_cap, "flow: input str"); return -1; }
            in->kind = EC_VAL_STRING;
          } else if (*c.p == 't' || *c.p == 'f') {
            int bv = (*c.p == 't') ? 1 : 0;
            if (fj_skip(&c, 0)) return -1;
            in->kind = EC_VAL_NUMBER;
            in->num_val = (double)bv;
          } else if (*c.p == 'n') {
            if (fj_skip(&c, 0)) return -1;
            in->kind = EC_VAL_NONE;
          } else {
            if (fj_num(&c, &in->num_val)) { flow_err(err, err_cap, "flow: input num"); return -1; }
            in->kind = EC_VAL_NUMBER;
          }
          nd->n_inputs++;
          fj_ws(&c);
          if (c.p < c.end && *c.p == ',') { c.p++; fj_ws(&c); continue; }
          break;
        }
        if (c.p >= c.end || *c.p != '}') { flow_err(err, err_cap, "flow: inputs }"); return -1; }
        c.p++;
      } else {
        if (fj_skip(&c, 0)) { flow_err(err, err_cap, "flow: skip unknown field"); return -1; }
      }
      fj_ws(&c);
      if (c.p < c.end && *c.p == ',') { c.p++; fj_ws(&c); continue; }
      break;
    }
    if (c.p >= c.end || *c.p != '}') { flow_err(err, err_cap, "flow: node }"); return -1; }
    c.p++;
    if (nd->class_type[0] == '\0') { flow_err(err, err_cap, "flow: missing class_type"); return -1; }
    g->n_nodes++;
    fj_ws(&c);
    if (c.p < c.end && *c.p == ',') { c.p++; fj_ws(&c); continue; }
    if (c.p < c.end && *c.p == '}') { c.p++; break; }
    flow_err(err, err_cap, "flow: root syntax");
    return -1;
  }
  return 0;
}

static int32_t find_node_idx(const ec_flow_graph_t *g, const char *id) {
  for (int32_t i = 0; i < g->n_nodes; i++) {
    if (!strcmp(g->nodes[i].id, id)) return i;
  }
  return -1;
}

int ec_flow_set_input_num(ec_flow_graph_t *g, const char *node_id, const char *key, double val) {
  int32_t idx = find_node_idx(g, node_id);
  if (idx < 0) return -1;
  ec_flow_node_t *nd = &g->nodes[idx];
  for (int32_t i = 0; i < nd->n_inputs; i++) {
    if (!strcmp(nd->inputs[i].key, key)) {
      nd->inputs[i].kind = EC_VAL_NUMBER;
      nd->inputs[i].num_val = val;
      g->compiled = 0;
      return 0;
    }
  }
  return -1;
}

int ec_flow_set_input_str(ec_flow_graph_t *g, const char *node_id, const char *key, const char *val) {
  int32_t idx = find_node_idx(g, node_id);
  if (idx < 0) return -1;
  ec_flow_node_t *nd = &g->nodes[idx];
  for (int32_t i = 0; i < nd->n_inputs; i++) {
    if (!strcmp(nd->inputs[i].key, key)) {
      nd->inputs[i].kind = EC_VAL_STRING;
      snprintf(nd->inputs[i].str_val, sizeof nd->inputs[i].str_val, "%s", val);
      g->compiled = 0;
      return 0;
    }
  }
  return -1;
}

int ec_flow_compile(ec_flow_graph_t *g, char *err, size_t err_cap) {
  int32_t indeg[EC_FLOW_MAX_NODES] = {0};
  int32_t adj[EC_FLOW_MAX_NODES][EC_FLOW_MAX_NODES];
  int32_t adj_n[EC_FLOW_MAX_NODES] = {0};

  // Resolve links & build adjacency
  for (int32_t i = 0; i < g->n_nodes; i++) {
    ec_flow_node_t *nd = &g->nodes[i];
    for (int32_t k = 0; k < nd->n_inputs; k++) {
      if (nd->inputs[k].kind == EC_VAL_LINK) {
        int32_t src = find_node_idx(g, nd->inputs[k].str_val);
        if (src < 0) { flow_err(err, err_cap, "flow: dangling link target"); return -1; }
        nd->inputs[k].link_node_idx = src;
        adj[src][adj_n[src]++] = i;
        indeg[i]++;
      } else {
        nd->inputs[k].link_node_idx = -1;
      }
    }
  }

  // Kahn topological sort (detects cycles)
  int32_t q[EC_FLOW_MAX_NODES], qh = 0, qt = 0;
  for (int32_t i = 0; i < g->n_nodes; i++) {
    if (indeg[i] == 0) q[qt++] = i;
  }
  int32_t out_n = 0;
  while (qh < qt) {
    int32_t u = q[qh++];
    g->topo_order[out_n++] = u;
    for (int32_t e = 0; e < adj_n[u]; e++) {
      int32_t v = adj[u][e];
      if (--indeg[v] == 0) q[qt++] = v;
    }
  }
  if (out_n != g->n_nodes) {
    flow_err(err, err_cap, "flow: cycle detected in DAG");
    return -1;
  }

  // Static VRAM liveness planner + canonical key-order-invariant SHA-256 Merkle hashing
  for (int32_t step = 0; step < g->n_nodes; step++) {
    int32_t idx = g->topo_order[step];
    ec_flow_node_t *nd = &g->nodes[idx];
    nd->last_consumer = step; // if leaf, stays live until end

    // Sort input indices lexicographically by key so {"a":1,"b":2} == {"b":2,"a":1} (Cowrie canonical parity!)
    int32_t ord[EC_FLOW_MAX_INPUTS];
    for (int32_t k = 0; k < nd->n_inputs; k++) ord[k] = k;
    for (int32_t a = 1; a < nd->n_inputs; a++) {
      int32_t key_idx = ord[a];
      int32_t b = a - 1;
      while (b >= 0 && strcmp(nd->inputs[ord[b]].key, nd->inputs[key_idx].key) > 0) {
        ord[b + 1] = ord[b];
        b--;
      }
      ord[b + 1] = key_idx;
    }

    char canon[8192];
    size_t cpos = (size_t)snprintf(canon, sizeof canon, "v7|%s|", nd->class_type);
    uint64_t h = fnv1a_init();
    h = fnv1a_str(h, nd->class_type);
    for (int32_t oi = 0; oi < nd->n_inputs; oi++) {
      const ec_flow_input_t *in = &nd->inputs[ord[oi]];
      h = fnv1a_str(h, in->key);
      uint8_t kbyte = (uint8_t)in->kind;
      h = fnv1a_mix(h, &kbyte, 1);
      if (in->kind == EC_VAL_LINK) {
        const ec_flow_node_t *up = &g->nodes[in->link_node_idx];
        uint64_t up_h = up->merkle_hash;
        h = fnv1a_mix(h, &up_h, sizeof up_h);
        h = fnv1a_mix(h, &in->link_slot, sizeof in->link_slot);
        if (cpos + 160 < sizeof canon) {
          cpos += (size_t)snprintf(canon + cpos, sizeof canon - cpos, "%s=L:%s:%d;",
                                   in->key, up->merkle_cid, in->link_slot);
        }
        // Extend upstream node's liveness to at least this step
        if (step > g->nodes[in->link_node_idx].last_consumer) {
          g->nodes[in->link_node_idx].last_consumer = step;
        }
      } else if (in->kind == EC_VAL_STRING) {
        h = fnv1a_str(h, in->str_val);
        if (cpos + 128 < sizeof canon) {
          cpos += (size_t)snprintf(canon + cpos, sizeof canon - cpos, "%s=S:%.*s;",
                                   in->key, (int)(sizeof canon - cpos - 96), in->str_val);
        }
      } else if (in->kind == EC_VAL_NUMBER) {
        // Canonicalize -0.0 -> +0.0 per Cowrie float rules
        double v = (in->num_val == 0.0) ? 0.0 : in->num_val;
        h = fnv1a_mix(h, &v, sizeof v);
        if (cpos + 96 < sizeof canon) {
          cpos += (size_t)snprintf(canon + cpos, sizeof canon - cpos, "%s=N:%.12g;", in->key, v);
        }
      }
    }
    ec_cowrie_sha256_hex(canon, cpos, nd->merkle_cid);
    if (ec_cowrie_available()) {
      // Mix high + low 64 bits of SHA-256 for 64-bit fast slot lookup
      char hi16[17];
      memcpy(hi16, nd->merkle_cid, 16); hi16[16] = '\0';
      char lo16[17];
      memcpy(lo16, nd->merkle_cid + 16, 16); lo16[16] = '\0';
      nd->merkle_hash = (uint64_t)strtoull(hi16, NULL, 16) ^ ((uint64_t)strtoull(lo16, NULL, 16) * 0x9E3779B97F4A7C15ULL);
    } else {
      nd->merkle_hash = h;
    }
  }

  g->compiled = 1;
  return 0;
}

static ec_flow_cache_entry_t *cache_find(ec_flow_graph_t *g, uint64_t hash) {
  for (int32_t i = 0; i < g->n_cache; i++) {
    if (g->cache[i].hash == hash) return &g->cache[i];
  }
  return NULL;
}

void ec_flow_clear_cache(ec_flow_graph_t *g) {
  if (!g) return;
  for (int32_t i = 0; i < g->n_cache; i++) {
    free(g->cache[i].payload);
    g->cache[i].payload = NULL;
  }
  g->n_cache = 0;
}

int ec_flow_cache_has(const ec_flow_graph_t *g, uint64_t hash) {
  if (!g) return 0;
  for (int32_t i = 0; i < g->n_cache; i++) {
    if (g->cache[i].hash == hash) return 1;
  }
  return 0;
}

int ec_flow_preload_cache_entry(
    ec_flow_graph_t *g, uint64_t hash, uint64_t result_tag,
    const void *payload, size_t payload_bytes) {
  if (!g || !payload || payload_bytes == 0) return -1;
  if (ec_flow_cache_has(g, hash)) return 0;
  if (g->n_cache >= EC_FLOW_MAX_CACHE) {
    int32_t drop = EC_FLOW_MAX_CACHE / 2;
    for (int32_t c = 0; c < drop; c++) free(g->cache[c].payload);
    memmove(g->cache, g->cache + drop, (size_t)(g->n_cache - drop) * sizeof(ec_flow_cache_entry_t));
    g->n_cache -= drop;
  }
  void *copy = malloc(payload_bytes);
  if (!copy) return -1;
  memcpy(copy, payload, payload_bytes);
  ec_flow_cache_entry_t *slot = &g->cache[g->n_cache++];
  memset(slot, 0, sizeof *slot);
  slot->hash          = hash;
  slot->result_tag    = result_tag;
  slot->payload       = copy;
  slot->payload_bytes = payload_bytes;
  return 0;
}

int ec_flow_execute_ex(
    ec_flow_graph_t *g, ec_node_exec_fn exec_fn, ec_flow_node_event_fn event_fn,
    void *user_ctx, char *err, size_t err_cap) {
  if (!g->compiled) {
    if (ec_flow_compile(g, err, err_cap)) return -1;
  }
  g->last_executed = 0;
  g->last_cache_hits = 0;
  g->last_evictions = 0;

  const ec_flow_cache_entry_t *live[EC_FLOW_MAX_NODES] = {0};

  for (int32_t step = 0; step < g->n_nodes; step++) {
    int32_t idx = g->topo_order[step];
    ec_flow_node_t *nd = &g->nodes[idx];

    ec_flow_cache_entry_t *hit = cache_find(g, nd->merkle_hash);
    if (hit) {
      live[idx] = hit;
      g->last_cache_hits++;
      if (event_fn && event_fn(g, step + 1, g->n_nodes, idx, "cached", hit, 0.0, user_ctx) != 0) {
        flow_err(err, err_cap, "flow: cancelled");
        return -2;
      }
    } else {
      const ec_flow_cache_entry_t *ups[EC_FLOW_MAX_INPUTS];
      int32_t n_up = 0;
      for (int32_t k = 0; k < nd->n_inputs; k++) {
        if (nd->inputs[k].kind == EC_VAL_LINK) {
          ups[n_up++] = live[nd->inputs[k].link_node_idx];
        }
      }
      if (g->n_cache >= EC_FLOW_MAX_CACHE) {
        // Evict oldest half of cache cleanly instead of hard failing long studio sessions
        int32_t drop = EC_FLOW_MAX_CACHE / 2;
        for (int32_t c = 0; c < drop; c++) free(g->cache[c].payload);
        memmove(g->cache, g->cache + drop, (size_t)(g->n_cache - drop) * sizeof(ec_flow_cache_entry_t));
        g->n_cache -= drop;
      }
      if (event_fn && event_fn(g, step, g->n_nodes, idx, "start", NULL, 0.0, user_ctx) != 0) {
        flow_err(err, err_cap, "flow: cancelled");
        return -2;
      }
      ec_flow_cache_entry_t *slot = &g->cache[g->n_cache];
      memset(slot, 0, sizeof *slot);
      slot->hash = nd->merkle_hash;
      double t0 = flow_now_ms();
      int rc = exec_fn(g, idx, ups, n_up, slot, user_ctx);
      double dt = flow_now_ms() - t0;
      if (rc == -2) {
        free(slot->payload);
        slot->payload = NULL;
        flow_err(err, err_cap, "flow: cancelled");
        return -2;
      }
      if (rc != 0) {
        free(slot->payload);
        slot->payload = NULL;
        flow_err(err, err_cap, "flow: node execution failed");
        return -1;
      }
      g->n_cache++;
      live[idx] = slot;
      g->last_executed++;
      if (event_fn && event_fn(g, step + 1, g->n_nodes, idx, "done", slot, dt, user_ctx) != 0) {
        flow_err(err, err_cap, "flow: cancelled");
        return -2;
      }
    }

    // Liveness VRAM eviction planning: mark any upstream whose last_consumer == step
    for (int32_t prev = 0; prev < step; prev++) {
      int32_t pidx = g->topo_order[prev];
      if (g->nodes[pidx].last_consumer == step && live[pidx] != NULL) {
        g->last_evictions++;
      }
    }
  }
  return 0;
}

int ec_flow_execute(ec_flow_graph_t *g, ec_node_exec_fn exec_fn, void *user_ctx, char *err, size_t err_cap) {
  return ec_flow_execute_ex(g, exec_fn, NULL, user_ctx, err, err_cap);
}

const ec_flow_input_t *ec_flow_get_input(const ec_flow_node_t *n, const char *key) {
  if (!n || !key) return NULL;
  for (int32_t i = 0; i < n->n_inputs; i++) {
    if (!strcmp(n->inputs[i].key, key)) return &n->inputs[i];
  }
  return NULL;
}
