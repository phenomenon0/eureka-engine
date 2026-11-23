// ec_cowrie.h — Optional Zero-Build-Dep Tessera / Cowrie v2 Bridge for Eureka Creative Engine
// Dynamically loads libcowrie_core.so via dlopen (-ldl only; graceful fallback when absent).
// Provides:
//   Phase A: 16 KB-aligned SHA-NI Merkle StatePages (.state.cowrie) with (model_sha, doc_cid, kv_quant)
//            Coordinate Contracts for Wan 2.2 UMT5-XXL, FLUX.2 Qwen3-4B, TRELLIS.2 DINOv3, and
//            persistent cross-restart ComfyUI DAG node caches.
//   Phase B: Canonical COWR framing, out-of-band Trust Membrane (UNTRUSTED_EXT = 0x04 + sticky
//            BIT_TAINTED = 0x80 causal propagation), Creative Trust Gate (403 on tainted exec),
//            and deterministic R0 / R1 / R2 Director LLM lenses.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Authority & epistemic constants matching Cowrie v2 SPEC.md §5
#define EC_COWRIE_AUTH_SYS_POLICY    0x01u
#define EC_COWRIE_AUTH_VERIFIED_TOOL 0x02u
#define EC_COWRIE_AUTH_UNTRUSTED_EXT 0x04u
#define EC_COWRIE_AUTH_MODEL_SYNTH   0x08u
#define EC_COWRIE_BIT_TAINTED        0x80u
#define EC_COWRIE_EPI_OBSERVED       0x01u

// StatePage kv_quant domain tags for Creative Engine tensors & DAG records
#define EC_COWRIE_KV_BF16            0u
#define EC_COWRIE_KV_DAG_NODEVAL     1u

// Exact CowrieError codes from cowrie-core/src/ffi.rs
#define EC_COWRIE_ERR_MODEL_MISMATCH          23
#define EC_COWRIE_ERR_STATEPAGE_HASH_MISMATCH 24
#define EC_COWRIE_ERR_TAINTED_EXEC_BLOCKED    25
#define EC_COWRIE_ERR_KV_QUANT_MISMATCH       29

typedef struct {
  int      enabled;
  int      shani;
  uint64_t sp_saves;
  uint64_t sp_hits;
  uint64_t sp_misses;
  uint64_t sp_mismatches;
  int      trust_entries;
  char     cache_dir[320];
} ec_cowrie_stats_t;

typedef struct {
  int     ok;
  char    cid_hex[69];   // 34-byte multihash hex (68 hex chars + NUL)
  char    handle[24];    // Alpha32 R0 short handle (e.g. "@a7f2bc")
  uint8_t authority;
  uint8_t epistemic;
  int     tainted;
  size_t  raw_bytes;
  size_t  cowr_bytes;
  char    r0[160];
  char    r1[1024];
} ec_cowrie_trust_meta_t;

// Lifecycle & configuration (thread-safe, idempotent)
int  ec_cowrie_init(void);
int  ec_cowrie_available(void);
int  ec_cowrie_has_shani(void);
void ec_cowrie_set_cache_dir(const char *dir);
const char *ec_cowrie_get_cache_dir(void);
void ec_cowrie_get_stats(ec_cowrie_stats_t *out);

// SHA-NI accelerated SHA-256 hex digest (64 hex chars + NUL). Returns 0 on success.
int ec_cowrie_sha256_hex(const void *data, size_t len, char out_hex_65[65]);

// Fast sub-millisecond deterministic 64-hex model fingerprint (hashes file size + head 64KB + tail 64KB).
int ec_cowrie_file_fingerprint(const char *path, char out_hex_65[65]);

// Combine multiple strings/digests into a deterministic 64-hex composite model contract SHA.
int ec_cowrie_composite_sha(const char *const *parts, int n_parts, char out_hex_65[65]);

// Phase A: Persistent 16 KB-aligned SHA-NI StatePage cache API
// Returns 0 on verified hit/save, negative CowrieError code (e.g. -23 ERR_MODEL_MISMATCH) on contract mismatch, -1 on miss.
int ec_cowrie_sp_save(
    const char *ns,
    const char *model_sha_64,
    const void *key_data, size_t key_len,
    uint32_t ctx_tokens, uint8_t kv_quant,
    const void *state_data, size_t state_len);

int ec_cowrie_sp_load(
    const char *ns,
    const char *expected_model_sha_64,
    const void *key_data, size_t key_len,
    int expected_kv_quant,
    void *out_buf, size_t expected_len);

// Low-level path-explicit StatePage API (used for direct G-C6 verification gates)
int ec_cowrie_sp_save_file(
    const char *path,
    const char *model_sha_64,
    const char *doc_cid_64,
    uint32_t ctx_tokens, uint8_t kv_quant,
    const void *state_data, size_t state_len);

int ec_cowrie_sp_load_file(
    const char *path,
    const char *expected_model_sha_64,
    int expected_kv_quant,
    void *out_buf, size_t max_len, size_t *out_actual_len);

// Purge cached .state.cowrie files whose basename starts with ns_prefix (e.g. "dag_")
int ec_cowrie_sp_purge_prefix(const char *ns_prefix);

// Phase B: Canonical COWR framing, Trust Membrane & R0/R1/R2 Lenses
uint8_t ec_cowrie_propagate_authority(uint8_t base_auth, const uint8_t *parents, size_t n_parents);
int     ec_cowrie_is_tainted(uint8_t authority);

// Canonicalizes JSON -> COWR frame, registers in Creative Trust Store, fills out_meta.
int ec_cowrie_trust_register_json(
    const char *json, size_t json_len,
    uint8_t base_authority,
    const char *source, const char *label,
    const char *const *parent_ids, int n_parents,
    ec_cowrie_trust_meta_t *out_meta);

// Evaluates Trust Gate for privileged vs read-only action.
// Returns HTTP status code: 200 (allowed) or 403 (ERR_TAINTED_EXECUTION_BLOCKED).
int ec_cowrie_trust_gate_json(
    const char *action,
    const char *const *arg_ids, int n_args,
    int owner_override,
    char *out_json, size_t out_cap);

// Lists registered Cowrie Trust Membrane entries as compact R0 triage JSON.
int ec_cowrie_trust_triage_json(char *out_json, size_t out_cap);

// Projects arbitrary Studio JSON payload into deterministic R0 (tier=0), R1 (tier=1), or R2 (tier=2) lens JSON response.
int ec_cowrie_lens_response_json(
    const char *target,
    const char *raw_json, size_t raw_len,
    uint8_t authority, int tier,
    char *out_json, size_t out_cap);

// Canonical COWR encode/verify helper for direct testing (returns 0 on OK + writes 68-hex CID)
int ec_cowrie_canonical_json_cid(
    const char *json, size_t json_len, uint8_t authority,
    char out_cid_hex[69], size_t *out_cowr_len);

#ifdef __cplusplus
}
#endif
