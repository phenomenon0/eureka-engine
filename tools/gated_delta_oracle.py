"""CPU Oracle Twin for Qwen3.5 Gated Linear Attention and Output-Gate Op.

Reference semantics for:
1. Gated Delta Net (recurrent formulation and chunked formulation)
2. 4-tap depthwise causal 1D convolution with SiLU
3. Per-head L2 normalization on query and key
4. Gated RMSNorm: RMSNorm(x) * weight * silu(z)
5. Full-attention output gate: x * sigmoid(gate)
6. 2-layer hybrid mini-model forward check
"""

import numpy as np

f32 = np.float32
EPS = f32(1e-6)


def l2norm(x, eps=EPS):
    """Per-head L2 normalization along the last dimension: x / sqrt(sum(x^2) + eps)."""
    x = np.asarray(x, dtype=f32)
    inv = f32(1.0) / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * inv


def silu(x):
    """SiLU / Swish activation: x / (1 + exp(-x))."""
    x = np.asarray(x, dtype=f32)
    return x / (f32(1.0) + np.exp(-x))


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    x = np.asarray(x, dtype=f32)
    return f32(1.0) / (f32(1.0) + np.exp(-x))


def causal_conv1d_step(x_t, state, weight):
    """Single-token causal depthwise 1D conv with kernel size 4 and SiLU.
    
    Args:
        x_t: [C] input channels for current token
        state: [C, 3] previous 3 tokens
        weight: [C, 4] depthwise conv weights
    Returns:
        out_t: [C] activated output
        next_state: [C, 3] updated state
    """
    x_t = np.asarray(x_t, dtype=f32)
    state = np.asarray(state, dtype=f32)
    weight = np.asarray(weight, dtype=f32)
    C = len(x_t)
    
    window = np.empty((C, 4), dtype=f32)
    window[:, :3] = state
    window[:, 3] = x_t
    
    # 4-tap dot product along kernel dimension
    conv_val = np.sum(window * weight, axis=1)
    out_t = silu(conv_val)
    next_state = window[:, 1:].copy()
    return out_t, next_state


def causal_conv1d_seq(x, weight, initial_state=None):
    """Multi-token causal depthwise 1D conv with kernel size 4 and SiLU.
    
    Args:
        x: [C, T] input channels over sequence
        weight: [C, 4] depthwise conv weights
        initial_state: [C, 3] or None
    Returns:
        out: [C, T]
        final_state: [C, 3]
    """
    x = np.asarray(x, dtype=f32)
    weight = np.asarray(weight, dtype=f32)
    C, T = x.shape
    if initial_state is None:
        state = np.zeros((C, 3), dtype=f32)
    else:
        state = np.asarray(initial_state, dtype=f32).copy()
        
    out = np.zeros((C, T), dtype=f32)
    for t in range(T):
        out_t, state = causal_conv1d_step(x[:, t], state, weight)
        out[:, t] = out_t
    return out, state


def recurrent_gated_delta_step(q_t, k_t, v_t, g_t, beta_t, state, scale_q=True):
    """Single-token recurrent state update and attention readout.
    
    Args:
        q_t: [H, D_k] query
        k_t: [H, D_k] key
        v_t: [H, D_v] value
        g_t: [H] log-decay (decay = exp(g_t))
        beta_t: [H] learning rate beta in (0, 1)
        state: [H, D_k, D_v] recurrent state
        scale_q: whether to scale query by 1 / sqrt(D_k)
    Returns:
        out_t: [H, D_v]
        next_state: [H, D_k, D_v]
    """
    q_t = np.asarray(q_t, dtype=f32)
    k_t = np.asarray(k_t, dtype=f32)
    v_t = np.asarray(v_t, dtype=f32)
    g_t = np.asarray(g_t, dtype=f32)
    beta_t = np.asarray(beta_t, dtype=f32)
    state = np.asarray(state, dtype=f32).copy()
    
    H, D_k = q_t.shape
    D_v = v_t.shape[-1]
    
    if scale_q:
        q_eff = q_t * f32(1.0 / np.sqrt(D_k))
    else:
        q_eff = q_t
        
    decay = np.exp(g_t) # [H]
    
    # 1. Decay the recurrent state
    state = state * decay[:, None, None]
    
    # 2. Read memory prediction: pred = sum_i k[i] * S[i, j]
    kv_mem = np.einsum("h i j, h i -> h j", state, k_t)
    
    # 3. Compute delta correction: (v - kv_mem) * beta
    delta = (v_t - kv_mem) * beta_t[:, None]
    
    # 4. Update state: S[i, j] += k[i] * delta[j]
    state += np.einsum("h i, h j -> h i j", k_t, delta)
    
    # 5. Read attention output: out = sum_i q[i] * S[i, j]
    out_t = np.einsum("h i j, h i -> h j", state, q_eff)
    
    return out_t, state


def recurrent_gated_delta_seq(q, k, v, g, beta, initial_state=None, scale_q=True):
    """Sequence recurrent gated delta net.
    
    Args:
        q: [T, H, D_k]
        k: [T, H, D_k]
        v: [T, H, D_v]
        g: [T, H]
        beta: [T, H]
        initial_state: [H, D_k, D_v] or None
        scale_q: bool
    Returns:
        out: [T, H, D_v]
        final_state: [H, D_k, D_v]
    """
    q = np.asarray(q, dtype=f32)
    k = np.asarray(k, dtype=f32)
    v = np.asarray(v, dtype=f32)
    g = np.asarray(g, dtype=f32)
    beta = np.asarray(beta, dtype=f32)
    
    T, H, D_k = q.shape
    D_v = v.shape[-1]
    
    if initial_state is None:
        state = np.zeros((H, D_k, D_v), dtype=f32)
    else:
        state = np.asarray(initial_state, dtype=f32).copy()
        
    out = np.zeros((T, H, D_v), dtype=f32)
    for t in range(T):
        out[t], state = recurrent_gated_delta_step(
            q[t], k[t], v[t], g[t], beta[t], state, scale_q=scale_q
        )
    return out, state


def solve_unit_lower_triangular(L, b):
    """Forward substitution solving L @ y = b where L has ones on diagonal.
    
    Args:
        L: [H, N, C, C] unit lower triangular
        b: [H, N, C, D]
    Returns:
        y: [H, N, C, D]
    """
    C = L.shape[-1]
    y = np.zeros_like(b)
    for i in range(C):
        if i > 0:
            sub = np.sum(L[..., i, :i, None] * y[..., :i, :], axis=-2)
            y[..., i, :] = b[..., i, :] - sub
        else:
            y[..., 0, :] = b[..., 0, :]
    return y


def chunk_gated_delta_seq(q, k, v, g, beta, chunk_size=16, initial_state=None, scale_q=True):
    """Chunked/parallel formulation of gated delta net.
    
    Splits sequence of length T into chunks of size chunk_size, computes intra-chunk
    quantities via lower triangular solve, and scans over chunks carrying the recurrent state.
    
    Args:
        q: [T, H, D_k]
        k: [T, H, D_k]
        v: [T, H, D_v]
        g: [T, H]
        beta: [T, H]
        chunk_size: int (e.g. 16 or 64)
        initial_state: [H, D_k, D_v] or None
        scale_q: bool
    Returns:
        out: [T, H, D_v]
        final_state: [H, D_k, D_v]
    """
    q = np.asarray(q, dtype=f32)
    k = np.asarray(k, dtype=f32)
    v = np.asarray(v, dtype=f32)
    g = np.asarray(g, dtype=f32)
    beta = np.asarray(beta, dtype=f32)
    
    T, H, D_k = q.shape
    D_v = v.shape[-1]
    
    # Transpose to [H, T, D]
    q_h = q.transpose(1, 0, 2).copy()
    k_h = k.transpose(1, 0, 2).copy()
    v_h = v.transpose(1, 0, 2).copy()
    g_h = g.transpose(1, 0).copy()
    beta_h = beta.transpose(1, 0).copy()
    
    if scale_q:
        q_h = q_h * f32(1.0 / np.sqrt(D_k))
        
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    if pad_size > 0:
        q_h = np.pad(q_h, ((0, 0), (0, pad_size), (0, 0)))
        k_h = np.pad(k_h, ((0, 0), (0, pad_size), (0, 0)))
        v_h = np.pad(v_h, ((0, 0), (0, pad_size), (0, 0)))
        g_h = np.pad(g_h, ((0, 0), (0, pad_size)))
        beta_h = np.pad(beta_h, ((0, 0), (0, pad_size)))
        
    total_T = T + pad_size
    num_chunks = total_T // chunk_size
    
    v_beta = v_h * beta_h[..., None]
    k_beta = k_h * beta_h[..., None]
    
    # Reshape to [H, num_chunks, chunk_size, D]
    q_c = q_h.reshape(H, num_chunks, chunk_size, D_k)
    k_c = k_h.reshape(H, num_chunks, chunk_size, D_k)
    k_beta_c = k_beta.reshape(H, num_chunks, chunk_size, D_k)
    v_beta_c = v_beta.reshape(H, num_chunks, chunk_size, D_v)
    g_c = g_h.reshape(H, num_chunks, chunk_size)
    
    cum_decay = np.cumsum(g_c, axis=-1)
    
    # pairwise_decay[h, n, i, j] = exp(cum_decay[i] - cum_decay[j]) for i >= j
    diff = cum_decay[..., :, None] - cum_decay[..., None, :]
    strictly_upper = np.triu(np.ones((chunk_size, chunk_size), dtype=bool), k=1)
    diff[..., strictly_upper] = -np.inf
    pairwise_decay = np.exp(diff)
    
    ut_system = (k_beta_c @ k_c.swapaxes(-1, -2)) * pairwise_decay
    intra_chunk_attn = (q_c @ k_c.swapaxes(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta_c * np.exp(cum_decay)[..., None]
    
    # L has 1s on diagonal, strictly lower triangular from ut_system
    L = np.tril(ut_system, -1) + np.eye(chunk_size, dtype=f32)
    new_values = solve_unit_lower_triangular(L, v_beta_c)
    k_cumdecay = solve_unit_lower_triangular(L, decayed_k_beta)
    
    if initial_state is None:
        last_state = np.zeros((H, D_k, D_v), dtype=f32)
    else:
        last_state = np.asarray(initial_state, dtype=f32).copy()
        
    core_attn_out = np.zeros((H, num_chunks, chunk_size, D_v), dtype=f32)
    
    q_decayed = q_c * np.exp(cum_decay)[..., None]
    k_decayed = k_c * np.exp(cum_decay[..., -1:] - cum_decay)[..., None]
    chunk_decay = np.exp(cum_decay[..., -1])[..., None, None]
    
    for i in range(num_chunks):
        v_new = new_values[:, i] - k_cumdecay[:, i] @ last_state
        inter_chunk_attn = q_decayed[:, i] @ last_state
        core_attn_out[:, i] = inter_chunk_attn + intra_chunk_attn[:, i] @ v_new
        last_state = last_state * chunk_decay[:, i] + k_decayed[:, i].swapaxes(-1, -2) @ v_new
        
    core_attn_out = core_attn_out.reshape(H, -1, D_v)[:, :T, :]
    out = core_attn_out.transpose(1, 0, 2)
    return out, last_state


def rmsnorm_gated(x, z, weight, eps=EPS):
    """Gated RMSNorm: RMSNorm(x) * weight * silu(z).
    
    Args:
        x: [..., D] values to normalize
        z: [..., D] gate inputs
        weight: [D] learned scale weight
        eps: float
    Returns:
        out: [..., D]
    """
    x = np.asarray(x, dtype=f32)
    z = np.asarray(z, dtype=f32)
    weight = np.asarray(weight, dtype=f32)
    
    var = np.mean(x * x, axis=-1, keepdims=True)
    normed = x * (f32(1.0) / np.sqrt(var + eps))
    return (normed * weight) * silu(z)


def attn_output_gate(x, gate):
    """Sigmoid output gate for full attention: x * sigmoid(gate).
    
    Args:
        x: [..., D] attention output
        gate: [..., D] gate logits
    Returns:
        out: [..., D]
    """
    x = np.asarray(x, dtype=f32)
    gate = np.asarray(gate, dtype=f32)
    return x * sigmoid(gate)


class Qwen35LinearAttentionWeights:
    """Convenience container for linear attention layer weights."""
    def __init__(self, hidden_size=4096, num_k_heads=16, num_v_heads=32, head_dim=128):
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim
        
        self.key_dim = num_k_heads * head_dim
        self.value_dim = num_v_heads * head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        
        # Projections
        self.w_qkv = np.zeros((self.conv_dim, hidden_size), dtype=f32)
        self.w_z = np.zeros((self.value_dim, hidden_size), dtype=f32)
        self.w_b = np.zeros((num_v_heads, hidden_size), dtype=f32)
        self.w_a = np.zeros((num_v_heads, hidden_size), dtype=f32)
        
        # Conv1d
        self.conv_weight = np.zeros((self.conv_dim, 4), dtype=f32)
        
        # Discretization parameters
        self.dt_bias = np.ones(num_v_heads, dtype=f32)
        self.a_log = np.log(np.full(num_v_heads, 1.0, dtype=f32))
        
        # RMSNormGated weight
        self.norm_weight = np.ones(head_dim, dtype=f32)
        
        # Out projection
        self.w_out = np.zeros((hidden_size, self.value_dim), dtype=f32)


def softplus(x):
    """Stable softplus: log(1 + exp(x))."""
    x = np.asarray(x, dtype=f32)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -80.0, 20.0))))


def linear_attention_layer_forward(h, weights: Qwen35LinearAttentionWeights, conv_state=None, rec_state=None):
    """Full single-token forward pass of Qwen3.5 linear attention layer."""
    h = np.asarray(h, dtype=f32) # [hidden_size]
    
    # 1. Input projections
    qkv = weights.w_qkv @ h      # [conv_dim]
    z = weights.w_z @ h          # [value_dim]
    b = weights.w_b @ h          # [num_v_heads]
    a = weights.w_a @ h          # [num_v_heads]
    
    # 2. Causal Conv1D
    if conv_state is None:
        conv_state = np.zeros((weights.conv_dim, 3), dtype=f32)
    qkv_conv, next_conv_state = causal_conv1d_step(qkv, conv_state, weights.conv_weight)
    
    # 3. Split into Q, K, V
    q = qkv_conv[:weights.key_dim].reshape(weights.num_k_heads, weights.head_dim)
    k = qkv_conv[weights.key_dim:2*weights.key_dim].reshape(weights.num_k_heads, weights.head_dim)
    v = qkv_conv[2*weights.key_dim:].reshape(weights.num_v_heads, weights.head_dim)
    
    # 4. L2-norm on Q and K
    q = l2norm(q)
    k = l2norm(k)
    
    # 5. Repeat Q and K to match V heads (GQA)
    rep = weights.num_v_heads // weights.num_k_heads
    if rep > 1:
        q = np.repeat(q, rep, axis=0)
        k = np.repeat(k, rep, axis=0)
        
    # 6. Compute beta and decay g
    beta = sigmoid(b)
    g = -np.exp(weights.a_log) * softplus(a + weights.dt_bias)
    
    # 7. Recurrent gated delta update
    if rec_state is None:
        rec_state = np.zeros((weights.num_v_heads, weights.head_dim, weights.head_dim), dtype=f32)
        
    core_attn_out, next_rec_state = recurrent_gated_delta_step(
        q, k, v, g, beta, rec_state, scale_q=True
    )
    
    # 8. Gated RMSNorm
    z = z.reshape(weights.num_v_heads, weights.head_dim)
    normed_out = rmsnorm_gated(core_attn_out, z, weights.norm_weight)
    
    # 9. Output projection
    out = weights.w_out @ normed_out.reshape(-1)
    
    return out, next_conv_state, next_rec_state
