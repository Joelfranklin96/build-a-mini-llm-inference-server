"""A single-layer causal attention block with prefill and incremental decode passes."""

import numpy as np

from .sampling import stable_softmax


def init_model_params(vocab_size, d_model, max_seq_len, rng):
    """Create randomly initialised weights for the single-layer attention block."""
    scale = 1.0 / np.sqrt(d_model)

    return {
        'embedding': rng.standard_normal((vocab_size, d_model)) * scale,
        'Wq': rng.standard_normal((d_model, d_model)) * scale,
        'Wk': rng.standard_normal((d_model, d_model)) * scale,
        'Wv': rng.standard_normal((d_model, d_model)) * scale,
        'Wo': rng.standard_normal((d_model, d_model)) * scale,
        'W_out': rng.standard_normal((d_model, vocab_size)) * scale,
        'd_model': d_model,
        'max_seq_len': max_seq_len,
    }


def embed_tokens(token_ids, embedding_matrix):
    """Look up the embedding rows for a sequence of token ids."""
    return embedding_matrix[token_ids]


def linear_projection(x, weight, bias=None):
    """Apply a dense projection ``x @ weight`` with an optional bias."""
    projected = np.matmul(x, weight)
    if bias is not None:
        projected = projected + bias

    return projected


def init_kv_cache(max_seq_len, d_model):
    """Allocate an empty contiguous key/value cache."""
    return {
        'K': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'V': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'length': 0,
    }


def append_kv(cache, k_new, v_new):
    """Append new key/value rows to a contiguous cache in place and return it."""
    length = cache['length']
    num_new = k_new.shape[0]

    cache['K'][length:length + num_new, :] = k_new
    cache['V'][length:length + num_new, :] = v_new
    cache['length'] = length + num_new

    return cache


def causal_attention(q, k, v, is_causal=True):
    """Scaled dot-product attention, masked so each query only sees keys up to its position."""
    num_queries, d_model = q.shape
    num_keys = k.shape[0]

    scores = np.matmul(q, k.T) / d_model ** 0.5
    if is_causal:
        mask = np.tril(np.ones((num_queries, num_keys), dtype=bool), k=num_keys - num_queries)
        scores = np.where(mask, scores, -np.inf)

    return np.matmul(stable_softmax(scores), v)


def model_prefill(token_ids, params):
    """Run the whole prompt through the model, returning last-position logits and a filled cache."""
    embeddings = embed_tokens(token_ids, params['embedding'])

    queries = linear_projection(embeddings, params['Wq'])
    keys = linear_projection(embeddings, params['Wk'])
    values = linear_projection(embeddings, params['Wv'])

    cache = init_kv_cache(params['max_seq_len'], keys.shape[-1])
    append_kv(cache, keys, values)

    attended = causal_attention(queries, keys, values, is_causal=True)
    hidden = linear_projection(attended, params['Wo'])
    logits = linear_projection(hidden[-1].astype(np.float64), params['W_out'].astype(np.float64))

    return logits, cache


def model_decode_step(token_id, cache, params):
    """Advance generation by one token, reusing and extending the contiguous KV cache."""
    embedding = embed_tokens([token_id], params['embedding'])
    query = linear_projection(embedding, params['Wq'])

    append_kv(cache,
              linear_projection(embedding, params['Wk']),
              linear_projection(embedding, params['Wv']))

    length = cache['length']
    attended = causal_attention(query, cache['K'][:length], cache['V'][:length], is_causal=True)
    hidden = linear_projection(attended, params['Wo'])
    logits = linear_projection(hidden, params['W_out'])

    return logits[0], cache
