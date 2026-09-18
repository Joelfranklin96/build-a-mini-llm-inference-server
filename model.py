"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np
def stable_softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    logits = np.exp(logits)
    temp = np.sum(logits, axis=-1, keepdims=True)
    return logits/temp

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    if temperature > 0:
        return logits/temperature
    return logits

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    
    v = logits.shape[-1]
    if k >= v:
        return logits
    
    kth = np.partition(logits, -k)[..., v-k]
    kth = np.expand_dims(kth, axis=-1)
    out = np.where(logits >= kth, logits, -np.inf)
    return out.astype(logits.dtype, copy=False)

# Step 4 - top_p_filter
import numpy as np

def top_p_filter(logits, p):
    order = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, order, axis=-1)

    shifted = sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)

    cum = np.cumsum(probs, axis=-1)
    keep_sorted = (cum - probs) < p
    keep_sorted[..., 0] = True

    keep = np.empty_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=-1)

    return np.where(keep, logits, -np.inf).astype(logits.dtype, copy=False)

# Step 5 - sample_from_probs
import numpy as np

def sample_from_probs(probs, rng):
    cum_probs = np.cumsum(probs)
    random_num = rng.random()
    
    for i in range(len(cum_probs)):
        if random_num < cum_probs[i]:
            return i
    
    for i in range(len(probs)-1, -1, -1):
        if probs[i] > 0:
            return i

# Step 6 - greedy_select
def greedy_select(logits):
    max_index = 0
    max_value = logits[0]
    for i in range(len(logits)):
        if logits[i] > max_value:
            max_index = i
            max_value = logits[i]
    return max_index

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    token_to_id = {}
    unique_char = set()
    
    for string in corpus:
        for char in string:
            if char not in unique_char:
                unique_char.add(char)

    unique_char = list(unique_char)
    unique_char.sort()
    total_unique_char = special_tokens + unique_char

    for i, char in enumerate(total_unique_char):
        token_to_id[char] = i
    
    return {'token_to_id': token_to_id, 'id_to_token': total_unique_char}

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    token_to_id = vocab['token_to_id']
    res = []
    unk_present = '<unk>' in token_to_id
    if add_bos:
        res.append(token_to_id['<bos>'])
    
    for char in text:
        if char in token_to_id:
            res.append(token_to_id[char])
        else:
            if unk_present:
                res.append(token_to_id['<unk>'])
    return res

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    id_to_token = vocab['id_to_token']
    res = []
    for t_id in token_ids:
        token = id_to_token[t_id]
        if token[0] == "<" and token[-1] == ">":
            if not(skip_special):
                res.append(token)
        else:
            res.append(token)
    
    return "".join(res)

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    return embedding_matrix[token_ids]

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    if bias is not None:
        out = np.matmul(x, weight) + bias
    else:
        out = np.matmul(x, weight)
    return out

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    K = np.zeros((max_seq_len, d_model), dtype=np.float32)
    V = np.zeros((max_seq_len, d_model), dtype=np.float32)
    length = 0
    return {'K': K, 'V': V, 'length': length}

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    length = cache['length']
    t = k_new.shape[0]
    cache['K'][length:length + t, :] = k_new
    cache['V'][length:length + t, :] = v_new
    cache['length'] += t
    return cache

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    Tq, D = q.shape
    Tk, D = k.shape
    raw_scores = np.matmul(q, np.transpose(k))
    raw_scores = raw_scores/D**0.5
    mask = np.tril(np.ones((Tq, Tk), dtype=bool), k=Tk - Tq)
    if is_causal:
        out = np.where(mask, raw_scores, -np.inf)
    else:
        out = raw_scores
    
    out = stable_softmax(out)
    return np.matmul(out, v)

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    embedding_matrix = params['embedding']
    embeddings = embed_tokens(token_ids, embedding_matrix)
    Wq = params['Wq']
    Wk = params['Wk']
    Wv = params['Wv']
    Wo = params['Wo']
    W_out = params['W_out']
    max_seq_len = params['max_seq_len']
    Q = linear_projection(embeddings, Wq)
    K = linear_projection(embeddings, Wk)
    V = linear_projection(embeddings, Wv)
    d_model = K.shape[-1]
    cache = init_kv_cache(max_seq_len, d_model)
    cache = append_kv(cache, K, V)
    output = causal_attention(Q, K, V, is_causal=True)
    final_output = np.matmul(output, Wo)
    last_position = final_output[-1]
    logits = np.matmul(last_position.astype(np.float64), W_out.astype(np.float64))
    return (logits, cache)

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    embedding_matrix = params['embedding']
    Wq = params['Wq']
    Wk = params['Wk']
    Wv = params['Wv']
    Wo = params['Wo']
    W_out = params['W_out']

    new_token_embedding = embed_tokens([token_id], embedding_matrix)
    new_k = linear_projection(new_token_embedding, Wk)
    new_v = linear_projection(new_token_embedding, Wv)
    new_q = linear_projection(new_token_embedding, Wq)
    cache = append_kv(cache, new_k, new_v)

    length = cache['length']
    K = cache['K'][0:length]
    V = cache['V'][0:length]

    output = causal_attention(new_q, K, V, is_causal=True)
    final_output = linear_projection(output, Wo)
    logits = np.matmul(final_output, W_out)
    logits = logits[0]
    return (logits, cache)

