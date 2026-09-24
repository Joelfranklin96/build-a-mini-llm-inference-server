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

    probs = stable_softmax(sorted_logits)

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
    final_output = linear_projection(output, Wo)
    last_position = final_output[-1]
    logits = linear_projection(last_position.astype(np.float64), W_out.astype(np.float64))
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

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    return int(np.ceil(num_tokens/block_size))

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    K_blocks = np.zeros((num_blocks, block_size, d_model), dtype=np.float32)
    V_blocks = np.zeros((num_blocks, block_size, d_model), dtype=np.float32)

    free_list = list(range(num_blocks))
    seq_tables = {}

    return {'K_blocks': K_blocks, 'V_blocks': V_blocks, 'free_list': free_list,
    'block_size': block_size, 'num_blocks': num_blocks, 'd_model': d_model,
    'seq_tables': seq_tables}

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):

    if len(allocator['free_list']) == 0:
        raise RuntimeError("out-of-memory")
    
    block_id = allocator['free_list'].pop()
    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'][seq_id] = []
    allocator['seq_tables'][seq_id].append(block_id)
    return block_id

# Step 20 - free_block
def free_block(allocator, block_id):
    allocator['free_list'].append(block_id)

# Step 21 - append_to_paged_cache
def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
    lengths = allocator.setdefault('seq_lengths', {})
    L = lengths.get(seq_id, 0)
    block_size = allocator['block_size']
    t = k_new.shape[0]

    written = 0
    while written < t:
        pos = L + written
        slot = pos % block_size
        if slot == 0:                                   
            allocate_block(allocator, seq_id)
        block_id = allocator['seq_tables'][seq_id][-1]
        n = min(block_size - slot, t - written)
        allocator['K_blocks'][block_id, slot:slot+n, :] = k_new[written: written+n, :]
        allocator['V_blocks'][block_id, slot:slot+n, :] = v_new[written: written+n, :]
        written += n

    lengths[seq_id] = L + t

# Step 22 - gather_kv_from_blocks
def gather_kv_from_blocks(allocator, seq_id):
    L = allocator.get('seq_lengths', {}).get(seq_id, 0)
    d_model = allocator['d_model']
    block_ids = allocator['seq_tables'][seq_id]
    k = allocator['K_blocks'][block_ids].reshape(-1, d_model)[:L]
    v = allocator['V_blocks'][block_ids].reshape(-1, d_model)[:L]
    return (k, v)

# Step 23 - paged_attention_step
def paged_attention_step(q, allocator, seq_id):
    k, v = gather_kv_from_blocks(allocator, seq_id)
    output = causal_attention(q, k, v, is_causal=True)
    return output.astype(np.float64)

# Step 24 - free_sequence_blocks
def free_sequence_blocks(allocator, seq_id):
    
    block_ids = allocator['seq_tables'].pop(seq_id, [])
    for block_id in block_ids:
        free_block(allocator, block_id)

    if 'seq_lengths' in allocator:
        allocator['seq_lengths'].pop(seq_id, None)

# Step 25 - kv_blocks_in_use
def kv_blocks_in_use(allocator):
    total = allocator['K_blocks'].shape[0]
    free = len(allocator['free_list'])
    used = total - free
    return {'used': used, 'free':free, 'total': total}

# Step 26 - make_request
def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    return {'request_id': request_id, 'prompt_token_ids': list(prompt_token_ids), 'max_new_tokens': max_new_tokens, 'sampling_params': sampling_params}

# Step 27 - init_sequence_state
def init_sequence_state(request, params):
    output = {}
    output['request_id'] = request['request_id']
    output['prompt_token_ids'] = request['prompt_token_ids']
    gen = []
    output['generated'] = gen
    output['generated_token_ids'] = gen
    logits, cache = model_prefill(request['prompt_token_ids'], params)
    output['last_logits'] = logits
    output['cache'] = cache
    output['done'] = False
    output['sampling_params'] = request['sampling_params']
    output['max_new_tokens'] = request['max_new_tokens']
    return output

# Step 28 - sequence_decode_step
def sequence_decode_step(state, params, rng):
    cache = state['cache']
    last_logits = state['last_logits']
    sampling_params = state['sampling_params']

    t = sampling_params.get('temperature')
    if sampling_params.get('greedy', False) or (t is not None and t <= 0):
        new_token_id = greedy_select(last_logits)
    else:
        if 'temperature' in sampling_params:
            t = sampling_params['temperature']
            last_logits = apply_temperature(last_logits, t)
        if sampling_params.get('top_k', 0) > 0:
            k = sampling_params['top_k']
            last_logits = top_k_filter(last_logits, k)
        if sampling_params.get('top_p', 1) < 1:
            p = sampling_params['top_p']
            last_logits = top_p_filter(last_logits, p)
    
        probs = stable_softmax(last_logits)
        new_token_id = sample_from_probs(probs, rng)
    
    state['generated'].append(new_token_id)
    next_last_logits, cache = model_decode_step(new_token_id, cache, params)
    state['cache'] = cache
    state['last_logits'] = next_last_logits
    return (new_token_id, state)

# Step 29 - is_sequence_done
def is_sequence_done(state, eos_token_id):
    
    gen = state['generated']
    gen_length = len(gen)
    max_new_tokens = state['max_new_tokens']

    if gen_length == 0:
        if max_new_tokens > 0:
            return False
        else:
            return True
    else:
        last_token_id = gen[-1]
        if gen_length >= max_new_tokens or last_token_id == eos_token_id:
            return True
        else:
            return False

# Step 30 - generate_single_sequence
def generate_single_sequence(request, params, eos_token_id, rng):
    state = init_sequence_state(request, params)
    while not(is_sequence_done(state, eos_token_id)):
        next_token_id, state = sequence_decode_step(state, params, rng)
    
    return state['generated']

# Step 31 - build_batch_step_input
import numpy as np

def build_batch_step_input(sequences):

    active_indices = []
    input_ids = []
    for i, sequence in enumerate(sequences):
        if not(sequence['done']):
            active_indices.append(i)
            input_ids.append(sequence['token_ids'][-1])
    
    return {'active_indices': active_indices, 'input_ids': np.array(input_ids, dtype=np.int64)}

# Step 32 - batched_decode_step
def batched_decode_step(params, sequences, sampling_params, rng=None):
    """Advance every active sequence by one token in lockstep; mutates and returns sequences."""
    if rng is None:
        rng = sampling_params.get('rng')
    if rng is None:
        rng = np.random.default_rng()

    for seq in sequences:
        if seq['done']:
            continue

        logits, seq['kv_cache'] = model_decode_step(seq['token_ids'][-1], seq['kv_cache'], params)

        t = sampling_params.get('temperature')
        if sampling_params.get('greedy', False) or (t is not None and t <= 0):
            next_token = greedy_select(logits)
        else:
            if t is not None:
                logits = apply_temperature(logits, t)
            if sampling_params.get('top_k', 0) > 0:
                logits = top_k_filter(logits, sampling_params['top_k'])
            if sampling_params.get('top_p', 1.0) < 1.0:
                logits = top_p_filter(logits, sampling_params['top_p'])
            next_token = sample_from_probs(stable_softmax(logits), rng)

        seq['token_ids'].append(next_token)

    return sequences

# Step 33 - static_batch_generate
def static_batch_generate(params, requests, sampling_config, max_new_tokens, rng=None):
    """Run prefill for all requests, then iterate batched decode steps until each
    sequence hits its per-request budget or the global max_new_tokens cap."""

    if rng is None:
        rng = sampling_config.get('rng') or np.random.default_rng()

    sequences = []
    for request in requests:
        sequence = {}
        sequence['request_id'] = request['request_id']
        sequence['budget'] = min(request['max_new_tokens'], max_new_tokens)
        sequence['token_ids'] = []
        prompt_token_ids = request['prompt_token_ids']
        if sequence['budget'] == 0:
            sequence['done'] = True
        else:
            sequence['done'] = False
            logits, cache = model_prefill(prompt_token_ids, params)
            sequence['kv_cache'] = cache

            t = sampling_config.get('temperature')
            if sampling_config.get('greedy', False) or (t is not None and t <= 0):
                next_token_id = greedy_select(logits)
            else:
                if t is not None:
                    logits = apply_temperature(logits, t)
                if sampling_config.get('top_k', 0) > 0:
                    logits = top_k_filter(logits, sampling_config['top_k'])
                if sampling_config.get('top_p', 1.0) < 1.0:
                    logits = top_p_filter(logits, sampling_config['top_p'])
                next_token_id = sample_from_probs(stable_softmax(logits), rng)
            sequence['token_ids'].append(next_token_id)
        sequences.append(sequence)

    remaining = sum(1 for seq in sequences if not seq['done'])
    while remaining > 0:
        for seq in sequences:
            if not seq['done'] and len(seq['token_ids']) >= seq['budget']:
                seq['done'] = True
                remaining -= 1
        if remaining == 0:
            break
        sequences = batched_decode_step(params, sequences, sampling_config, rng=rng)
    
    output = []
    for seq in sequences:
        output.append({'request_id': seq['request_id'], 'output_ids': seq['token_ids']})
    
    return output

# Step 34 - has_free_capacity
def has_free_capacity(allocator, required_blocks):
    if len(allocator['free_list']) >= required_blocks:
        return True
    else:
        return False

# Step 35 - continuous_batch_step
def continuous_batch_step(params, running, allocator, sampling_config):
    """Advance every active sequence in `running` by one decoded token using the paged allocator."""
    rng = sampling_config.get('rng')
    if rng is None:
        rng = np.random.default_rng()

    for req in running:

        if req['done']:
            continue
        if len(req['generated']) >= req['max_new_tokens']:
            req['done'] = True
            continue
        seq_id = req['request_id']
        last_token_id = req['token_ids'][-1]
        last_token_embedding = embed_tokens([last_token_id], params['embedding'])
        q = linear_projection(last_token_embedding, params['Wq'])
        k_new = linear_projection(last_token_embedding, params['Wk'])
        v_new = linear_projection(last_token_embedding, params['Wv'])
        append_to_paged_cache(allocator, seq_id, k_new, v_new)
        output = paged_attention_step(q, allocator, seq_id)
        final_output = linear_projection(output, params['Wo'])
        logits = linear_projection(final_output, params['W_out'])
        logits = logits[0]

        t = sampling_config.get('temperature')
        if sampling_config.get('greedy', False) or (t is not None and t <= 0):
            next_token_id = greedy_select(logits)
        else:
            if t is not None:
                logits = apply_temperature(logits, t)
            if sampling_config.get('top_k', 0) > 0:
                logits = top_k_filter(logits, sampling_config['top_k'])
            if sampling_config.get('top_p', 1.0) < 1.0:
                logits = top_p_filter(logits, sampling_config['top_p'])
            next_token_id = sample_from_probs(stable_softmax(logits), rng)
        
        req['token_ids'].append(next_token_id)
        req['generated'].append(next_token_id)
        req['length'] += 1

        new_tokens = len(req['generated'])
        if (new_tokens >= req['max_new_tokens'] or req['token_ids'][-1] == sampling_config.get('eos_token_id')):
            req['done'] = True
    
    return running

# Step 36 - run_continuous_batching
def run_continuous_batching(params, requests, allocator, sampling_config, max_steps):
    """Admit requests while KV blocks allow, decode the running batch one token per step,
    retire finished sequences (freeing their blocks), and return their completions."""
    block_size = allocator['block_size']
    waiting = sorted(requests, key=lambda r: r.get('priority', 0))
    running = []
    completed = []
    steps = 0

    while (waiting or running) and steps < max_steps:
        owed = sum(seq['reserved'] - len(allocator['seq_tables'].get(seq['request_id'], []))
                   for seq in running)

        while waiting:
            req = waiting[0]
            prompt = list(req['prompt_token_ids'])
            need = blocks_needed(len(prompt) + req['max_new_tokens'], block_size)
            if not has_free_capacity(allocator, need + owed):
                break
            waiting.pop(0)

            if not prompt:
                completed.append({'request_id': req['request_id'], 'output_ids': []})
                continue

            seq_id = req['request_id']
            if len(prompt) > 1:
                emb = embed_tokens(prompt[:-1], params['embedding'])
                append_to_paged_cache(allocator, seq_id,
                                      linear_projection(emb, params['Wk']),
                                      linear_projection(emb, params['Wv']))

            running.append({
                'request_id': seq_id,
                'token_ids': prompt,
                'generated': [],
                'length': len(prompt),
                'done': False,
                'max_new_tokens': req['max_new_tokens'],
                'reserved': need,
            })
            owed += need - len(allocator['seq_tables'].get(seq_id, []))

        if not running:
            break

        running = continuous_batch_step(params, running, allocator, sampling_config)
        steps += 1

        still_running = []
        for seq in running:
            if seq['done']:
                free_sequence_blocks(allocator, seq['request_id'])
                completed.append({'request_id': seq['request_id'], 'output_ids': seq['generated']})
            else:
                still_running.append(seq)
        running = still_running

    for seq in running:
        free_sequence_blocks(allocator, seq['request_id'])
        completed.append({'request_id': seq['request_id'], 'output_ids': seq['generated']})

    return completed

# Step 37 - priority_queue_push
import heapq

def priority_queue_push(heap, priority, request):
    counter = len(heap)
    heapq.heappush(heap, (priority, counter, request))
    return heap

# Step 38 - priority_queue_pop
import heapq

def priority_queue_pop(heap):
    
    if len(heap) > 0:
        priority, counter, request = heapq.heappop(heap)
        return request
    return None

# Step 39 - select_admissions
def select_admissions(waiting_heap, allocator, block_size, max_admit):
    count = 0
    admitted = []
    blocks_req = 0
    while len(waiting_heap) > 0 and count < max_admit:
        request = waiting_heap[0][-1]
        total_tokens = len(request['prompt_token_ids'])
        blocks_req += blocks_needed(total_tokens, block_size)
        if has_free_capacity(allocator, blocks_req):
            admitted.append(request)
            priority_queue_pop(waiting_heap)
        else:
            break
        count += 1
    return admitted

# Step 40 - preempt_sequence
def preempt_sequence(sequence, allocator, waiting_heap):
    free_sequence_blocks(allocator, sequence['request_id'])
    request = {'request_id': sequence['request_id'], 'prompt_token_ids': sequence['prompt_token_ids'], 
    'max_new_tokens': sequence['max_new_tokens'], 'priority': sequence['priority']}
    priority_queue_push(waiting_heap, sequence['priority'], request)
    return request

# Step 41 - schedule_step
def schedule_step(waiting_heap, running, allocator, block_size, max_running):

    while len(running) > max_running:
        request = running.pop()
        preempt_sequence(request, allocator, waiting_heap)
    
    max_admit = max_running - len(running)
    
    newly_admitted = select_admissions(waiting_heap, allocator, block_size, max_admit)

    return {'running': running, 'newly_admitted': newly_admitted}

# Step 42 - format_stream_chunk
def format_stream_chunk(request_id, token_id, token_text, finished):
    return {'request_id': request_id, 'token_id': token_id, 'text': token_text,
    'finished': finished}

# Step 43 - submit_request
def submit_request(server_state, prompt, max_new_tokens, priority, vocab):
    prompt_token_ids = encode_prompt(prompt, vocab)
    request_id = 'req-{}'.format(server_state['next_request_id'])
    request = {'request_id': request_id, 'prompt_token_ids': prompt_token_ids, 'max_new_tokens': max_new_tokens, 'priority': priority}
    priority_queue_push(server_state['waiting_heap'], priority, request)
    server_state['next_request_id'] += 1
    return request_id

# Step 44 - drive_until_complete
def drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps):
    steps = 0
    eos_token_id = server_state.get('eos_token_id', sampling_config.get('eos_token_id', -1))
    server_state.setdefault("running", [])
    server_state.setdefault("completed", {})
    server_state.setdefault("streams", {})
    server_state.setdefault("waiting_heap", [])
    block_size = allocator.get('block_size')
    max_running = server_state.get('max_running', 4)
    output = []
    while (server_state['waiting_heap'] or server_state['running']) and steps < max_steps:
        hashmap = schedule_step(server_state['waiting_heap'], server_state['running'], allocator, block_size, max_running)
        server_state['running'] = hashmap['running']
        newly_admitted = hashmap['newly_admitted']
        if len(server_state['running']) == 0 and len(newly_admitted) == 0:
            break
        for admit in newly_admitted:
            admit['done'] = False
            admit['generated'] = []
            admit['token_ids'] = list(admit['prompt_token_ids'])
            admit['length'] = 0
            seq_id = admit['request_id']
            prompt_token_ids = admit['prompt_token_ids']
            if len(prompt_token_ids) == 0:
                server_state['completed'][seq_id] = {'request_id': seq_id, 'output_ids': [], 
                'chunks': [], 'finish_reason' : 'length'}
                server_state['streams'][seq_id] = []
                admit['done'] = True
                continue
            prompt_token_ids = prompt_token_ids[:-1]
            if len(prompt_token_ids) == 0:
                server_state['running'].append(admit)
                continue
            embeddings = embed_tokens(prompt_token_ids, params['embedding'])
            Wk, Wv = params['Wk'], params['Wv']
            k_new = linear_projection(embeddings, Wk)
            v_new = linear_projection(embeddings, Wv)
            append_to_paged_cache(allocator, seq_id, k_new, v_new)
            server_state['running'].append(admit)
        
        server_state['running'] = continuous_batch_step(params, server_state['running'], allocator, sampling_config)
        running = []
        for req in server_state['running']:
            seq_id = req['request_id']
            if len(req['generated']) == 0:
                server_state['completed'][seq_id] = {'request_id': seq_id, 'output_ids': req['generated'], 
                'chunks': [], 'finish_reason': 'length'}
                server_state['streams'][seq_id] = []
                free_sequence_blocks(allocator, seq_id)
                continue
            latest_token_id = req['generated'][-1]
            latest_token_text = vocab['id_to_token'][latest_token_id]
            latest_chunk = format_stream_chunk(seq_id, latest_token_id, latest_token_text, req['done'])
            output.append(latest_chunk)
            if seq_id not in server_state['streams']:
                server_state['streams'][seq_id] = []
            server_state['streams'][seq_id].append(latest_chunk)
            if req['done']:
                if latest_token_id == eos_token_id:
                    finish_reason = 'stop'
                else:
                    finish_reason = 'length'
                server_state['completed'][seq_id] = {'request_id': seq_id, 'output_ids': req['generated'], 
                'chunks': server_state['streams'][seq_id], 'finish_reason' : finish_reason}
                free_sequence_blocks(allocator, seq_id)
            else:
                running.append(req)
        server_state['running'] = running
        steps += 1
    
    return output

# Step 45 - collect_request_output
def collect_request_output(server_state, request_id):
    if request_id not in server_state.get('completed', {}):
        return None
    output = {}
    output['request_id'] = request_id
    output['output_ids'] = server_state['completed'][request_id]['output_ids']
    output['chunks'] = server_state['completed'][request_id]['chunks']
    return output

# Step 46 - build_completion_response
def build_completion_response(server_state, request_id, vocab):
    output = collect_request_output(server_state, request_id)
    if not output:
        return None
    
    text = decode_tokens(output['output_ids'], vocab)
    return {'request_id': request_id, 'text': text, 'output_ids': list(output['output_ids']),
    'finish_reason': server_state['completed'][request_id].get('finish_reason', 'stop')}

# Step 47 - time_to_first_token
def time_to_first_token(events):
    l = len(events)
    submits = {}
    minimum_time = {}
    for i in range(l):
        if events[i]['event'] == 'submit':
            submits[events[i]['request_id']] = events[i]['time']
        elif events[i]['event'] == 'token':
            request_id = events[i]['request_id']
            if request_id not in minimum_time:
                minimum_time[request_id] = float('inf')
            minimum_time[request_id] = min(minimum_time[request_id], events[i]['time'])
    
    output = {}
    for r_id in submits.keys():
        if r_id in minimum_time:
            output[r_id] = minimum_time[r_id] - submits[r_id]
    
    return output

# Step 48 - inter_token_latency
def inter_token_latency(events):
    times = {}
    for i in range(len(events)):
        if events[i]['event'] == 'token':
            r_id = events[i]['request_id']
            if r_id not in times:
                times[r_id] = []
            times[r_id].append(events[i]['time'])
    
    output = {}
    for key in times.keys():
        arr = times[key]
        arr.sort()
        if len(arr) <= 1:
            output[key] = 0.0
            continue
        count = 0
        total = 0
        for i in range(len(arr) - 1):
            total += arr[i+1] - arr[i]
            count += 1
        output[key] = total/count
    
    return output

# Step 49 - aggregate_throughput
def aggregate_throughput(events, total_time):
    total_tokens = 0
    total_requests = 0
    for event in events:
        if event['type'] == 'first_token' or event['type'] == 'token':
            total_tokens += 1
        elif event['type'] == 'finish':
            total_requests += 1
    
    output = {'tokens_per_second': total_tokens/total_time, 'requests_per_second': total_requests/total_time,
    'total_tokens': total_tokens, 'total_requests': total_requests}

    return output

# Step 50 - latency_percentiles
def latency_percentiles(latencies, percentiles):
    if len(latencies) == 0:
        return {float(p): 0.0 for p in percentiles}
    
    output = {}
    for p in percentiles:
        output[float(p)] = float(np.percentile(latencies, p))

    return output

# Step 51 - run_throughput_latency_benchmark
import time
def run_throughput_latency_benchmark(params, allocator, vocab, prompts, sampling_config,
                                     max_new_tokens, max_steps):
    server_state = {
        'running': [],
        'completed': {},
        'stream_buffer': {},
        'waiting_heap': [],
        'next_request_id': 0,
    }
    events = []
    t0 = time.perf_counter()

    for prompt in prompts:
        r_id = submit_request(server_state, prompt, max_new_tokens, priority, vocab)
        events.append({'request_id': r_id, 'event': 'submit', 'type': submit, 'time': time.perf_counter() - t0})

    seen = set()
    for step in range(max_steps):
        chunks = drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps=1)
        for chunk in chunks:
            r_id = chunk['request_id']
            if r_id in seen:
                if chunk['finished']:
                    event = {'request_id': r_id, 'event': 'token', 'type': 'finish', 'time': time.perf_counter() - t0}
                else:
                    event = {'request_id': r_id, 'event': 'token', 'type': 'token', 'time': time.perf_counter() - t0}
            else:
                if chunk['finished']:
                    event = {'request_id': r_id, 'event': 'first_token', 'type': 'finish', 'time': time.perf_counter() - t0}
                else:
                    event = {'request_id': r_id, 'event': 'first_token', 'type': 'token', 'time': time.perf_counter() - t0}
                    seen.add(r_id)
            events.append(event)
    
    total_time = time.perf_counter() - t0
    ttft = time_to_first_token(events)
    latencies = [ttft[key] for key in ttft.keys()]
    itl = inter_token_latency(events)
    throughput = aggregate_throughput(events, total_time)
    percentiles = latency_percentiles(latencies, sampling_config.get('percentiles', [50, 90, 99]))
    output = {'ttft': ttft, 'itl': itl, 'throughput': throughput, 'percentiles': percentiles,
    'total_time': total_time}

    return output

