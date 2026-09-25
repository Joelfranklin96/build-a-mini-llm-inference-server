"""Per-sequence decoding state plus the single, static-batch, and continuous-batch loops."""

import numpy as np

from .paged_cache import (
    append_to_paged_cache,
    blocks_needed,
    free_sequence_blocks,
    has_free_capacity,
    paged_attention_step,
)
from .sampling import resolve_rng, select_next_token
from .transformer import embed_tokens, linear_projection, model_decode_step, model_prefill


def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    """Package a generation request for the server."""
    return {
        'request_id': request_id,
        'prompt_token_ids': list(prompt_token_ids),
        'max_new_tokens': max_new_tokens,
        'sampling_params': sampling_params,
    }


def init_sequence_state(request, params):
    """Prefill a request and return the mutable per-sequence decoding state."""
    logits, cache = model_prefill(request['prompt_token_ids'], params)
    generated = []

    return {
        'request_id': request['request_id'],
        'prompt_token_ids': request['prompt_token_ids'],
        'generated': generated,
        'generated_token_ids': generated,
        'last_logits': logits,
        'cache': cache,
        'done': False,
        'sampling_params': request['sampling_params'],
        'max_new_tokens': request['max_new_tokens'],
    }


def sequence_decode_step(state, params, rng=None):
    """Emit one token for a sequence and advance its KV cache; returns the token and state."""
    sampling_params = state['sampling_params']
    rng = resolve_rng(rng, sampling_params)

    next_token_id = select_next_token(state['last_logits'], sampling_params, rng)
    state['generated'].append(next_token_id)
    state['last_logits'], state['cache'] = model_decode_step(next_token_id, state['cache'], params)

    return next_token_id, state


def is_sequence_done(state, eos_token_id):
    """Check whether a sequence hit its token budget or produced the end-of-sequence token."""
    generated = state['generated']
    if not generated:
        return state['max_new_tokens'] <= 0

    return len(generated) >= state['max_new_tokens'] or generated[-1] == eos_token_id


def generate_single_sequence(request, params, eos_token_id, rng):
    """Generate a full completion for one request using the contiguous KV cache."""
    state = init_sequence_state(request, params)
    while not is_sequence_done(state, eos_token_id):
        sequence_decode_step(state, params, rng)

    return state['generated']


def build_batch_step_input(sequences):
    """Collect the indices and most recent token ids of the still-active sequences."""
    active_indices = []
    input_ids = []

    for index, sequence in enumerate(sequences):
        if not sequence['done']:
            active_indices.append(index)
            input_ids.append(sequence['token_ids'][-1])

    return {'active_indices': active_indices, 'input_ids': np.array(input_ids, dtype=np.int64)}


def batched_decode_step(params, sequences, sampling_params, rng=None):
    """Advance every active sequence by one token in lockstep; mutates and returns the batch."""
    rng = resolve_rng(rng, sampling_params)

    for sequence in sequences:
        if sequence['done']:
            continue

        logits, sequence['kv_cache'] = model_decode_step(
            sequence['token_ids'][-1], sequence['kv_cache'], params)
        sequence['token_ids'].append(select_next_token(logits, sampling_params, rng))

    return sequences


def static_batch_generate(params, requests, sampling_config, max_new_tokens, rng=None):
    """Prefill a batch of requests, then decode in lockstep until every token budget is spent."""
    rng = resolve_rng(rng, sampling_config)

    sequences = []
    for request in requests:
        budget = min(request['max_new_tokens'], max_new_tokens)
        sequence = {
            'request_id': request['request_id'],
            'budget': budget,
            'token_ids': [],
            'done': budget == 0,
        }

        if not sequence['done']:
            logits, sequence['kv_cache'] = model_prefill(request['prompt_token_ids'], params)
            sequence['token_ids'].append(select_next_token(logits, sampling_config, rng))

        sequences.append(sequence)

    remaining = sum(1 for sequence in sequences if not sequence['done'])
    while remaining > 0:
        for sequence in sequences:
            if not sequence['done'] and len(sequence['token_ids']) >= sequence['budget']:
                sequence['done'] = True
                remaining -= 1

        if remaining == 0:
            break

        sequences = batched_decode_step(params, sequences, sampling_config, rng=rng)

    return [{'request_id': sequence['request_id'], 'output_ids': sequence['token_ids']}
            for sequence in sequences]


def continuous_batch_step(params, running, allocator, sampling_config):
    """Decode one token for every active sequence in the running batch via the paged cache."""
    rng = resolve_rng(None, sampling_config)
    eos_token_id = sampling_config.get('eos_token_id')

    for sequence in running:
        if sequence['done']:
            continue

        if len(sequence['generated']) >= sequence['max_new_tokens']:
            sequence['done'] = True
            continue

        seq_id = sequence['request_id']
        embedding = embed_tokens([sequence['token_ids'][-1]], params['embedding'])
        query = linear_projection(embedding, params['Wq'])

        append_to_paged_cache(allocator, seq_id,
                              linear_projection(embedding, params['Wk']),
                              linear_projection(embedding, params['Wv']))

        attended = paged_attention_step(query, allocator, seq_id)
        hidden = linear_projection(attended, params['Wo'])
        logits = linear_projection(hidden, params['W_out'])[0]

        next_token_id = select_next_token(logits, sampling_config, rng)
        sequence['token_ids'].append(next_token_id)
        sequence['generated'].append(next_token_id)
        sequence['length'] += 1

        if (len(sequence['generated']) >= sequence['max_new_tokens']
                or next_token_id == eos_token_id):
            sequence['done'] = True

    return running


def run_continuous_batching(params, requests, allocator, sampling_config, max_steps):
    """Admit requests as KV blocks allow, decode the running batch one token per step, and
    retire finished sequences until the queue drains or the step budget runs out."""
    block_size = allocator['block_size']
    waiting = sorted(requests, key=lambda request: request.get('priority', 0))
    running = []
    completed = []
    steps = 0

    while (waiting or running) and steps < max_steps:
        reserved_but_unallocated = sum(
            sequence['reserved'] - len(allocator['seq_tables'].get(sequence['request_id'], []))
            for sequence in running)

        while waiting:
            request = waiting[0]
            prompt_token_ids = list(request['prompt_token_ids'])
            required_blocks = blocks_needed(
                len(prompt_token_ids) + request['max_new_tokens'], block_size)

            if not has_free_capacity(allocator, required_blocks + reserved_but_unallocated):
                break
            waiting.pop(0)

            if not prompt_token_ids:
                completed.append({'request_id': request['request_id'], 'output_ids': []})
                continue

            seq_id = request['request_id']
            if len(prompt_token_ids) > 1:
                embeddings = embed_tokens(prompt_token_ids[:-1], params['embedding'])
                append_to_paged_cache(allocator, seq_id,
                                      linear_projection(embeddings, params['Wk']),
                                      linear_projection(embeddings, params['Wv']))

            running.append({
                'request_id': seq_id,
                'token_ids': prompt_token_ids,
                'generated': [],
                'length': len(prompt_token_ids),
                'done': False,
                'max_new_tokens': request['max_new_tokens'],
                'reserved': required_blocks,
            })
            reserved_but_unallocated += (
                required_blocks - len(allocator['seq_tables'].get(seq_id, [])))

        if not running:
            break

        running = continuous_batch_step(params, running, allocator, sampling_config)
        steps += 1

        still_running = []
        for sequence in running:
            if sequence['done']:
                free_sequence_blocks(allocator, sequence['request_id'])
                completed.append({'request_id': sequence['request_id'],
                                  'output_ids': sequence['generated']})
            else:
                still_running.append(sequence)
        running = still_running

    for sequence in running:
        free_sequence_blocks(allocator, sequence['request_id'])
        completed.append({'request_id': sequence['request_id'],
                          'output_ids': sequence['generated']})

    return completed
