"""Request submission, the schedule-then-decode driver loop, and streamed responses."""

from .batching import continuous_batch_step
from .paged_cache import append_to_paged_cache, free_sequence_blocks
from .scheduler import priority_queue_push, schedule_step
from .tokenizer import decode_tokens, encode_prompt
from .transformer import embed_tokens, linear_projection


def format_stream_chunk(request_id, token_id, token_text, finished):
    """Build a single streaming response chunk."""
    return {'request_id': request_id, 'token_id': token_id, 'text': token_text,
            'finished': finished}


def submit_request(server_state, prompt, max_new_tokens, priority, vocab):
    """Tokenize a prompt, enqueue it on the waiting heap, and return its generated request id."""
    request_id = 'req-{}'.format(server_state['next_request_id'])
    request = {
        'request_id': request_id,
        'prompt_token_ids': encode_prompt(prompt, vocab),
        'max_new_tokens': max_new_tokens,
        'priority': priority,
    }

    priority_queue_push(server_state['waiting_heap'], priority, request)
    server_state['next_request_id'] += 1

    return request_id


def drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps):
    """Run schedule-then-decode rounds until the queues drain or the step budget is exhausted,
    returning every stream chunk emitted along the way."""
    server_state.setdefault('running', [])
    server_state.setdefault('completed', {})
    server_state.setdefault('streams', {})
    server_state.setdefault('waiting_heap', [])

    eos_token_id = server_state.get('eos_token_id', sampling_config.get('eos_token_id', -1))
    block_size = allocator['block_size']
    max_running = server_state.get('max_running', 4)

    emitted_chunks = []
    steps = 0

    while (server_state['waiting_heap'] or server_state['running']) and steps < max_steps:
        schedule = schedule_step(server_state['waiting_heap'], server_state['running'],
                                 allocator, block_size, max_running)
        server_state['running'] = schedule['running']
        newly_admitted = schedule['newly_admitted']

        if not server_state['running'] and not newly_admitted:
            break

        for request in newly_admitted:
            seq_id = request['request_id']
            prompt_token_ids = request['prompt_token_ids']
            request['done'] = False
            request['generated'] = []
            request['token_ids'] = list(prompt_token_ids)
            request['length'] = 0

            if not prompt_token_ids:
                server_state['completed'][seq_id] = {
                    'request_id': seq_id,
                    'output_ids': [],
                    'chunks': [],
                    'finish_reason': 'length',
                }
                server_state['streams'][seq_id] = []
                request['done'] = True
                continue

            prefill_token_ids = prompt_token_ids[:-1]
            if prefill_token_ids:
                embeddings = embed_tokens(prefill_token_ids, params['embedding'])
                append_to_paged_cache(allocator, seq_id,
                                      linear_projection(embeddings, params['Wk']),
                                      linear_projection(embeddings, params['Wv']))

            server_state['running'].append(request)

        server_state['running'] = continuous_batch_step(
            params, server_state['running'], allocator, sampling_config)

        still_running = []
        for request in server_state['running']:
            seq_id = request['request_id']

            if not request['generated']:
                server_state['completed'][seq_id] = {
                    'request_id': seq_id,
                    'output_ids': request['generated'],
                    'chunks': [],
                    'finish_reason': 'length',
                }
                server_state['streams'][seq_id] = []
                free_sequence_blocks(allocator, seq_id)
                continue

            token_id = request['generated'][-1]
            chunk = format_stream_chunk(seq_id, token_id, vocab['id_to_token'][token_id],
                                        request['done'])
            emitted_chunks.append(chunk)
            server_state['streams'].setdefault(seq_id, []).append(chunk)

            if request['done']:
                server_state['completed'][seq_id] = {
                    'request_id': seq_id,
                    'output_ids': request['generated'],
                    'chunks': server_state['streams'][seq_id],
                    'finish_reason': 'stop' if token_id == eos_token_id else 'length',
                }
                free_sequence_blocks(allocator, seq_id)
            else:
                still_running.append(request)

        server_state['running'] = still_running
        steps += 1

    return emitted_chunks


def collect_request_output(server_state, request_id):
    """Return the raw token ids and stream chunks of a finished request, or ``None``."""
    completion = server_state.get('completed', {}).get(request_id)
    if completion is None:
        return None

    return {
        'request_id': request_id,
        'output_ids': completion['output_ids'],
        'chunks': completion['chunks'],
    }


def build_completion_response(server_state, request_id, vocab):
    """Render a finished request as a decoded completion response, or ``None``."""
    output = collect_request_output(server_state, request_id)
    if not output:
        return None

    return {
        'request_id': request_id,
        'text': decode_tokens(output['output_ids'], vocab),
        'output_ids': list(output['output_ids']),
        'finish_reason': server_state['completed'][request_id].get('finish_reason', 'stop'),
    }
