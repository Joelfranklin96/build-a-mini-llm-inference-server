"""End-to-end throughput and latency benchmark driven through the serving API."""

import time

from .metrics import (
    aggregate_throughput,
    inter_token_latency,
    latency_percentiles,
    time_to_first_token,
)
from .server import drive_until_complete, submit_request


def run_throughput_latency_benchmark(params, allocator, vocab, prompts, sampling_config,
                                     max_new_tokens, max_steps):
    """Submit prompts, drive the server one step at a time, and report latency and throughput."""
    server_state = {
        'running': [],
        'completed': {},
        'streams': {},
        'waiting_heap': [],
        'next_request_id': 0,
    }

    events = []
    start = time.perf_counter()

    for priority, prompt in enumerate(prompts):
        request_id = submit_request(server_state, prompt, max_new_tokens, priority, vocab)
        events.append({'request_id': request_id, 'event': 'submit', 'type': 'submit',
                       'time': time.perf_counter() - start})

    for _ in range(max_steps):
        chunks = drive_until_complete(server_state, params, allocator, sampling_config,
                                      vocab, max_steps=1)
        for chunk in chunks:
            now = time.perf_counter() - start
            events.append({'request_id': chunk['request_id'], 'event': 'token',
                           'type': 'token', 'time': now})
            if chunk['finished']:
                events.append({'request_id': chunk['request_id'], 'event': 'finish',
                               'type': 'finish', 'time': now})

        if not server_state['waiting_heap'] and not server_state['running']:
            break

    total_time = time.perf_counter() - start
    ttft = time_to_first_token(events)

    return {
        'ttft': ttft,
        'itl': inter_token_latency(events),
        'throughput': aggregate_throughput(events, total_time),
        'percentiles': latency_percentiles(list(ttft.values()),
                                           sampling_config.get('percentiles', [50, 90, 99])),
        'total_time': total_time,
    }
