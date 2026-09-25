"""Serving metrics derived from a request/token event log."""

import numpy as np


def time_to_first_token(events):
    """Per-request latency from the submit event to the earliest token event."""
    submitted_at = {}
    first_token_at = {}

    for event in events:
        request_id = event['request_id']
        if event['event'] == 'submit':
            submitted_at[request_id] = event['time']
        elif event['event'] == 'token':
            if request_id not in first_token_at or event['time'] < first_token_at[request_id]:
                first_token_at[request_id] = event['time']

    return {request_id: first_token_at[request_id] - submitted_at[request_id]
            for request_id in submitted_at if request_id in first_token_at}


def inter_token_latency(events):
    """Per-request mean gap between consecutive token events."""
    token_times = {}
    for event in events:
        if event['event'] == 'token':
            token_times.setdefault(event['request_id'], []).append(event['time'])

    latencies = {}
    for request_id, times in token_times.items():
        times.sort()
        if len(times) <= 1:
            latencies[request_id] = 0.0
            continue

        total = 0.0
        for index in range(len(times) - 1):
            total += times[index + 1] - times[index]
        latencies[request_id] = total / (len(times) - 1)

    return latencies


def aggregate_throughput(events, total_time):
    """Token and request throughput over an elapsed wall-clock window."""
    total_tokens = sum(1 for event in events if event['type'] in ('first_token', 'token'))
    total_requests = sum(1 for event in events if event['type'] == 'finish')

    return {
        'tokens_per_second': total_tokens / total_time,
        'requests_per_second': total_requests / total_time,
        'total_tokens': total_tokens,
        'total_requests': total_requests,
    }


def latency_percentiles(latencies, percentiles):
    """Requested percentiles of a latency sample, keyed by float percentile."""
    if len(latencies) == 0:
        return {float(percentile): 0.0 for percentile in percentiles}

    return {float(percentile): float(np.percentile(latencies, percentile))
            for percentile in percentiles}
