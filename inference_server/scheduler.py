"""Priority queueing, capacity-aware admission control, and preemption."""

import heapq

from .paged_cache import blocks_needed, free_sequence_blocks, has_free_capacity


def priority_queue_push(heap, priority, request):
    """Push a request onto the waiting heap, breaking priority ties by insertion order."""
    heapq.heappush(heap, (priority, len(heap), request))

    return heap


def priority_queue_pop(heap):
    """Pop the highest-priority waiting request, or ``None`` when the heap is empty."""
    if not heap:
        return None

    return heapq.heappop(heap)[-1]


def select_admissions(waiting_heap, allocator, block_size, max_admit):
    """Admit queued requests while the allocator still has room for their prompt blocks."""
    admitted = []
    required_blocks = 0

    while waiting_heap and len(admitted) < max_admit:
        request = waiting_heap[0][-1]
        required_blocks += blocks_needed(len(request['prompt_token_ids']), block_size)
        if not has_free_capacity(allocator, required_blocks):
            break

        admitted.append(priority_queue_pop(waiting_heap))

    return admitted


def preempt_sequence(sequence, allocator, waiting_heap):
    """Evict a running sequence, free its blocks, and requeue it at its original priority."""
    free_sequence_blocks(allocator, sequence['request_id'])
    request = {
        'request_id': sequence['request_id'],
        'prompt_token_ids': sequence['prompt_token_ids'],
        'max_new_tokens': sequence['max_new_tokens'],
        'priority': sequence['priority'],
    }
    priority_queue_push(waiting_heap, sequence['priority'], request)

    return request


def schedule_step(waiting_heap, running, allocator, block_size, max_running):
    """Preempt sequences beyond the concurrency limit, then admit as many waiting ones as fit."""
    while len(running) > max_running:
        preempt_sequence(running.pop(), allocator, waiting_heap)

    newly_admitted = select_admissions(waiting_heap, allocator, block_size,
                                       max_running - len(running))

    return {'running': running, 'newly_admitted': newly_admitted}
