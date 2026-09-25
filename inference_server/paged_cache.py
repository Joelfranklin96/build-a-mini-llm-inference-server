"""Block-paged KV cache: a fixed-size block pool with per-sequence block tables."""

import numpy as np

from .transformer import causal_attention


def blocks_needed(num_tokens, block_size):
    """Number of fixed-size blocks required to hold ``num_tokens`` KV entries."""
    return (num_tokens + block_size - 1) // block_size


def init_block_allocator(num_blocks, block_size, d_model):
    """Create a paged KV allocator backed by a pool of fixed-size key/value blocks."""
    return {
        'K_blocks': np.zeros((num_blocks, block_size, d_model), dtype=np.float32),
        'V_blocks': np.zeros((num_blocks, block_size, d_model), dtype=np.float32),
        'free_list': list(range(num_blocks)),
        'block_size': block_size,
        'num_blocks': num_blocks,
        'd_model': d_model,
        'seq_tables': {},
    }


def allocate_block(allocator, seq_id):
    """Reserve a free block for a sequence and return its block id."""
    if not allocator['free_list']:
        raise RuntimeError("out-of-memory")

    block_id = allocator['free_list'].pop()
    allocator['seq_tables'].setdefault(seq_id, []).append(block_id)

    return block_id


def free_block(allocator, block_id):
    """Return a block to the allocator's free list."""
    allocator['free_list'].append(block_id)


def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write new key/value rows into a sequence's paged blocks, allocating more as needed."""
    seq_lengths = allocator.setdefault('seq_lengths', {})
    length = seq_lengths.get(seq_id, 0)
    block_size = allocator['block_size']
    num_new = k_new.shape[0]

    written = 0
    while written < num_new:
        slot = (length + written) % block_size
        if slot == 0:
            allocate_block(allocator, seq_id)

        block_id = allocator['seq_tables'][seq_id][-1]
        count = min(block_size - slot, num_new - written)
        allocator['K_blocks'][block_id, slot:slot + count, :] = k_new[written:written + count, :]
        allocator['V_blocks'][block_id, slot:slot + count, :] = v_new[written:written + count, :]
        written += count

    seq_lengths[seq_id] = length + num_new


def gather_kv_from_blocks(allocator, seq_id):
    """Concatenate a sequence's paged blocks back into contiguous key/value matrices."""
    length = allocator.get('seq_lengths', {}).get(seq_id, 0)
    d_model = allocator['d_model']
    block_ids = allocator['seq_tables'][seq_id]

    keys = allocator['K_blocks'][block_ids].reshape(-1, d_model)[:length]
    values = allocator['V_blocks'][block_ids].reshape(-1, d_model)[:length]

    return keys, values


def paged_attention_step(q, allocator, seq_id):
    """Attend a single query against a sequence's paged KV cache."""
    keys, values = gather_kv_from_blocks(allocator, seq_id)

    return causal_attention(q, keys, values, is_causal=True).astype(np.float64)


def free_sequence_blocks(allocator, seq_id):
    """Release every block owned by a sequence and forget its cached length."""
    for block_id in allocator['seq_tables'].pop(seq_id, []):
        free_block(allocator, block_id)

    if 'seq_lengths' in allocator:
        allocator['seq_lengths'].pop(seq_id, None)


def kv_blocks_in_use(allocator):
    """Report used, free, and total block counts for the paged cache."""
    total = allocator['K_blocks'].shape[0]
    free = len(allocator['free_list'])

    return {'used': total - free, 'free': free, 'total': total}


def has_free_capacity(allocator, required_blocks):
    """Check whether the allocator can still hand out ``required_blocks`` blocks."""
    return len(allocator['free_list']) >= required_blocks
