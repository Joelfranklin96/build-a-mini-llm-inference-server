"""Decoding-time token selection: softmax, temperature, top-k/top-p filtering, and sampling."""

import numpy as np


def stable_softmax(logits):
    """Softmax over the last axis, shifted by the row max for numerical stability."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def apply_temperature(logits, temperature):
    """Scale logits by a sampling temperature; non-positive values leave them unchanged."""
    if temperature > 0:
        return logits / temperature
    return logits


def top_k_filter(logits, k):
    """Mask every logit outside the top ``k`` of each row to ``-inf``."""
    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits

    kth_largest = np.partition(logits, -k, axis=-1)[..., vocab_size - k]
    kth_largest = np.expand_dims(kth_largest, axis=-1)

    return np.where(logits >= kth_largest, logits, -np.inf).astype(logits.dtype, copy=False)


def top_p_filter(logits, p):
    """Mask logits outside the smallest nucleus whose cumulative probability reaches ``p``."""
    sorted_indices = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)

    sorted_probs = stable_softmax(sorted_logits)
    cumulative = np.cumsum(sorted_probs, axis=-1)

    keep_sorted = (cumulative - sorted_probs) < p
    keep_sorted[..., 0] = True

    keep = np.empty_like(keep_sorted)
    np.put_along_axis(keep, sorted_indices, keep_sorted, axis=-1)

    return np.where(keep, logits, -np.inf).astype(logits.dtype, copy=False)


def sample_from_probs(probs, rng):
    """Draw a token index from a probability vector by inverse-CDF sampling."""
    cumulative = np.cumsum(probs)
    index = int(np.searchsorted(cumulative, rng.random(), side='right'))
    if index < len(cumulative):
        return index

    non_zero = np.flatnonzero(np.asarray(probs) > 0)
    return int(non_zero[-1]) if non_zero.size else len(cumulative) - 1


def greedy_select(logits):
    """Return the index of the highest logit."""
    return int(np.argmax(logits))


def resolve_rng(rng, sampling_params):
    """Pick the first available generator: the argument, the sampling config, or a fresh one."""
    if rng is None:
        rng = sampling_params.get('rng')
    if rng is None:
        rng = np.random.default_rng()
    return rng


def select_next_token(logits, sampling_params, rng):
    """Choose the next token id: greedy when requested, else temperature/top-k/top-p sampling."""
    temperature = sampling_params.get('temperature')
    if sampling_params.get('greedy', False) or (temperature is not None and temperature <= 0):
        return greedy_select(logits)

    if temperature is not None:
        logits = apply_temperature(logits, temperature)

    top_k = sampling_params.get('top_k', 0)
    if top_k > 0:
        logits = top_k_filter(logits, top_k)

    top_p = sampling_params.get('top_p', 1.0)
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    return sample_from_probs(stable_softmax(logits), rng)
