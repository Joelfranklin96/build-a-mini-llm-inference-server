"""End-to-end demo of the inference server: sampling, tokenization, single-sequence
generation, static batching, continuous batching, streaming serving, and benchmarking."""

import numpy as np

from inference_server import (
    apply_temperature,
    build_completion_response,
    build_vocab,
    collect_request_output,
    decode_tokens,
    drive_until_complete,
    encode_prompt,
    generate_single_sequence,
    greedy_select,
    init_block_allocator,
    init_model_params,
    kv_blocks_in_use,
    make_request,
    run_continuous_batching,
    run_throughput_latency_benchmark,
    sample_from_probs,
    stable_softmax,
    static_batch_generate,
    submit_request,
    top_k_filter,
    top_p_filter,
)

CORPUS = [
    "hello world this is a tiny llm",
    "the quick brown fox jumps over the lazy dog",
    "paged attention enables continuous batching",
]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PROMPTS = ["hello world", "the quick brown", "paged attention", "tiny llm"]

D_MODEL = 16
MAX_SEQ_LEN = 64
BLOCK_SIZE = 8
NUM_BLOCKS = 32


def demo_sampling(rng):
    """Run the decoding primitives over a handful of logits."""
    logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
    filtered = top_p_filter(top_k_filter(apply_temperature(logits, 0.8), k=3), p=0.9)
    probs = stable_softmax(filtered)

    print("[sampling] greedy={} sampled={} probs={}".format(
        greedy_select(logits), sample_from_probs(probs, rng), np.round(probs, 4)))


def demo_tokenizer():
    """Build the character vocabulary and show a prompt round-trip."""
    vocab = build_vocab(CORPUS, SPECIAL_TOKENS)
    prompt_ids = encode_prompt("hello world", vocab, add_bos=True)

    print("[tokenizer] vocab_size={} ids={} roundtrip={!r}".format(
        len(vocab['token_to_id']), prompt_ids, decode_tokens(prompt_ids, vocab)))

    return vocab


def demo_single_sequence(params, vocab, sampling_config, eos_token_id, rng):
    """Generate one completion with the contiguous KV cache."""
    prompt_ids = encode_prompt("hello world", vocab)
    request = make_request("req-solo", prompt_ids, max_new_tokens=6,
                           sampling_params=sampling_config)
    output_ids = generate_single_sequence(request, params, eos_token_id, rng)

    print("[single] ids={} text={!r}".format(output_ids, decode_tokens(output_ids, vocab)))


def demo_static_batching(params, vocab, sampling_config, rng):
    """Prefill a batch of prompts and decode them in lockstep."""
    requests = [
        make_request("static-{}".format(index), encode_prompt(prompt, vocab),
                     max_new_tokens=5, sampling_params=sampling_config)
        for index, prompt in enumerate(PROMPTS)
    ]
    outputs = static_batch_generate(params, requests, sampling_config, max_new_tokens=5, rng=rng)

    for output in outputs:
        print("[static] {}: text={!r}".format(
            output['request_id'], decode_tokens(output['output_ids'], vocab)))


def demo_continuous_batching(params, vocab, sampling_config):
    """Stream prompts through the paged allocator with rolling admission."""
    allocator = init_block_allocator(NUM_BLOCKS, BLOCK_SIZE, D_MODEL)
    requests = [
        {
            'request_id': "cont-{}".format(index),
            'prompt_token_ids': encode_prompt(prompt, vocab),
            'max_new_tokens': 5,
            'priority': index,
        }
        for index, prompt in enumerate(PROMPTS)
    ]
    outputs = run_continuous_batching(params, requests, allocator, sampling_config, max_steps=64)

    for output in outputs:
        print("[continuous] {}: text={!r}".format(
            output['request_id'], decode_tokens(output['output_ids'], vocab)))
    print("[continuous] allocator={}".format(kv_blocks_in_use(allocator)))


def demo_serving(params, vocab, sampling_config, eos_token_id, rng):
    """Submit prompts to the scheduler and collect streamed completions."""
    allocator = init_block_allocator(NUM_BLOCKS, BLOCK_SIZE, D_MODEL)
    server_state = {
        'waiting_heap': [],
        'running': [],
        'completed': {},
        'streams': {},
        'next_request_id': 0,
        'eos_token_id': eos_token_id,
        'max_running': 4,
        'rng': rng,
    }

    request_ids = [submit_request(server_state, prompt, 5, priority, vocab)
                   for priority, prompt in enumerate(PROMPTS)]
    chunks = drive_until_complete(server_state, params, allocator, sampling_config,
                                  vocab, max_steps=64)
    print("[serving] submitted={} streamed_chunks={}".format(request_ids, len(chunks)))

    for request_id in request_ids:
        output = collect_request_output(server_state, request_id) or {}
        response = build_completion_response(server_state, request_id, vocab) or {}
        print("[serving] {}: tokens={} reason={} text={!r}".format(
            request_id, output.get('output_ids', []),
            response.get('finish_reason', '-'), response.get('text', '')))

    print("[serving] allocator={}".format(kv_blocks_in_use(allocator)))


def demo_benchmark(params, vocab, sampling_config):
    """Measure time to first token, inter-token latency, and throughput."""
    allocator = init_block_allocator(NUM_BLOCKS, BLOCK_SIZE, D_MODEL)
    report = run_throughput_latency_benchmark(params, allocator, vocab, PROMPTS[:3],
                                              sampling_config, max_new_tokens=5, max_steps=64)
    throughput = report['throughput']

    print("[bench] total_time={:.4f}s tokens/s={:.1f} requests/s={:.1f}".format(
        report['total_time'], throughput['tokens_per_second'],
        throughput['requests_per_second']))
    print("[bench] ttft_p50={:.6f}s ttft_p90={:.6f}s".format(
        report['percentiles'][50.0], report['percentiles'][90.0]))
    print("[bench] mean_itl={:.6f}s".format(
        np.mean(list(report['itl'].values())) if report['itl'] else 0.0))


def main():
    """Run every stage of the inference server against a small random model."""
    rng = np.random.default_rng(0)

    demo_sampling(rng)
    vocab = demo_tokenizer()

    params = init_model_params(len(vocab['token_to_id']), D_MODEL, MAX_SEQ_LEN, rng)
    eos_token_id = vocab['token_to_id']['<eos>']
    sampling_config = {
        'temperature': 1.0,
        'top_k': 5,
        'top_p': 0.9,
        'eos_token_id': eos_token_id,
        'rng': rng,
    }

    demo_single_sequence(params, vocab, sampling_config, eos_token_id, rng)
    demo_static_batching(params, vocab, sampling_config, rng)
    demo_continuous_batching(params, vocab, sampling_config)
    demo_serving(params, vocab, sampling_config, eos_token_id, rng)
    demo_benchmark(params, vocab, sampling_config)


if __name__ == "__main__":
    main()
