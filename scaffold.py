"""
Build a Mini LLM Inference Server scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo for the Mini LLM Inference Server project."""

import numpy as np
import time


def _make_params(vocab_size, d_model, max_seq_len, rng):
    scale = 1.0 / np.sqrt(d_model)
    return {
        "embedding": rng.standard_normal((vocab_size, d_model)) * scale,
        "Wq": rng.standard_normal((d_model, d_model)) * scale,
        "Wk": rng.standard_normal((d_model, d_model)) * scale,
        "Wv": rng.standard_normal((d_model, d_model)) * scale,
        "Wo": rng.standard_normal((d_model, d_model)) * scale,
        "W_out": rng.standard_normal((d_model, vocab_size)) * scale,
        "d_model": d_model,
        "max_seq_len": max_seq_len,
    }


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # ---- Sampling primitives sanity check ----
    raw_logits = np.array([[2.0, 1.0, 0.1, -1.0, 3.0]])
    tempered = apply_temperature(raw_logits, 0.8)
    topk = top_k_filter(tempered, k=3)
    topp = top_p_filter(topk, p=0.9)
    probs = stable_softmax(topp)
    greedy_tok = greedy_select(raw_logits)
    sampled_tok = sample_from_probs(probs[0], rng)
    print(f"[sampling] greedy={greedy_tok}, sampled={sampled_tok}, probs={probs[0]}")

    # ---- Vocab / tokenizer ----
    corpus = [
        "hello world this is a tiny llm",
        "the quick brown fox jumps over the lazy dog",
        "paged attention enables continuous batching",
    ]
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = build_vocab(corpus, specials)
    bos_id = vocab["token_to_id"]["<bos>"]
    eos_id = vocab["token_to_id"]["<eos>"]
    vocab_size = len(vocab["token_to_id"])
    print(f"[vocab] size={vocab_size}, bos={bos_id}, eos={eos_id}")

    prompt = "hello world"
    prompt_ids = encode_prompt(prompt, vocab, add_bos=True)
    roundtrip = decode_tokens(prompt_ids, vocab, skip_special=True)
    print(f"[tokenize] prompt={prompt!r} ids={prompt_ids} roundtrip={roundtrip!r}")

    # ---- Model params ----
    d_model = 16
    max_seq_len = 64
    params = _make_params(vocab_size, d_model, max_seq_len, rng)

    # ---- Single-sequence generate (contiguous KV cache) ----
    sampling_cfg = {"temperature": 1.0, "top_k": 5, "top_p": 0.9, "rng": rng}
    request = make_request("req-solo", prompt_ids, max_new_tokens=6,
                           sampling_params=sampling_cfg)
    solo_tokens = generate_single_sequence(request, params, eos_id, rng)
    print(f"[single] generated ids={solo_tokens} text={decode_tokens(solo_tokens, vocab)!r}")

    # ---- Paged block allocator + continuous batching server ----
    block_size = 8
    num_blocks = 32
    allocator = init_block_allocator(num_blocks, block_size, d_model)
    print(f"[allocator] blocks={num_blocks}, block_size={block_size}, "
          f"usage={kv_blocks_in_use(allocator)}")

    server_state = {
        "waiting_heap": [],
        "running": [],
        "next_request_id": 0,
        "completed": {},
        "streams": {},
        "events": [],
        "outputs": {},
        "eos_token_id": eos_id,
        "block_size": block_size,
        "max_running": 4,
        "rng": rng,
        "tick": 0,
    }

    prompts = [
        "hello world",
        "the quick brown",
        "paged attention",
        "tiny llm",
    ]
    req_ids = []
    for i, p in enumerate(prompts):
        rid = submit_request(server_state, p, max_new_tokens=5,
                             priority=i, vocab=vocab)
        req_ids.append(rid)
    print(f"[server] submitted requests: {req_ids}")

    drive_until_complete(server_state, params, allocator, sampling_cfg,
                         vocab, max_steps=64)

    for rid in req_ids:
        out = collect_request_output(server_state, rid)
        resp = build_completion_response(server_state, rid, vocab)
        out = out or {}
        resp = resp or {}
        print(f"[server] {rid}: tokens={out.get('output_ids', [])} "
              f"text={resp.get('text', '')!r}")

    print(f"[allocator] post-run usage={kv_blocks_in_use(allocator)}")

    # ---- Benchmark harness ----
    bench_allocator = init_block_allocator(num_blocks, block_size, d_model)
    bench_prompts = ["hello world", "tiny llm", "the quick brown"]
    t0 = time.time()
    report = run_throughput_latency_benchmark(
        params, bench_allocator, vocab, bench_prompts,
        sampling_cfg, max_new_tokens=5, max_steps=64,
    )
    t1 = time.time()
    print(f"[bench] wall={t1 - t0:.4f}s report keys={list(report.keys())}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
