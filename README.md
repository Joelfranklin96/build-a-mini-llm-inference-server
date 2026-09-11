# Build a Mini LLM Inference Server

Construct a complete LLM inference stack from scratch: sampling, tokenization, a tiny transformer with KV caching, a paged attention allocator, continuous batching with scheduling, a streaming serving API, and a throughput/latency benchmark harness. This mirrors the architecture of modern serving systems like vLLM at a digestible scale.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** stable_softmax
- [ ] **2.** apply_temperature
- [ ] **3.** top_k_filter
- [ ] **4.** top_p_filter
- [ ] **5.** sample_from_probs
- [ ] **6.** greedy_select
- [ ] **7.** build_vocab
- [ ] **8.** encode_prompt
- [ ] **9.** decode_tokens
- [ ] **10.** embed_tokens
- [ ] **11.** linear_projection
- [ ] **12.** init_kv_cache
- [ ] **13.** append_kv
- [ ] **14.** causal_attention
- [ ] **15.** model_prefill
- [ ] **16.** model_decode_step
- [ ] **17.** blocks_needed
- [ ] **18.** init_block_allocator
- [ ] **19.** allocate_block
- [ ] **20.** free_block
- [ ] **21.** append_to_paged_cache
- [ ] **22.** gather_kv_from_blocks
- [ ] **23.** paged_attention_step
- [ ] **24.** free_sequence_blocks
- [ ] **25.** kv_blocks_in_use
- [ ] **26.** make_request
- [ ] **27.** init_sequence_state
- [ ] **28.** sequence_decode_step
- [ ] **29.** is_sequence_done
- [ ] **30.** generate_single_sequence
- [ ] **31.** build_batch_step_input
- [ ] **32.** batched_decode_step
- [ ] **33.** static_batch_generate
- [ ] **34.** has_free_capacity
- [ ] **35.** continuous_batch_step
- [ ] **36.** run_continuous_batching
- [ ] **37.** priority_queue_push
- [ ] **38.** priority_queue_pop
- [ ] **39.** select_admissions
- [ ] **40.** preempt_sequence
- [ ] **41.** schedule_step
- [ ] **42.** format_stream_chunk
- [ ] **43.** submit_request
- [ ] **44.** drive_until_complete
- [ ] **45.** collect_request_output
- [ ] **46.** build_completion_response
- [ ] **47.** time_to_first_token
- [ ] **48.** inter_token_latency
- [ ] **49.** aggregate_throughput
- [ ] **50.** latency_percentiles
- [ ] **51.** run_throughput_latency_benchmark

---

Built on Deep-ML.
