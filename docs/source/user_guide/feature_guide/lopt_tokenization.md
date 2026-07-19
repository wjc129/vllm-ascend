# Lossless overlapping parallel tokenization (experimental)

LoPT reduces CPU tokenization latency for a single long text prompt by
tokenizing overlapping character chunks in a thread pool and merging only
tokens that agree on token ID and global character offsets.

This feature is experimental and disabled by default. It is currently limited
to text-only Hugging Face Fast Tokenizers. Slow tokenizers, multimodal models,
invalid offset mappings, ambiguous overlap matches, and worker failures use the
standard vLLM tokenizer path automatically.

## Usage

Set the environment variable before starting vLLM:

```bash
VLLM_ASCEND_LOPT_ENABLE=1 vllm serve <model>
```

The default configuration is:

| Environment variable | Default | Description |
|---|---:|---|
| `VLLM_ASCEND_LOPT_THREAD_WORKERS` | `4` | Chunk tokenizer threads per renderer |
| `VLLM_ASCEND_LOPT_MIN_CHARS` | `32768` | Minimum prompt length in Unicode characters |
| `VLLM_ASCEND_LOPT_CHUNK_CHARS` | `32768` | Non-overlapping chunk body length |
| `VLLM_ASCEND_LOPT_OVERLAP_CHARS` | `512` | Adjacent chunk overlap length |
| `VLLM_ASCEND_LOPT_MIN_MATCH_TOKENS` | `2` | Minimum position-identical overlap tokens |
| `VLLM_ASCEND_LOPT_MAX_RETRIES` | `3` | Retries that double the chunk body length |
| `VLLM_ASCEND_LOPT_VERIFY` | `0` | Compare every LoPT result with standard tokenization |

When LoPT is enabled, vLLM Ascend defaults `RAYON_NUM_THREADS` to `1` and
`TOKENIZERS_PARALLELISM` to `false` unless the user already set them. This
avoids multiplying the outer LoPT pool by the tokenizer's internal Rayon pool.

Use `VLLM_ASCEND_LOPT_VERIFY=1` during tokenizer qualification. It tokenizes
the prompt through both paths and returns the standard result if they differ,
so it validates correctness but removes the expected tokenization speedup.

Tune worker, chunk, overlap, and threshold values on the target host. LoPT
should remain disabled when it does not provide a stable latency improvement
for the deployed tokenizer and prompt distribution.
