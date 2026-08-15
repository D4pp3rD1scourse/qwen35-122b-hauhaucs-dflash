# Reproducibility record

## Exact target

- Model repository: `HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive`
- Quant tested: `Q4_K_M`
- Target GGUF size: approximately 74 GB
- Context: 262,144 tokens
- Parallel slots: 2
- Decoding: temperature 0, seed `20260712`, thinking disabled

The target weights must remain linked from HauhauCS. Do not copy the target GGUF into this release.

## Trained draft artifacts

| Artifact | SHA-256 |
|---|---|
| `epoch-011-Q4_K_M.gguf` | `0a730eca469337839f545f719019153d56a3290f0cbe029eb45dc5f067b4944e` |
| `epoch-011-BF16.gguf` | `1d43c14474969e3266ea7f8780756bcd71ecb7b0f8938eb0005a631e71cbeb4b` |
| `model.safetensors` | `818edcfccdee68b0c02b75da6acf6ba0a6e93b7461c2396f653ad1f901024268` |
| `config.json` | `e0cea25421cbbc456edd82093e4a9c608193c3511916c6a44a7f216ea228a77a` |
| `training.json` | `b53a3521fba6d458531a86ed97437a2bfaf0ec7fbe66e9904a52117cfbe20c4e` |

## Runtime pins

- llama.cpp base commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`
- Speculators base commit: `6030a44f2f25c53c031d82caa46631311c1a920a`
- `llama-server` SHA-256: `316f50a2e355bed5d894e439f499e2f5afd4eccb5607df447224daa716367bb1`
- `convert_hf_to_gguf.py` SHA-256: `c819f18fb22927b49fabc3b35d1c9e21ee638b3817eccd1bd4efbcc7116eeb4d`
- `llama-quantize` SHA-256: `2f29c0a446937ae3ad8ae4c4f677e18623e15ba9209b2b880183d61083668c05`

## Training summary

- Six-layer DFlash draft, block size 16
- Runtime depth optimized for this route: 6
- Hard-target training with target embedding and LM head frozen
- Adafactor, learning rate `5e-6`, weight decay `0.01`
- Gradient accumulation 8, seed `20260712`, early-position gamma `0.6`
- Selected artifact: epoch 11

The original `training.json` contains absolute DGX paths. Publish a sanitized copy with paths expressed relative to the released layout.

## Benchmark contract

The baseline and routed phases used the same loaded target, draft allocation, process, KV configuration, and prompts. The baseline set `speculative.n_max` to zero per request. Text correctness compares speculative output against the depth-zero output under deterministic decoding.

The eight exact comparisons and 40-request soak are useful evidence, but still small. Consumers should remeasure acceptance, correctness, throughput, latency, memory pressure, and concurrency on their own workload and hardware.
