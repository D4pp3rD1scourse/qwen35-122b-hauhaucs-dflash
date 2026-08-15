# HauhauCS Qwen3.5 122B DFlash release code

This staged repository preserves the code, patches, launch logic, and reproducibility notes for a DFlash draft trained specifically for:

`HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive:Q4_K_M`

The target model is not included. Download it from the original HauhauCS repository. The trained draft is a speculative-decoding companion, not a standalone language model.

## Verified local result

| Configuration | Median output rate | p50 latency |
|---|---:|---:|
| Baseline | 25.58 tok/s | 4.19 s |
| Routed DFlash | 44.91 tok/s | 2.10 s |

- Median throughput change: +75.6%.
- p50 latency change: -49.9%.
- Routed draft acceptance: 77.78%.
- Exact comparison: 8/8 outputs matched the depth-zero baseline.
- Routed soak: 40/40 requests, 0 errors, 74.37% draft acceptance.

These are local, small-sample results for the stated target, runtime, prompts, and decoding settings. They are not a universal performance claim.

## Repository layout

- `scripts/`: corpus generation, capture finalization, training, evaluation, export, routing, and single-load A/B tooling.
- `patches/llama.cpp-e3546c7-dflash-capture-and-request-depth.patch`: hidden-state capture plus per-request `speculative.n_max` support.
- `patches/speculators-6030a44-hard-target-training.patch`: hard-target DFlash training and bounded speculative-position loss.
- `third_party_licenses/`: upstream license texts preserved with the patch sources.
- `REPRODUCIBILITY.md`: pinned commits, hashes, benchmark contract, and model links.

## Upstream attribution

- DFlash paper and reference implementation: <https://github.com/z-lab/dflash>
- Stock Qwen3.5 122B DFlash draft: <https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash>
- Speculators framework: <https://github.com/vllm-project/speculators>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Exact target model and quantization: <https://huggingface.co/HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive>

## Licensing

Original release scripts and documentation are staged under Apache-2.0. Patch files are modifications to their respective upstream projects and remain subject to the upstream notices preserved in `third_party_licenses/`.

## Release pair

This repository contains the code, patches, and reproducibility record. Paired model weights, metadata, and checksums: <https://huggingface.co/D4pp3rD1scourse/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-DFlash>. Use the draft only with the linked target and compatible patched runtime.
