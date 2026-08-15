#!/usr/bin/env python3
"""Fine-tune a stock Z-Lab DFlash body on reconciled quantized-target captures."""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def build_model(stock, frozen):
    source = json.loads((stock / "config.json").read_text())
    excluded = {"architectures", "auto_map", "dflash_config", "num_target_layers", "dtype"}
    transformer = Qwen3Config(**{key: value for key, value in source.items() if key not in excluded})
    dflash = source["dflash_config"]
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer,
        draft_vocab_size=source["vocab_size"], block_size=dflash["block_size"],
        aux_hidden_state_layer_ids=[value + 1 for value in dflash["target_layer_ids"]],
        mask_token_id=dflash["mask_token_id"],
        speculators_config=SpeculatorsConfig(
            algorithm="dflash", proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=dflash["block_size"] - 1)],
            default_proposal_method="greedy", verifier=VerifierConfig(name_or_path=None, architectures=[])),
    )
    # Keep the draft body in FP32 so Adam updates are not rounded away. Forward
    # execution uses BF16 autocast and exported body weights are converted to BF16.
    model = DFlashDraftModel(config)
    body = load_file(stock / "model.safetensors", device="cpu")
    missing, unexpected = model.load_state_dict(body, strict=False)
    critical = [name for name in missing if not name.startswith(("embed_tokens.", "lm_head.", "verifier_lm_head.", "verifier_norm."))]
    if critical or unexpected:
        raise ValueError(f"stock body mismatch: missing={critical}, unexpected={unexpected}")
    embed = load_file(frozen / "embed_tokens.weight.safetensors", device="cpu")["embed_tokens.weight"]
    head = load_file(frozen / "lm_head.weight.safetensors", device="cpu")["lm_head.weight"]
    model.embed_tokens.to(torch.bfloat16)
    model.lm_head.to(torch.bfloat16)
    model.embed_tokens.weight.data.copy_(embed)
    model.lm_head.weight.data.copy_(head)
    # Hard-target mode never evaluates verifier logits. Remove the duplicate 1.5 GB head.
    model.verifier_lm_head = nn.Identity()
    model.verifier_norm = nn.Identity()
    model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    return model, set(body)


def configure_trainable(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "fc-only":
        modules = [model.fc, model.hidden_norm]
    elif mode == "full-body":
        modules = [model.fc, model.hidden_norm, model.layers, model.norm]
    else:
        raise ValueError(mode)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def sample_paths(directory):
    paths = sorted(directory.glob("hs_*.safetensors"))
    if not paths:
        raise ValueError(f"no reconciled samples in {directory}")
    return paths


@torch.no_grad()
def evaluate(model, paths, device, max_anchors, runtime_depth=4, gamma=0.75):
    model.eval()
    total_loss = 0.0
    totals = {}
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(20260712)
        torch.cuda.manual_seed_all(20260712)
        for path in paths:
            data = load_file(path, device="cpu")
            hidden = data["hidden_states"].flatten(1).unsqueeze(0).to(device)
            tokens = data["token_ids"].unsqueeze(0).to(device)
            length = tokens.shape[1]
            mask = torch.ones((1, length), dtype=torch.float32, device=device)
            docs = torch.zeros((1, length), dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, metrics = model(hidden, tokens, mask, None, docs, hard_target_ids=tokens,
                                         max_anchors=max_anchors, max_speculative_position=runtime_depth,
                                         gamma=gamma)
            total_loss += float(loss)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
    result = {"loss": total_loss / len(paths)}
    for key in ("full_acc", "eal"):
        denominator = totals.get(f"{key}_total", 0.0)
        result[key] = totals.get(f"{key}_sum", 0.0) / denominator if denominator else 0.0
    result["per_position_accuracy"] = {}
    for position in range(1, runtime_depth + 1):
        denominator = totals.get(f"position_{position}_acc_total", 0.0)
        result["per_position_accuracy"][str(position)] = (
            totals.get(f"position_{position}_acc_sum", 0.0) / denominator if denominator else 0.0
        )
    return result


def save_body(model, body_names, stock, output, metadata):
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    state = model.state_dict()
    body = {name: state[name].detach().to(torch.bfloat16).cpu().contiguous() for name in sorted(body_names)}
    save_file(body, partial / "model.safetensors", metadata={"format": "pt"})
    shutil.copy2(stock / "config.json", partial / "config.json")
    (partial / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.rename(partial, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["fc-only", "full-body"], default="fc-only")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-anchors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--runtime-depth", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--early-position-gamma", type=float, default=0.75,
                        help="Exponential decay denominator; smaller values penalize early rejection more")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, body_names = build_model(args.stock, args.frozen)
    parameters = configure_trainable(model, args.mode)
    model.to(device)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    train = sample_paths(args.train)
    validation = sample_paths(args.validation)
    if args.max_train_samples:
        train = train[:args.max_train_samples]
    if args.max_validation_samples:
        validation = validation[:args.max_validation_samples]
    baseline = evaluate(model, validation, device, args.max_anchors, args.runtime_depth, args.early_position_gamma)
    history = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for index, path in enumerate(train):
            data = load_file(path, device="cpu")
            hidden = data["hidden_states"].flatten(1).unsqueeze(0).to(device)
            tokens = data["token_ids"].unsqueeze(0).to(device)
            length = tokens.shape[1]
            mask = torch.ones((1, length), dtype=torch.float32, device=device)
            docs = torch.zeros((1, length), dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(hidden, tokens, mask, None, docs, hard_target_ids=tokens,
                                   max_anchors=args.max_anchors,
                                   max_speculative_position=args.runtime_depth,
                                   gamma=args.early_position_gamma)
            (loss / args.grad_accum).backward()
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 10 == 0:
                print(json.dumps({"epoch": epoch, "sample": index + 1, "loss": float(loss.detach())}), flush=True)
        validation_metrics = evaluate(model, validation, device, args.max_anchors,
                                      args.runtime_depth, args.early_position_gamma)
        history.append({"epoch": epoch, "validation": validation_metrics})
        model.train()
    metadata = {"mode": args.mode, "epochs": args.epochs, "lr": args.lr,
                "weight_decay": args.weight_decay, "grad_accum": args.grad_accum,
                "max_anchors": args.max_anchors, "seed": args.seed,
                "runtime_depth": args.runtime_depth, "early_position_gamma": args.early_position_gamma,
                "train_samples": len(train), "validation_samples": len(validation),
                "baseline": baseline, "history": history}
    save_body(model, body_names, args.stock, args.output, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
