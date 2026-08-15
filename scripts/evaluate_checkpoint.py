#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from train_hard_targets import build_model, evaluate, sample_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-anchors", type=int, default=8)
    parser.add_argument("--runtime-depth", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--early-position-gamma", type=float, default=0.75)
    args = parser.parse_args()
    model, _ = build_model(args.checkpoint, args.frozen)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda")
    model.to(device)
    metrics = evaluate(model, sample_paths(args.dataset), device, args.max_anchors,
                       args.runtime_depth, args.early_position_gamma)
    print(json.dumps({"checkpoint": str(args.checkpoint), "samples": len(sample_paths(args.dataset)),
                      "max_anchors": args.max_anchors, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
