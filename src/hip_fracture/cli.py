"""Command-line entry points for the two-class research pilot."""

import argparse
import json
from pathlib import Path

from hip_fracture.data import class_counts, load_manifest, split_manifest
from hip_fracture.training import TrainConfig, evaluate, predict, train


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Research-only two-class hip-fracture pilot.")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("validate", "split", "train"):
        command = commands.add_parser(name)
        command.add_argument("--classes", nargs=2, required=True, metavar=("CLASS_0", "CLASS_1"))
        if name == "train":
            command.add_argument("--train", type=Path, required=True)
            command.add_argument("--val", type=Path, required=True)
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--epochs", type=int, default=20)
            command.add_argument("--batch-size", type=int, default=4)
            command.add_argument("--lr", type=float, default=0.001)
            command.add_argument("--seed", type=int, default=2025)
            command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        else:
            command.add_argument("--manifest", type=Path, required=True)
        if name == "split":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--seed", type=int, default=2025)
            command.add_argument("--val-fraction", type=float, default=0.2)
            command.add_argument("--test-fraction", type=float, default=0.2)
    for name in ("evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--model", type=Path, help="Relocated copy of the exact original backbone.")
        command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        if name == "evaluate":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--batch-size", type=int, default=4)
        else:
            command.add_argument("--image", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate":
        print(json.dumps(class_counts(load_manifest(args.manifest, args.classes)), indent=2))
    elif args.command == "split":
        splits = split_manifest(
            args.manifest, args.output, args.classes, args.seed, args.val_fraction, args.test_fraction,
        )
        print(json.dumps({name: class_counts(samples) for name, samples in splits.items()}, indent=2))
    elif args.command == "train":
        path = train(TrainConfig(
            train_manifest=args.train, val_manifest=args.val, model=args.model, output=args.output,
            classes=args.classes, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, seed=args.seed, device=args.device,
        ))
        print(f"Best validation checkpoint: {path}")
    elif args.command == "evaluate":
        metrics = evaluate(args.checkpoint, args.manifest, args.output, args.batch_size, args.device, args.model)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        print(json.dumps(predict(args.checkpoint, args.image, args.device, args.model), indent=2))


if __name__ == "__main__":
    main()
