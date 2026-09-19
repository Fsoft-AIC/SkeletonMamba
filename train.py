"""Train SkeletonMamba on prepared EgoAIST++ clips."""
import argparse

from trainer import Trainer
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/egoaistpp_cuda.yaml")
    parser.add_argument("--resume", help="Versioned checkpoint; restores optimizer, RNG and data order")
    parser.add_argument("--device", help="cpu, cuda:0, or auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-after", type=int, help="Pause at this successful update without changing the LR schedule")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    trainer = Trainer(config)
    if args.resume:
        trainer.load(args.resume)
    elif (trainer.output / "latest.pt").exists():
        raise FileExistsError("This output directory already has a checkpoint; use --resume or a new --output-dir")
    trainer.train(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
