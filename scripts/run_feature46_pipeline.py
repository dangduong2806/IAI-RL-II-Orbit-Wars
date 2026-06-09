from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_phase(name: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] START {name}\n")
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[1],
        )
        code = proc.wait()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n[{stamp}] END {name} exit_code={code}\n")
        log.flush()
    if code != 0:
        raise SystemExit(f"{name} failed with exit code {code}. See {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bc-episodes", type=int, default=1000)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--ppo-episodes", type=int, default=3000)
    parser.add_argument("--selfplay-start", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dataset-save-every", type=int, default=10)
    parser.add_argument("--rollout-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="logs/feature46_pipeline")
    parser.add_argument("--data", default="data/bc_dataset_v3_feature46.npz")
    parser.add_argument("--bc-save", default="checkpoints/bc_policy.pt")
    parser.add_argument("--ppo-save", default="checkpoints/ppo_selfplay.pt")
    parser.add_argument("--ppo-save-best", default="checkpoints/ppo_selfplay_best.pt")
    args = parser.parse_args()

    py = args.python
    log_dir = Path(args.log_dir)
    common_device = []
    if args.device:
        common_device = ["--device", args.device]

    run_phase(
        "01_bc_dataset",
        [
            py,
            "-u",
            "-m",
            "src.orbit_rl.bc_dataset",
            "--episodes",
            str(args.bc_episodes),
            "--top-k",
            str(args.top_k),
            "--save",
            args.data,
            "--save-every",
            str(args.dataset_save_every),
            "--resume",
        ],
        log_dir,
    )

    run_phase(
        "02_train_bc",
        [
            py,
            "-u",
            "-m",
            "src.orbit_rl.train_bc",
            "--data",
            args.data,
            "--epochs",
            str(args.bc_epochs),
            "--top-k",
            str(args.top_k),
            "--save",
            args.bc_save,
            *common_device,
        ],
        log_dir,
    )

    run_phase(
        "03_train_ppo_selfplay",
        [
            py,
            "-u",
            "-m",
            "src.orbit_rl.train_ppo_selfplay",
            "--init",
            args.bc_save,
            "--episodes",
            str(args.ppo_episodes),
            "--top-k",
            str(args.top_k),
            "--rollout-size",
            str(args.rollout_size),
            "--selfplay-start",
            str(args.selfplay_start),
            "--save",
            args.ppo_save,
            "--save-best",
            args.ppo_save_best,
            "--eval-interval",
            str(args.eval_interval),
            "--eval-episodes",
            str(args.eval_episodes),
            *common_device,
        ],
        log_dir,
    )


if __name__ == "__main__":
    main()
