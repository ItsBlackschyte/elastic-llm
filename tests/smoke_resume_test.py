"""THE test that matters: an interrupted-and-resumed run must produce
bit-identical losses to a never-interrupted run.

Run A: 30 steps straight through.
Run B: 15 steps -> process exits -> new process resumes from checkpoint -> 30.
Assert: losses for steps 16..30 match exactly.

Runs on CPU in ~1 minute. If this passes, Kaggle->Colab migration is sound.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
WORK = os.path.join(ROOT, "tests", "_work")


def run_train(out_dir, max_steps, loss_log, resume, halt_step=0, preset="tiny"):
    cmd = [
        sys.executable, os.path.join(SRC, "train.py"),
        "--preset", preset, "--data-dir", os.path.join(WORK, "data"),
        "--out-dir", out_dir, "--batch-size", "4", "--grad-accum", "2",
        "--max-steps", str(max_steps), "--halt-step", str(halt_step), "--ckpt-every", "15",
        "--eval-every", "0", "--log-every", "10", "--warmup-steps", "5",
        "--loss-log", loss_log, "--resume", resume, "--seed", "42",
    ]
    r = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("training process failed")


def read_losses(path):
    with open(path) as f:
        return {json.loads(l)["step"]: json.loads(l)["loss"] for l in f}


def check_preset(preset):
    work = os.path.join(WORK, preset)
    print(f"\n#### arch check: preset={preset} ####")
    print("== run A: 30 steps, uninterrupted ==")
    run_train(os.path.join(work, "runA"), 30, os.path.join(work, "a.jsonl"), "none", preset=preset)
    print("== run B part 1: halted at step 15 (schedule horizon still 30) ==")
    run_train(os.path.join(work, "runB"), 30, os.path.join(work, "b.jsonl"), "none",
              halt_step=15, preset=preset)
    print("== run B part 2: fresh process resumes 15 -> 30 ==")
    run_train(os.path.join(work, "runB"), 30, os.path.join(work, "b.jsonl"), "auto", preset=preset)
    a = read_losses(os.path.join(work, "a.jsonl"))
    b = read_losses(os.path.join(work, "b.jsonl"))
    bad = [s for s in range(16, 31) if abs(a[s] - b[s]) > 1e-9]
    for s in (16, 20, 25, 30):
        print(f"step {s}: A={a[s]:.6f}  B={b[s]:.6f}  {'OK' if s not in bad else 'MISMATCH'}")
    if bad:
        raise SystemExit(f"FAIL [{preset}]: resumed run diverged at steps {bad}")
    print(f"PASS [{preset}]: interrupted+resumed is bit-identical.")


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(os.path.join(WORK, "data"))
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size=300_000, dtype=np.uint16).tofile(
        os.path.join(WORK, "data", "train_0000.bin"))
    rng.integers(0, 256, size=30_000, dtype=np.uint16).tofile(
        os.path.join(WORK, "data", "val_0000.bin"))

    check_preset("tiny")        # gpt2 arch (protects mpu-30m lineage)
    check_preset("tiny-llama")  # llama arch (protects mpu-124m-coder)
    print("\nALL PASS: both architectures resume bit-identically.")


if __name__ == "__main__":
    main()
