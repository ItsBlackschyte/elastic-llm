"""Ledger integration test (CPU, ~1 min):
1. training with a ledger writes heartbeats and releases its lease
2. a second worker CANNOT claim a live-leased job (mutual exclusion)
3. a paused job refuses to start
4. resumed training after a pause continues from the checkpoint
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
WORK = os.path.join(ROOT, "tests", "_work_ledger")
sys.path.insert(0, SRC)


def train(job_id, url, out_dir, max_steps, halt=0, worker="w1", expect_fail=False):
    cmd = [sys.executable, os.path.join(SRC, "train.py"),
           "--preset", "tiny", "--data-dir", os.path.join(WORK, "data"),
           "--out-dir", out_dir, "--batch-size", "4", "--grad-accum", "2",
           "--max-steps", str(max_steps), "--halt-step", str(halt),
           "--ckpt-every", "10", "--eval-every", "0", "--log-every", "5",
           "--warmup-steps", "5", "--seed", "42",
           "--ledger-url", url, "--job-id", job_id, "--worker-id", worker]
    r = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    if r.returncode != 0 and not expect_fail:
        print(r.stdout); print(r.stderr)
        raise SystemExit("training failed")
    return r.stdout


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(os.path.join(WORK, "data"))
    rng = np.random.default_rng(0)
    rng.integers(0, 256, 300_000, dtype=np.uint16).tofile(
        os.path.join(WORK, "data", "train_0000.bin"))

    url = f"sqlite:///{os.path.join(WORK, 'ledger.db')}"
    from ledger import Ledger
    led = Ledger(url)
    jid = led.create_job("test", "tiny", "data", None, 100_000)
    out = os.path.join(WORK, "run")

    print("== 1. train 10 steps with ledger, halt, check heartbeats+release ==")
    train(jid, url, out, 30, halt=10)
    ms = led.recent_metrics(jid)
    assert len(ms) >= 2, "no heartbeats written"
    assert led.get_job(jid)["leased_by"] is None, "lease not released"
    print(f"   OK: {len(ms)} heartbeats, lease released")

    print("== 2. mutual exclusion: worker2 must refuse while lease is live ==")
    assert led.claim(jid, "worker1", ttl_minutes=15)
    log = train(jid, url, out, 30, worker="worker2", expect_fail=True)
    assert "lease held by worker1" in log, log
    ms_before = len(led.recent_metrics(jid))
    led.release(jid, "worker1")
    print("   OK: second worker refused, no training happened")
    assert len(led.recent_metrics(jid)) == ms_before

    print("== 3. paused job refuses to start ==")
    led.set_status(jid, "paused")
    log = train(jid, url, out, 30, expect_fail=True)
    assert "'paused', not starting" in log, log
    print("   OK: paused job did not start")

    print("== 4. resume: continues from step 10 checkpoint to 20 ==")
    led.set_status(jid, "active")
    log = train(jid, url, out, 30, halt=20)
    assert "resuming from local: step_000000010" in log, log
    last = led.recent_metrics(jid)[-1]
    assert last["step"] == 20, last
    print(f"   OK: resumed and reached step {last['step']}")

    print("\nPASS: ledger integration (leases, heartbeats, pause, resume) works.")


if __name__ == "__main__":
    main()
