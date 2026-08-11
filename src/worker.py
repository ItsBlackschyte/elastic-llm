"""Generic worker: the "just press Run" entrypoint.

Asks the ledger which job to run (first active job with a free lease, or an
explicit --job-id), downloads the job's data shards from HF Hub if missing,
then launches train.py with the job's config. All configuration lives in the
ledger (edited from the dashboard) -- this script never needs editing.

Colab cell:   !python src/worker.py --max-minutes 150
Kaggle cell:  !python src/worker.py --max-minutes 690
Env needed:   LEDGER_URL, HF_TOKEN
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ledger import Ledger  # noqa: E402


def pick_job(led, job_id=None):
    if job_id:
        return led.get_job(job_id)
    for j in led.list_jobs():
        if j["status"] != "active":
            continue
        live = j["lease_expires"] and j["lease_expires"] > dt.datetime.utcnow()
        if not live:
            return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-url", default=os.environ.get("LEDGER_URL"))
    ap.add_argument("--job-id", default=None, help="default: first runnable job")
    ap.add_argument("--max-minutes", type=float, default=150)
    args = ap.parse_args()
    if not args.ledger_url:
        sys.exit("worker: set LEDGER_URL (env or --ledger-url)")

    led = Ledger(args.ledger_url)
    job = pick_job(led, args.job_id)
    if job is None:
        sys.exit("worker: no runnable job (none active, or all leases live)")
    cfg = json.loads(job.get("config_json") or "{}")
    print(f"[worker] job {job['id']} '{job['name']}' preset={job['preset']} "
          f"target={job['target_tokens']:,} tokens")

    # data: download shards from HF if the local dir is missing/empty
    data_dir = job["data_dir"]
    if not (os.path.isdir(data_dir) and any(
            f.endswith(".bin") for f in os.listdir(data_dir))):
        repo = cfg.get("data_repo")
        if not repo:
            sys.exit(f"worker: {data_dir} has no shards and job has no data_repo")
        print(f"[worker] downloading shards from {repo} -> {data_dir}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo, repo_type="dataset", local_dir=data_dir)

    n_gpu = 0
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    base = [os.path.join(here, "train.py"),
            "--preset", job["preset"], "--data-dir", data_dir,
            "--target-tokens", str(job["target_tokens"]),
            "--max-minutes", str(args.max_minutes),
            "--ledger-url", args.ledger_url, "--job-id", job["id"]]
    if job.get("hf_repo"):
        base += ["--hf-repo", job["hf_repo"]]
    base += cfg.get("extra_args", [])
    cmd = (["torchrun", "--standalone", f"--nproc_per_node={n_gpu}"] + base
           if n_gpu >= 2 else [sys.executable] + base)
    print("[worker] launching:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
