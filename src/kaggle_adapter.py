"""Launch a training session on Kaggle via the official API (batch execution).

Account credentials: one folder per person under accounts/, each containing
that person's own kaggle.json (Kaggle -> Settings -> Create New Token):
    accounts/you/kaggle.json
    accounts/friend/kaggle.json

Secrets: in EACH Kaggle account, add secrets HF_TOKEN and LEDGER_URL
(Add-ons -> Secrets) and enable them for notebooks.

GPU note: the API enables GPU but the *type* (T4 x2 vs P100) follows the
account's notebook default -- set it once manually in any notebook's settings.

Usage:
    python kaggle_adapter.py --account accounts/you --repo-url https://github.com/YOU/elastic-llm \
        --job-id abc123 --max-minutes 690
"""
import argparse
import json
import os
import subprocess
import tempfile

SCRIPT_TEMPLATE = '''
import os, subprocess, sys

from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["LEDGER_URL"] = s.get_secret("LEDGER_URL")
os.environ["PROVIDER"] = "kaggle-{account_label}"

subprocess.run(["git", "clone", "--depth", "1", "{repo_url}", "elastic-llm"], check=True)
os.chdir("elastic-llm")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run(["huggingface-cli", "download", "{data_repo}", "--repo-type", "dataset",
                "--local-dir", "{data_dir}"], check=True)

n_gpu = 0
try:
    import torch; n_gpu = torch.cuda.device_count()
except Exception: pass
base = ["src/train.py", "--preset", "{preset}", "--data-dir", "{data_dir}",
        "--hf-repo", "{hf_repo}", "--target-tokens", "{target_tokens}",
        "--max-minutes", "{max_minutes}", "--ledger-url", os.environ["LEDGER_URL"],
        "--job-id", "{job_id}"] + {extra_args}
if n_gpu >= 2:
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={{n_gpu}}"] + base
else:
    cmd = [sys.executable] + base
print("launching:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
'''


def launch(account_dir, repo_url, job, max_minutes=690, slug_prefix="elastic"):
    account_label = os.path.basename(os.path.normpath(account_dir))
    with open(os.path.join(account_dir, "kaggle.json")) as f:
        username = json.load(f)["username"]

    cfg = json.loads(job.get("config_json") or "{}")
    extra = cfg.get("extra_args", ["--batch-size", "8", "--grad-accum", "16",
                                   "--ckpt-every", "400"])
    script = SCRIPT_TEMPLATE.format(
        account_label=account_label, repo_url=repo_url,
        data_repo=cfg.get("data_repo", ""), data_dir=job["data_dir"],
        preset=job["preset"], hf_repo=job["hf_repo"],
        target_tokens=job["target_tokens"], max_minutes=int(max_minutes),
        job_id=job["id"], extra_args=json.dumps(extra))

    slug = f"{slug_prefix}-{job['id']}"
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "kernel.py"), "w") as f:
            f.write(script)
        with open(os.path.join(td, "kernel-metadata.json"), "w") as f:
            json.dump({
                "id": f"{username}/{slug}",
                "title": slug,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
            }, f, indent=2)
        env = {**os.environ, "KAGGLE_CONFIG_DIR": os.path.abspath(account_dir)}
        r = subprocess.run(["kaggle", "kernels", "push", "-p", td],
                           env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise RuntimeError(f"kaggle push failed: {out}")
    return f"https://www.kaggle.com/code/{username}/{slug}", out


def status(account_dir, kernel_ref):
    env = {**os.environ, "KAGGLE_CONFIG_DIR": os.path.abspath(account_dir)}
    r = subprocess.run(["kaggle", "kernels", "status", kernel_ref],
                       env=env, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--ledger-url", default=None)
    ap.add_argument("--max-minutes", type=float, default=690)
    args = ap.parse_args()
    from ledger import Ledger
    job = Ledger(args.ledger_url).get_job(args.job_id)
    url, out = launch(args.account, args.repo_url, job, args.max_minutes)
    print(out)
    print(f"session: {url}")
