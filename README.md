# elastic-llm — stage 1: portable training runtime

Train a GPT-style model on any GPU, stop anywhere, resume anywhere.
Checkpoints are the persistent state; compute is disposable.

Verified property (see `tests/smoke_resume_test.py`): a run that is killed and
resumed in a fresh process produces **bit-identical** losses to a run that was
never interrupted. This is what makes Kaggle -> Colab -> AWS handoffs safe.

## Layout

```
src/config.py              model presets (tiny / 10m / 100m)
src/model.py               GPT (pre-LN, tied embeddings, SDPA attention)
src/data.py                memmap shard loader; batches are a pure function of
                           (seed, step, micro, rank) -> exact resume for free
src/prepare_data.py        tokenize an HF dataset into uint16 shards (run once)
src/checkpoint_manager.py  atomic local saves, sha256 manifests,
                           HF Hub push + latest.json promotion
src/train.py               training loop: AMP (bf16/fp16 auto), DDP, grad accum,
                           cosine LR, graceful stop on time limit / SIGTERM
tests/smoke_resume_test.py the kill-and-resume equivalence test (CPU, ~1 min)
```

## One-time setup

1. Create a HF account, make a **write** token (settings -> access tokens).
2. On Kaggle: Add-ons -> Secrets -> add `HF_TOKEN`. On Colab: use the key icon
   (secrets) or `huggingface-cli login`.
3. Prepare data once (any machine with disk + internet), then upload the shard
   folder to a HF dataset repo so every worker can pull it:

```bash
pip install -r requirements.txt
# 10M warm-up (~15 min):
python src/prepare_data.py --dataset roneneldan/TinyStories \
    --out data/tinystories --max-tokens 200_000_000
# 100M main run (~2B tokens, ~4GB of shards):
python src/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
    --out data/fineweb2b --max-tokens 2_000_000_000
huggingface-cli upload YOURNAME/fineweb2b-shards data/fineweb2b --repo-type dataset
```

## Phase 0 — 10M model on Colab (one afternoon)

```bash
python src/train.py --preset 10m --data-dir data/tinystories \
    --hf-repo YOURNAME/elastic-10m --target-tokens 400_000_000 \
    --batch-size 16 --grad-accum 4 --ckpt-every 300 --max-minutes 150
```
Kill it halfway. Run the same command again. It resumes from the hub. That is
the whole system working.

## Phase 1 — 100M model on Kaggle T4x2 (batch sessions)

Kaggle notebook cell (enable GPU T4 x2, internet on):

```bash
!git clone <your repo> && cd elastic-llm && pip install -q -r requirements.txt
!cd elastic-llm && huggingface-cli download YOURNAME/fineweb2b-shards \
    --repo-type dataset --local-dir data/fineweb2b
!cd elastic-llm && torchrun --standalone --nproc_per_node=2 src/train.py \
    --preset 100m --data-dir data/fineweb2b --hf-repo YOURNAME/elastic-100m \
    --target-tokens 2_000_000_000 --batch-size 8 --grad-accum 16 \
    --ckpt-every 400 --max-minutes 690
```

`--max-minutes 690` = self-terminate at 11.5h with a final checkpoint, before
Kaggle's 12h kill. Use "Save & Run All (Commit)" for background execution.
Every new session runs the identical command — resume is automatic.

### Two-person relay
Both accounts run the same notebook against the same `--hf-repo`. Person A's
session ends -> final checkpoint promoted -> person B starts their session ->
it pulls `latest.json` and continues. Never run two sessions on the same repo
simultaneously (stage 2's ledger will enforce this; until then, coordinate).

## Notes

- T4 has no bf16: the script auto-selects fp16 + GradScaler (scaler state is
  checkpointed). On Ampere+ (A10G etc.) it auto-selects bf16.
- LR schedule horizon comes from `--target-tokens`; keep it identical across
  all sessions of one run, or the schedule (and loss) will silently differ.
- `--halt-step N` stops at an exact step without touching the schedule —
  useful for controlled handoff tests.
- Checkpoint size ~= 12 bytes/param (fp32 weights + Adam moments): ~1.4GB for
  the 100m preset. At `--ckpt-every 400` that is a push every ~25 min.

## Stage 2+3 — ledger + dashboard (full control UI)

```
src/ledger.py            jobs, leases (mutual exclusion), heartbeats/metrics.
                         SQLite locally, Neon Postgres for the real thing.
src/kaggle_adapter.py    push a batch training session via the official
                         Kaggle API (one accounts/<name>/kaggle.json per person)
dashboard.py             Streamlit mission control: create jobs, edit config,
                         pause/resume (graceful), launch Kaggle sessions,
                         live loss/throughput/worker view
tests/ledger_integration_test.py  leases block concurrent workers, pause works
```

Quickstart (local, zero setup):
```bash
pip install -r requirements.txt
python src/ledger.py create-job --name run-100m --preset 100m \
    --data-dir data/fineweb2b --hf-repo YOU/elastic-100m --target-tokens 2_000_000_000
streamlit run dashboard.py           # opens the UI in your browser
```

Going multi-person: create a free Postgres at neon.tech, set
`LEDGER_URL=postgresql://...` (and add it as a Kaggle secret in both
accounts) -- nothing else changes. Workers launched from the dashboard claim
the job's lease; a second session politely refuses to start while a lease is
live, and a dead session's lease expires on its own (15 min).

Pause semantics: the button sets a flag; workers see it at the next heartbeat
(~1-2 min), finish the step, push a final checkpoint, release the lease.
Colab has no launch API -- its button in the UI only opens the notebook; you
press run. That is a Colab ToS boundary, not a missing feature.


## mpu-124m-coder-base (llama-style arch, StarCoder tokenizer)

Prereq: accept licenses on HF for `bigcode/starcoderdata` AND
`bigcode/starcoder2-15b` (tokenizer) while logged in, or downloads 403.

Data prep (Colab, ~4-6h, run by ~15:30 IST for a 22:00 launch):
```bash
python src/prepare_data.py --dataset bigcode/starcoderdata --data-dir python \
    --text-key content --tokenizer bigcode/starcoder2-15b \
    --out data/starcoder-py --max-tokens 2_000_000_000
hf upload YOU/starcoder-py-shards data/starcoder-py --repo-type dataset
```

Train (Kaggle T4x2 batch session, per relay leg):
```bash
torchrun --standalone --nproc_per_node=2 src/train.py --preset 124m-coder \
    --data-dir data/starcoder-py --hf-repo YOU/mpu-124m-coder-base \
    --target-tokens 2_000_000_000 --batch-size 8 --grad-accum 8 \
    --ckpt-every 400 --max-minutes 690 \
    --ledger-url $LEDGER_URL --job-id JOBID
```
tokens/step = 8x1024x8x2 = 131,072. Keep batch/accum identical every session.

## Roadmap

- stage 4: SkyPilot task YAML for AWS spot (the 1B run)
- stage 5: router scoring from measured tokens/sec telemetry per provider
