"""Elastic training runtime.

Single GPU:
  python train.py --preset 100m --data-dir data/fineweb2b --hf-repo you/run1 \
      --target-tokens 2_000_000_000 --max-minutes 690

Kaggle T4x2 (DDP):
  torchrun --standalone --nproc_per_node=2 train.py --preset 100m ...

Resume is automatic (--resume auto): pulls latest verified checkpoint from the
HF repo (or local dir), restores model/optimizer/scaler/RNG/step, and data
sampling continues bit-identically because batches are a pure function of
(seed, step, micro, rank).
"""
import argparse
import json
import math
import os
import signal
import time

import torch

from checkpoint_manager import CheckpointManager
from config import PRESETS, ModelConfig
from data import ShardedTokens, batch_seed
from model import GPT


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="100m", choices=list(PRESETS))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--hf-repo", default=None, help="e.g. username/elastic-100m")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch-size", type=int, default=8, help="micro-batch per GPU")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--target-tokens", type=lambda s: int(s.replace("_", "")), default=None)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=0, help="graceful stop; 0=off")
    ap.add_argument("--halt-step", type=int, default=0,
                    help="checkpoint and exit at this step (schedule keeps full horizon); 0=off")
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--loss-log", default=None, help="jsonl per-step loss (for tests)")
    ap.add_argument("--ledger-url", default=None, help="e.g. sqlite:///ledger.db or Neon URL")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--worker-id", default=None)
    return ap.parse_args()


def lr_at(step, args, max_steps):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    min_lr = args.lr * args.min_lr_ratio
    t = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def unwrap(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def main():
    args = parse_args()

    # ---- distributed setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # ---- ledger: claim the lease before doing any work ----
    ledger, worker_id = None, None
    if args.ledger_url and args.job_id:
        abort = 0
        if master:
            import socket
            from ledger import Ledger
            worker_id = args.worker_id or (
                os.environ.get("PROVIDER", "local") + "-" + socket.gethostname()[:12])
            ledger = Ledger(args.ledger_url)
            job = ledger.get_job(args.job_id)
            if job is None:
                print(f"[ledger] job {args.job_id} not found"); abort = 1
            elif job["status"] != "active":
                print(f"[ledger] job status is '{job['status']}', not starting"); abort = 1
            elif not ledger.claim(args.job_id, worker_id):
                print(f"[ledger] lease held by {job['leased_by']}, not starting"); abort = 1
            else:
                print(f"[ledger] lease claimed by {worker_id}")
        if ddp:
            t = torch.tensor([abort], device=device)
            torch.distributed.broadcast(t, src=0)
            abort = int(t.item())
        if abort:
            if ddp:
                torch.distributed.destroy_process_group()
            return

    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- precision: bf16 on Ampere+, fp16+scaler on T4, fp32 on CPU ----
    # bf16 only on Ampere+ (compute capability >= 8): T4 reports bf16 "support"
    # but emulates it slowly and blows memory -- learned the hard way.
    if (device_type == "cuda" and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability()[0] >= 8):
        amp_dtype = torch.bfloat16
    elif device_type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)

    def autocast():
        if amp_dtype is None:
            return torch.autocast(device_type, enabled=False)
        return torch.autocast(device_type, dtype=amp_dtype)

    # ---- model / data / optimizer ----
    cfg = PRESETS[args.preset]
    if args.block_size:
        cfg = ModelConfig(**{**cfg.__dict__, "block_size": args.block_size})
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(
        args.weight_decay, args.lr, (0.9, 0.95), device_type
    )
    train_data = ShardedTokens(args.data_dir, "train")
    val_data = None
    try:
        val_data = ShardedTokens(args.data_dir, "val")
    except FileNotFoundError:
        pass

    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum * world
    if args.max_steps is None:
        assert args.target_tokens, "set --max-steps or --target-tokens"
        args.max_steps = math.ceil(args.target_tokens / tokens_per_step)

    # ---- resume ----
    ckpt_mgr = CheckpointManager(args.out_dir, args.hf_repo if master else None)
    start_step, tokens_seen = 0, 0
    if args.resume == "auto":
        ckpt_dir = ckpt_mgr.resolve_latest() if master else None
        if ddp:
            holder = [ckpt_dir]
            torch.distributed.broadcast_object_list(holder, src=0)
            ckpt_dir = holder[0]
        if ckpt_dir:
            model_sd, optim_sd, extra, saved_cfg = CheckpointManager.load(ckpt_dir)
            assert saved_cfg["n_embd"] == cfg.n_embd, "preset mismatch with checkpoint"
            assert saved_cfg.get("arch", "gpt2") == cfg.arch, (
                f"checkpoint arch {saved_cfg.get('arch', 'gpt2')} != preset arch {cfg.arch}")
            missing, unexpected = unwrap(model).load_state_dict(model_sd, strict=False)
            assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
            # tied aliases (lm_head.weight) are absent by design; anything else is a bug
            assert all("lm_head" in k or "wte" in k for k in missing), missing
            optimizer.load_state_dict(optim_sd)
            if use_scaler and extra.get("scaler"):
                scaler.load_state_dict(extra["scaler"])
            torch.set_rng_state(extra["torch_rng"])
            if device_type == "cuda" and extra.get("cuda_rng") is not None:
                try:
                    torch.cuda.set_rng_state_all(extra["cuda_rng"])
                except Exception:
                    pass  # different GPU count than saved run; sampling unaffected
            start_step = extra["step"]
            tokens_seen = extra["tokens_seen"]

    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])]
        )

    if master:
        n = unwrap(model).num_params()
        print(f"[train] {args.preset}: {n/1e6:.1f}M params (non-emb) | "
              f"{tokens_per_step:,} tok/step | steps {start_step}->{args.max_steps} | "
              f"amp={amp_dtype} world={world}")

    wandb = None
    if master and args.wandb_project:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, resume="allow",
                   id=f"{args.preset}-{args.seed}", config=vars(args))

    # ---- graceful stop: time limit or SIGTERM/SIGINT ----
    stop = {"flag": False}
    def _handler(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    t0 = time.time()

    def time_up():
        return args.max_minutes > 0 and (time.time() - t0) / 60 >= args.max_minutes

    @torch.no_grad()
    def eval_loss(step):
        if val_data is None:
            return None
        model.eval()
        total = 0.0
        for i in range(args.eval_iters):
            x, y = val_data.get_batch(
                args.batch_size, cfg.block_size,
                batch_seed(args.seed, 10_000_000 + step, i, rank), device)
            with autocast():
                _, loss = model(x, y)
            total += loss.item()
        model.train()
        return total / args.eval_iters

    def save_and_push(step):
        if not master:
            return
        extra = {
            "step": step,
            "tokens_seen": tokens_seen,
            "scaler": scaler.state_dict() if use_scaler else None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device_type == "cuda" else None,
            "world_size": world,
            "args": vars(args),
        }
        path, tag = ckpt_mgr.save(
            step, unwrap(model).state_dict(), optimizer.state_dict(), extra, cfg)
        ckpt_mgr.push(path, tag)

    # ---- training loop ----
    model.train()
    loss_f = open(args.loss_log, "a") if (args.loss_log and master) else None
    running = None

    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(args.grad_accum):
            x, y = train_data.get_batch(
                args.batch_size, cfg.block_size,
                batch_seed(args.seed, step, micro, rank), device)
            if ddp:
                model.require_backward_grad_sync = micro == args.grad_accum - 1
            with autocast():
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += tokens_per_step
        step_now = step + 1

        running = loss_acc if running is None else 0.98 * running + 0.02 * loss_acc
        if loss_f:
            loss_f.write(json.dumps({"step": step_now, "loss": loss_acc}) + "\n")
            loss_f.flush()
        if master and step_now % args.log_every == 0:
            el = time.time() - t0
            tps = (step_now - start_step) * tokens_per_step / max(el, 1e-9)
            print(f"step {step_now}/{args.max_steps} loss {loss_acc:.4f} "
                  f"(ema {running:.4f}) lr {lr:.2e} {tps/1e3:.1f}k tok/s")
            if wandb:
                wandb.log({"loss": loss_acc, "lr": lr, "tokens": tokens_seen,
                           "tok_per_sec": tps}, step=step_now)
            if ledger:
                try:
                    hb = ledger.heartbeat(args.job_id, worker_id, step_now,
                                          loss_acc, tokens_seen, tps)
                    if hb["status"] == "paused":
                        print("[ledger] pause requested -> graceful stop")
                        stop["flag"] = True
                    if not hb["lease_ok"]:
                        print("[ledger] LOST LEASE (another worker claimed) -> stopping "
                              "WITHOUT final checkpoint to avoid clobbering")
                        stop["flag"] = True
                        stop["no_ckpt"] = True
                except Exception as e:  # noqa: BLE001
                    print(f"[ledger] heartbeat failed ({e}); training continues")

        if args.eval_every and step_now % args.eval_every == 0:
            vl = eval_loss(step_now)
            if master and vl is not None:
                print(f"step {step_now} val_loss {vl:.4f}")
                if wandb:
                    wandb.log({"val_loss": vl}, step=step_now)

        must_stop = stop["flag"] or time_up() or (args.halt_step and step_now >= args.halt_step)
        if ddp:  # all ranks must agree, else barrier/save deadlocks
            t = torch.tensor([1 if must_stop else 0], device=device)
            torch.distributed.broadcast(t, src=0)
            must_stop = bool(t.item())
        do_ckpt = (step_now % args.ckpt_every == 0 or must_stop
                   or step_now == args.max_steps) and not stop.get("no_ckpt")
        if do_ckpt:
            if ddp:
                torch.distributed.barrier()
            save_and_push(step_now)
        if must_stop:
            if master:
                print(f"[train] graceful stop at step {step_now} "
                      f"({'signal/ledger' if stop['flag'] else 'time limit'})")
            break

    if master and ledger:
        try:
            if step_now == args.max_steps:
                ledger.set_status(args.job_id, "done")
            ledger.release(args.job_id, worker_id)
            print("[ledger] lease released")
        except Exception as e:  # noqa: BLE001
            print(f"[ledger] release failed ({e}); lease will expire on its own")

    if loss_f:
        loss_f.close()
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
