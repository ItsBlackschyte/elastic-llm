"""Checkpoint manager.

A checkpoint is a directory:
    model.safetensors   weights
    optim.pt            AdamW state
    state.pt            step, tokens_seen, scaler, RNG states, run args
    config.json         ModelConfig
    manifest.json       sha256 of every file (integrity check)

Atomicity:
  local : write to <dir>.tmp, fsync, os.replace -> never a half-written dir
  hub   : upload the checkpoint folder first, then upload latest.json pointing
          at it. Readers only ever follow latest.json, so a checkpoint becomes
          visible only after it is fully uploaded ("promotion").
"""
import dataclasses
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

LATEST = "latest.json"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_manifest(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir) if f != "manifest.json"]
    manifest = {f: _sha256(os.path.join(ckpt_dir, f)) for f in sorted(files)}
    with open(os.path.join(ckpt_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def verify_manifest(ckpt_dir):
    mpath = os.path.join(ckpt_dir, "manifest.json")
    with open(mpath) as f:
        manifest = json.load(f)
    for name, digest in manifest.items():
        p = os.path.join(ckpt_dir, name)
        if not os.path.exists(p) or _sha256(p) != digest:
            raise RuntimeError(f"checkpoint integrity check FAILED for {name}")
    return True


class CheckpointManager:
    def __init__(self, local_dir, hf_repo=None, keep_local=2):
        self.local_dir = local_dir
        self.hf_repo = hf_repo
        self.keep_local = keep_local
        os.makedirs(local_dir, exist_ok=True)
        self.api = None
        if hf_repo:
            from huggingface_hub import HfApi

            self.api = HfApi()
            self.api.create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)

    # ---------- save ----------
    def save(self, step, model_sd, optim_sd, extra_state, model_cfg):
        tag = f"step_{step:09d}"
        final = os.path.join(self.local_dir, tag)
        tmp = final + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)

        # safetensors rejects memory-shared tensors (e.g. tied wte/lm_head):
        # keep the first name per storage, drop aliases; loader re-ties.
        seen, deduped = {}, {}
        for k, v in model_sd.items():
            ptr = v.data_ptr()
            if ptr in seen:
                continue
            seen[ptr] = k
            deduped[k] = v.contiguous()
        save_file(deduped, os.path.join(tmp, "model.safetensors"))
        torch.save(optim_sd, os.path.join(tmp, "optim.pt"))
        torch.save(extra_state, os.path.join(tmp, "state.pt"))
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(model_cfg), f, indent=2)
        _write_manifest(tmp)

        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(tmp, final)  # atomic promotion (local)
        with open(os.path.join(self.local_dir, LATEST), "w") as f:
            json.dump({"tag": tag, "step": step}, f)
        self._prune_local()
        return final, tag

    def _prune_local(self):
        tags = sorted(d for d in os.listdir(self.local_dir) if d.startswith("step_"))
        for old in tags[: -self.keep_local]:
            shutil.rmtree(os.path.join(self.local_dir, old), ignore_errors=True)

    # ---------- hub ----------
    def push(self, ckpt_path, tag):
        if not self.api:
            return
        self.api.upload_folder(
            repo_id=self.hf_repo,
            folder_path=ckpt_path,
            path_in_repo=f"checkpoints/{tag}",
        )
        # promotion: latest.json is uploaded only after the folder succeeded
        latest = json.dumps({"tag": tag}).encode()
        self.api.upload_file(
            repo_id=self.hf_repo,
            path_or_fileobj=latest,
            path_in_repo=LATEST,
        )
        print(f"[ckpt] pushed and promoted {tag} -> hf.co/{self.hf_repo}")

    # ---------- resume ----------
    def resolve_latest(self):
        """Prefer the hub (canonical); fall back to local. Returns dir or None."""
        if self.api:
            try:
                from huggingface_hub import hf_hub_download, snapshot_download

                lpath = hf_hub_download(self.hf_repo, LATEST, force_download=True)
                tag = json.load(open(lpath))["tag"]
                snap = snapshot_download(
                    self.hf_repo, allow_patterns=[f"checkpoints/{tag}/*"]
                )
                ckpt = os.path.join(snap, "checkpoints", tag)
                verify_manifest(ckpt)
                print(f"[ckpt] resuming from hub: {tag}")
                return ckpt
            except Exception as e:  # noqa: BLE001
                print(f"[ckpt] hub resume unavailable ({e}); trying local")
        lpath = os.path.join(self.local_dir, LATEST)
        if os.path.exists(lpath):
            tag = json.load(open(lpath))["tag"]
            ckpt = os.path.join(self.local_dir, tag)
            if os.path.isdir(ckpt):
                verify_manifest(ckpt)
                print(f"[ckpt] resuming from local: {tag}")
                return ckpt
        return None

    @staticmethod
    def load(ckpt_dir, device="cpu"):
        model_sd = load_file(os.path.join(ckpt_dir, "model.safetensors"), device=device)
        optim_sd = torch.load(os.path.join(ckpt_dir, "optim.pt"), map_location=device)
        extra = torch.load(os.path.join(ckpt_dir, "state.pt"), map_location="cpu")
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        return model_sd, optim_sd, extra, cfg
