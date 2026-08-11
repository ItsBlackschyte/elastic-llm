"""Talk to your trained model.

    python src/sample.py --hf-repo dangeA/elastic-10m \
        --prompt "Once upon a time there was a little robot" --max-new-tokens 200

Downloads the latest promoted checkpoint from your HF repo, loads the
weights, and generates text. Works on GPU or CPU (a 10m-preset model
samples fine on CPU).
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_manager import CheckpointManager  # noqa: E402
from config import ModelConfig  # noqa: E402
from model import GPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt_dir = CheckpointManager("out_sample", args.hf_repo).resolve_latest()
    if ckpt_dir is None:
        sys.exit("no checkpoint found on the hub or locally")

    cfg = ModelConfig(**json.load(open(os.path.join(ckpt_dir, "config.json"))))
    model = GPT(cfg).to(device)
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"), device=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {n/1e6:.1f}M params from {ckpt_dir} on {device}\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(getattr(cfg, "tokenizer", "gpt2"))
    print(f"tokenizer: {getattr(cfg, 'tokenizer', 'gpt2')}")
    idx = torch.tensor([tok(args.prompt).input_ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(idx, args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k)
        text = tok.decode(out[0].tolist())
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
