"""Tokenize a HuggingFace dataset into uint16 memmap shards (run once, upload
shards to HF Hub or Kaggle Dataset so every worker can pull them).

Examples:
  # 124m coder (StarCoderData Python, StarCoder2 tokenizer; accept both
  # licenses on HF first or downloads 403):
  python prepare_data.py --dataset bigcode/starcoderdata --data-dir python \\
      --text-key content --tokenizer bigcode/starcoder2-15b \\
      --out data/starcoder-py --max-tokens 2_000_000_000

  # 10M warm-up model (~200M tokens of TinyStories):
  python prepare_data.py --dataset roneneldan/TinyStories --out data/tinystories \
      --max-tokens 200_000_000

  # 100M model (~2B tokens of FineWeb-Edu sample):
  python prepare_data.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
      --out data/fineweb2b --max-tokens 2_000_000_000
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SHARD_TOKENS = 100_000_000  # 100M tokens/shard = 200MB uint16 files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", default=None, help="dataset config name")
    ap.add_argument("--data-dir", default=None, help="e.g. python for starcoderdata")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=lambda s: int(s.replace("_", "")), required=True)
    ap.add_argument("--val-tokens", type=lambda s: int(s.replace("_", "")), default=5_000_000)
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = tok.eos_token_id
    assert tok.vocab_size <= 65535, "uint16 storage requires vocab <= 65535"

    ds = load_dataset(args.dataset, name=args.name, data_dir=args.data_dir,
                      split=args.split, streaming=True)

    buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    fill, shard_i, total = 0, 0, 0
    val_written = False

    def flush(prefix, n, idx):
        path = os.path.join(args.out, f"{prefix}_{idx:04d}.bin")
        buf[:n].tofile(path)
        print(f"wrote {path} ({n:,} tokens)")

    for ex in ds:
        ids = tok(ex[args.text_key]).input_ids + [eot]
        i = 0
        while i < len(ids):
            take = min(len(ids) - i, SHARD_TOKENS - fill)
            buf[fill : fill + take] = ids[i : i + take]
            fill += take
            i += take
            total += take
            if total % 10_000_000 < take:
                print(f"  ...{total/1e6:.0f}M tokens", flush=True)
            if not val_written and total >= args.val_tokens:
                flush("val", fill, 0)
                fill, val_written = 0, True
            elif fill == SHARD_TOKENS:
                flush("train", fill, shard_i)
                shard_i += 1
                fill = 0
            if total >= args.max_tokens + args.val_tokens:
                break
        if total >= args.max_tokens + args.val_tokens:
            break

    if fill > 0:
        flush("train", fill, shard_i)
    print(f"done: {total:,} tokens total -> {args.out}")


if __name__ == "__main__":
    main()
