"""Sharded uint16 memmap token loader.

Key design decision: batch sampling is a *pure function of (seed, step,
micro_step, rank)* -- there is no stateful RNG to checkpoint for data. A
resumed run at step N draws bit-identical batches to a run that never
stopped, on any machine, for any world size. This is what makes
cross-provider migration exactly reproducible.
"""
import glob
import os

import numpy as np
import torch


class ShardedTokens:
    def __init__(self, data_dir: str, split: str):
        pattern = os.path.join(data_dir, f"{split}*.bin")
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no shards matching {pattern}")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.lengths = torch.tensor([len(s) for s in self.shards], dtype=torch.float64)
        self.weights = self.lengths / self.lengths.sum()
        self.total_tokens = int(self.lengths.sum().item())

    def get_batch(self, batch_size, block_size, seed, device):
        """Deterministic batch: `seed` must uniquely encode (run_seed, step, micro, rank)."""
        g = torch.Generator()
        g.manual_seed(seed)
        shard_idx = torch.multinomial(self.weights.float(), batch_size, replacement=True, generator=g)
        xs, ys = [], []
        for si in shard_idx.tolist():
            shard = self.shards[si]
            hi = len(shard) - block_size - 1
            off = int(torch.randint(0, hi, (1,), generator=g).item())
            buf = torch.from_numpy(shard[off : off + block_size + 1].astype(np.int64))
            xs.append(buf[:-1])
            ys.append(buf[1:])
        x = torch.stack(xs)
        y = torch.stack(ys)
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def batch_seed(run_seed: int, step: int, micro: int, rank: int) -> int:
    """Collision-free-enough packing of the batch coordinates into one seed."""
    return (run_seed * 1_000_003 + step) * 4096 + micro * 64 + rank
