"""Token layout helpers: real tokens <-> the block-padded layout of both operators.

The kernels consume tokens already ordered so that
  * each 64-token hardware block is one (tt, th, tw) tile, tt*th*tw = 64
    (VSA uses 4x4x4), valid tokens first, padding last;
  * inside a block, every consecutive R tokens form one router group. For a
    full tile the order is sub-cube major: the tile is split into
    (st, sh, sw) sub-cubes with st*sh*sw = R, raster over sub-cubes, raster
    inside each sub-cube. R-groups are then genuine spatio-temporal
    sub-cubes.
  * partial tiles at the video boundary keep the same ordering restricted to
    the tokens that exist, packed to the front; their R-groups are then
    consecutive runs of that order (not exact sub-cubes). This is the
    documented convention for boundary tiles.

This is an explicit, invertible permutation; `to_blocked` / `from_blocked`
gather/scatter with it. It mirrors FastVideo's `get_tile_partition_indices`
(tile raster order) with the added sub-cube level. The video shape is always
given explicitly, never inferred from L; `sequence_layout` is the 1-D form.
Original VSA ignores the order inside a block, so one layout serves both.

Padding contract: `to_blocked` writes the real tokens into a zero buffer and
both are differentiable, so gradients reach real tokens only; padded slots are
constants. The kernels never pool padded slots (the original's sum over all 64
zero-padded slots / count is the same value), output 0 there and give them zero
gradient. The original's physical-buffer adjoint also reaches padded slots;
that is not reproduced, and with real tokens in and out it changes nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

SUBCUBE_FOR_R = {4: (1, 2, 2), 8: (2, 2, 2), 16: (1, 4, 4), 32: (2, 4, 4)}


@dataclass
class BlockLayout:
    shape: Tuple[int, ...]
    tile: Tuple[int, ...]
    sub: Tuple[int, ...]
    gather: torch.Tensor   # [NB*64] raster token index per padded slot, -1 for padding
    scatter: torch.Tensor  # [T*H*W] padded slot of each raster token
    blk_cnt: torch.Tensor  # [NB] int32

    @property
    def num_blocks(self) -> int:
        return self.blk_cnt.numel()

    @property
    def padded_len(self) -> int:
        return self.gather.numel()


def build_layout(shape, tile=(4, 4, 4), sub=None, R: int = 8, device="cpu") -> BlockLayout:
    T, H, W = shape
    tt, th, tw = tile
    assert tt * th * tw == 64, "hardware block must hold 64 tokens"
    sub = tuple(sub) if sub is not None else SUBCUBE_FOR_R[R]
    st, sh, sw = sub
    assert tt % st == 0 and th % sh == 0 and tw % sw == 0, "sub-cube must tile the block"
    idx = torch.arange(T * H * W).reshape(T, H, W)
    # intra-tile order of local coordinates (sub-cube major)
    local = []
    for a in range(tt // st):
        for b in range(th // sh):
            for c in range(tw // sw):
                for x in range(st):
                    for y in range(sh):
                        for z in range(sw):
                            local.append((a * st + x, b * sh + y, c * sw + z))
    gather = []
    cnts = []
    for t0 in range(0, T, tt):
        for h0 in range(0, H, th):
            for w0 in range(0, W, tw):
                blk = [int(idx[t0 + x, h0 + y, w0 + z]) for (x, y, z) in local
                       if t0 + x < T and h0 + y < H and w0 + z < W]
                cnts.append(len(blk))
                gather.extend(blk + [-1] * (64 - len(blk)))
    gather = torch.tensor(gather, dtype=torch.int64)
    scatter = torch.empty(T * H * W, dtype=torch.int64)
    pos = torch.nonzero(gather >= 0).squeeze(1)
    scatter[gather[pos]] = pos
    return BlockLayout(tuple(shape), tuple(tile), sub, gather.to(device), scatter.to(device),
                       torch.tensor(cnts, dtype=torch.int32, device=device))


def sequence_layout(n: int, device="cpu") -> BlockLayout:
    """1-D sequences: blocks of 64 consecutive tokens, the last one partial."""
    cnt = tail_block_counts(n)
    gather = torch.full((cnt.numel() * 64,), -1, dtype=torch.int64)
    gather[:n] = torch.arange(n)
    return BlockLayout((n,), (64,), (), gather.to(device), torch.arange(n, device=device), cnt.to(device))


def to_blocked(x: torch.Tensor, lay: BlockLayout) -> torch.Tensor:
    """[B, H, N, ...] real tokens -> [B, H, NB*64, ...] block-padded (padding = 0); q/k/v, gates and weights."""
    out = x.new_zeros(*x.shape[:2], lay.padded_len, *x.shape[3:])
    out[:, :, lay.scatter] = x
    return out


def from_blocked(y: torch.Tensor, lay: BlockLayout) -> torch.Tensor:
    """[B, H, NB*64, ...] -> [B, H, N, ...] real tokens in their original order."""
    return y[:, :, lay.scatter]


def tail_block_counts(L_valid: int, device="cpu") -> torch.Tensor:
    """1-D sequences: blocks of 64 consecutive tokens, last block partial."""
    nb = math.ceil(L_valid / 64)
    cnt = torch.full((nb,), 64, dtype=torch.int32)
    cnt[-1] = L_valid - 64 * (nb - 1)
    return cnt.to(device)


def pad_tail(x: torch.Tensor, L_valid: int) -> torch.Tensor:
    """[B,H,L_valid,D] -> [B,H,ceil(L/64)*64,D] zero-padded at the end."""
    nb = math.ceil(L_valid / 64)
    if nb * 64 == L_valid:
        return x.contiguous()
    out = x.new_zeros(*x.shape[:2], nb * 64, x.shape[3])
    out[:, :, :L_valid] = x
    return out
