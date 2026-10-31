"""Wan2.1 text-to-video (diffusers `WanPipeline`).

Self-attention goes through EchoSparse attention, cross-attention through FA4. Video tokens are put in Morton
order once before the first block (together with the rotary embedding) and restored after the last block, so a
64-token block covers a compact 3D patch instead of part of one row.
"""
import torch
from diffusers.models.transformers import transformer_wan

from ..attention import dense_attention
from .common import Clock, grid_morton_order

_self_attention_impl = None


@torch.library.custom_op("echosparse::wan_self_attention", mutates_args=())
def _self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _self_attention_impl(q, k, v)


@_self_attention.register_fake
def _(q, k, v):
    return torch.empty_like(q)


@torch.library.custom_op("echosparse::wan_cross_attention", mutates_args=())
def _cross_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return dense_attention(q, k, v)


@_cross_attention.register_fake
def _(q, k, v):
    return torch.empty_like(q)


def _dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False,
              attention_kwargs=None, *, backend=None, parallel_config=None):
    assert attn_mask is None and dropout_p == 0.0 and not is_causal and scale is None
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    if key.shape[1] == query.shape[1]:
        return _self_attention(q, k, v)
    return _cross_attention(q, k, v)


class _SelfAttention:
    def __init__(self, attention, clock):
        self.attention, self.clock = attention, clock

    def __call__(self, q, k, v):
        c = self.clock
        out = self.attention(q, k, v, c.step, c.branch, c.layer)
        c.layer += 1
        return out


class _MortonOrder:
    """Permutes the tokens (and the rotary embedding) at the first block and restores them after the last."""

    def __init__(self, transformer):
        self.patch = transformer.config.patch_size
        self.perm = self.inv = None
        rope = transformer.rope.forward

        def rope_in_order(hidden_states):
            cos, sin = rope(hidden_states)
            return cos.index_select(1, self.perm), sin.index_select(1, self.perm)

        transformer.rope.forward = rope_in_order
        transformer.register_forward_pre_hook(self._set_grid, with_kwargs=True)
        transformer.blocks[0].register_forward_pre_hook(
            lambda _m, args: (args[0].index_select(1, self.perm),) + tuple(args[1:]))
        transformer.blocks[-1].register_forward_hook(lambda _m, _args, out: out.index_select(1, self.inv))

    def _set_grid(self, _module, args, kwargs):
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        _, _, f, h, w = x.shape
        pt, ph, pw = self.patch
        self.perm, self.inv = grid_morton_order(f // pt, h // ph, w // pw, x.device)


def enable(pipe, attention, morton=True):
    global _self_attention_impl
    clock = Clock(pipe, on_reset=[attention.reset])
    _self_attention_impl = _SelfAttention(attention, clock)
    transformer_wan.dispatch_attention_fn = _dispatch
    if morton:
        _MortonOrder(pipe.transformer)
