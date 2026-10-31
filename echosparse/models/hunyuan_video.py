"""HunyuanVideo text-to-video (diffusers `HunyuanVideoPipeline`).

The dual- and single-stream blocks attend over [video tokens, padded text tokens] with a key-padding mask. The
padding is cropped; video query rows go through EchoSparse attention with the video tokens in Morton order and the
text tokens appended as a sink that every tile computes exactly; text query rows use FA4. Other attention calls
(the token refiner) keep diffusers' dispatcher. One transformer forward per step (embedded guidance).
"""
import torch
from diffusers.models.transformers import transformer_hunyuan_video as hv

from ..attention import BK, dense_attention
from .common import Clock, grid_morton_order

_joint_impl = None
_original_dispatch = hv.dispatch_attention_fn


@torch.library.custom_op("echosparse::hunyuan_joint_attention", mutates_args=())
def _joint_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _joint_impl(q, k, v, mask)


@_joint_attention.register_fake
def _(q, k, v, mask):
    return torch.empty_like(q)


class _JointAttention:
    def __init__(self, pipe, attention):
        self.attention = attention
        self.clock = Clock(pipe, on_reset=[attention.reset, self._reset])
        self.patch = (pipe.transformer.config.patch_size_t, pipe.transformer.config.patch_size)
        self.video_len = 0
        self.order = None
        self.text_len = None
        pipe.transformer.register_forward_pre_hook(self._set_grid, with_kwargs=True)

    def _reset(self):
        self.text_len = None

    def _set_grid(self, _module, args, kwargs):
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        _, _, f, h, w = x.shape
        pt, p = self.patch
        grid = (f // pt, h // p, w // p)
        self.video_len = grid[0] * grid[1] * grid[2]
        self.order = grid_morton_order(*grid, x.device)

    def is_joint(self, query, key, value, attn_mask, dropout_p, is_causal, scale, parallel_config):
        return (attn_mask is not None and query.shape == key.shape == value.shape and query.shape[1] > self.video_len
                and dropout_p == 0.0 and not is_causal and scale is None and parallel_config is None)

    def __call__(self, query, key, value, mask):
        vl = self.video_len
        B, N, H, D = query.shape
        if self.text_len is None:           # valid text tokens, once per video (one host sync)
            m = mask.reshape(B, -1)
            m = m if m.dtype == torch.bool else m > -1.0
            self.text_len = int(m[0, vl:].sum())
        L = vl + self.text_len
        q, k, v = query[:, :L], key[:, :L], value[:, :L]
        perm, inv = self.order
        rest = torch.arange(vl, L, device=q.device)
        idx = torch.cat([perm, rest])
        c = self.clock
        o = self.attention(*(x.index_select(1, idx) for x in (q, k, v)), c.step, c.branch, c.layer, sink=vl // BK)
        o = o.index_select(1, torch.cat([inv, rest]))
        out = torch.zeros_like(query)
        out[:, :vl] = o[:, :vl]
        if self.text_len:
            out[:, vl:L] = dense_attention(q[:, vl:].contiguous(), k.contiguous(), v.contiguous())
        c.layer += 1
        return out


def _dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False,
              attention_kwargs=None, *, backend=None, parallel_config=None):
    if not _joint_impl.is_joint(query, key, value, attn_mask, dropout_p, is_causal, scale, parallel_config):
        return _original_dispatch(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
                                  attention_kwargs, backend=backend, parallel_config=parallel_config)
    return _joint_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask)


def enable(pipe, attention):
    global _joint_impl
    _joint_impl = _JointAttention(pipe, attention)
    hv.dispatch_attention_fn = _dispatch
