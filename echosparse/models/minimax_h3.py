"""MiniMax-H3 text-to-video-and-audio (the minimax-h3 branch of diffusers, `ModularPipeline`).

H3 attends over one packed sequence [text | conditioning | audio | target video] without a mask. Everything
before the target-video tail is the prefix: its keys are a sink that every tile computes exactly and its own query
rows use FA4, so EchoSparse attention only needs the video rows. Target-video rows go through EchoSparse attention
in Morton order, taken from their rotary (t, h, w) coordinates. Attention calls of other lengths (text-only rows
before packing) keep diffusers' dispatcher. The checkpoint is guidance-distilled: one transformer forward per step.

On the RTX 5090 the token layout is set once per forward: a pre-hook remaps the model's index arguments (positions, token tags,
timestep and modality indices), so the model scatters its tokens straight into [target video in Morton order |
prefix] and every layer's q / k / v arrive in the attention's order. The modality index vectors keep their element
order, so the outputs come back in H3's own order. The prefix query rows attend to the keys in their original
order, as without the layout, and rotary embedding runs as one kernel with the same BF16 rounding. B200 keeps the
original per-layer token gathers and Diffusers rotary embedding; so does the H100.
"""
import torch
from diffusers.models.transformers import transformer_minimax_h3 as th3

from .. import backends
from ..attention import BK, dense_attention
from .common import Clock, morton_order

_impl = None
_original_dispatch = th3.dispatch_attention_fn
_original_rotary = th3._apply_rotary_emb
_SM120 = {}


class _Layout:
    def __init__(self, position_ids, video_indices):
        L = int(position_ids.shape[0])
        # the target video is the contiguous tail of video_indices (one host sync per layout)
        breaks = ((video_indices[1:] - video_indices[:-1]) != 1).nonzero()
        start = int(breaks[-1].item()) + 1 if breaks.numel() else 0
        vs = int(video_indices[start].item())
        # rotary coordinates are not consecutive integers (t is strided): rank each axis first
        t, y, x = (torch.unique(position_ids[vs:, i], return_inverse=True)[1].long() for i in range(3))
        perm, _ = morton_order(t, y, x)
        self.L, self.video, self.video_start = L, L - vs, vs
        self.positions = position_ids.clone()
        self.packed_to_original = torch.cat([perm + vs, torch.arange(vs, device=perm.device)])
        self.original_to_packed = torch.argsort(self.packed_to_original)

    def remap(self, kwargs):
        new = dict(kwargs)
        for name in ("position_ids", "token_tags", "timestep_indices"):          # one entry per token
            new[name] = kwargs[name].index_select(0, self.packed_to_original)
        for name in ("video_indices", "audio_indices", "text_indices"):          # token positions
            new[name] = self.original_to_packed.index_select(0, kwargs[name])
        return new


class _PackedAttention:
    def __init__(self, pipe, attention):
        self.attention = attention
        self.clock = Clock(pipe, on_reset=[attention.reset])
        self.layout = None
        self.packed = None
        pipe.transformer.register_forward_pre_hook(self._set_layout, with_kwargs=True)

    @property
    def seq_len(self):
        return None if self.layout is None else self.layout.L

    def _set_layout(self, _module, args, kwargs):
        if self.packed is None:
            gpu = self.attention.gpu
            if gpu == "auto":
                device = next(t.device for t in (*args, *kwargs.values()) if torch.is_tensor(t) and t.is_cuda)
                gpu = backends.resolve(gpu, device)
            self.packed = gpu == "rtx5090"
            if self.packed:
                th3._apply_rotary_emb = _rotary
        pos = kwargs["position_ids"]
        if (self.layout is None or pos.shape[0] != self.layout.L
                or self.packed and not torch.equal(pos, self.layout.positions)):
            self.layout = _Layout(pos, kwargs["video_indices"])
        return (args, self.layout.remap(kwargs)) if self.packed else None

    def __call__(self, q, k, v):
        nv, back = self.layout.video, self.layout.original_to_packed
        c = self.clock
        if self.packed:
            out = self.attention(q, k, v, c.step, c.branch, c.layer, sink=nv // BK, rows=nv)
            out[:, nv:] = dense_attention(q[:, nv:].contiguous(), k.index_select(1, back), v.index_select(1, back))
        else:
            idx, vs = self.layout.packed_to_original, self.layout.video_start
            o = self.attention(*(x.index_select(1, idx) for x in (q, k, v)), c.step, c.branch, c.layer,
                               sink=nv // BK, rows=nv)
            out = torch.empty_like(q)
            out[:, vs:] = o[:, :nv].index_select(1, back[vs:])
            out[:, :vs] = dense_attention(q[:, :vs].contiguous(), k, v)
        c.layer += 1
        return out


def _dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False,
              attention_kwargs=None, *, backend=None, parallel_config=None):
    if query.shape[1] != _impl.seq_len:
        return _original_dispatch(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                                  scale=scale, enable_gqa=enable_gqa, attention_kwargs=attention_kwargs,
                                  backend=backend, parallel_config=parallel_config)
    assert attn_mask is None and dropout_p == 0.0 and not is_causal and scale is None and parallel_config is None
    return _impl(query.contiguous(), key.contiguous(), value.contiguous())


def _rotary(hidden_states, cos, sin):
    x = hidden_states
    if x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == 128 and cos.shape[-1] == 96:
        if x.device not in _SM120:
            _SM120[x.device] = torch.cuda.get_device_capability(x.device)[0] == 12
        if _SM120[x.device]:
            from ..kernels.sm120 import h3_rotary
            return h3_rotary(x, cos, sin)
    return _original_rotary(hidden_states, cos, sin)


def enable(pipe, attention):
    global _impl
    _impl = _PackedAttention(pipe, attention)
    th3.dispatch_attention_fn = _dispatch
    th3._apply_rotary_emb = _original_rotary
