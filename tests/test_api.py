"""GPU selection and presets; runs without a GPU."""
import subprocess
import sys
import types

import pytest
import torch

import echosparse
from echosparse import backends
from echosparse.models import PRESETS, settings


@pytest.fixture
def capability(monkeypatch):
    def set_capability(major, minor=0):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (major, minor))
    return set_capability


def test_auto_follows_the_device(capability):
    capability(9)
    assert backends.resolve("auto", "cuda:0") == "h100"
    capability(10)
    assert backends.resolve("auto", "cuda:0") == "b200"
    capability(12)
    assert backends.resolve("auto", "cuda:1") == "rtx5090"
    assert backends.resolve("rtx5090", torch.device("cuda", 0)) == "rtx5090"


def test_mismatch_and_unsupported_devices_fail(capability):
    capability(12)
    with pytest.raises(RuntimeError, match="gpu='b200' needs compute capability 10.x.*use gpu='auto' or gpu='rtx5090'"):
        backends.resolve("b200", "cuda:0")
    capability(8)
    with pytest.raises(RuntimeError, match="capability 8.0; EchoSparse supports"):
        backends.resolve("auto", "cuda:0")
    with pytest.raises(RuntimeError, match="no EchoSparse backend supports"):
        backends.resolve("rtx5090", "cuda:0")
    with pytest.raises(ValueError, match="CUDA tensors"):
        backends.resolve("auto", "cpu")
    with pytest.raises(ValueError, match="gpu must be"):
        backends.resolve("a100", "cuda:0")


def test_attention_resolves_at_the_first_call():
    attn = echosparse.EchoSparseAttention(gpu="rtx5090")         # no GPU needed to construct
    assert attn.backend is None
    q = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA tensors"):
        attn(q, q, q, step=0, branch=0, layer=0)
    with pytest.raises(ValueError, match="gpu must be"):
        echosparse.EchoSparseAttention(gpu="5090")


def test_presets():
    assert settings("wan") == PRESETS["wan"]
    assert settings("minimax_h3")["density"] == 0.10
    assert settings("minimax_h3", "minimax_h3_rtx5090")["density"] == 0.24
    assert settings("minimax_h3", "minimax_h3_rtx5090", density=0.3)["density"] == 0.3
    with pytest.raises(ValueError, match="not a wan preset"):
        settings("wan", "minimax_h3_rtx5090")


def test_device_change_releases_old_history(monkeypatch, capability):
    import echosparse.attention as attention

    capability(12)
    monkeypatch.setattr(backends, "kernels", lambda name: None)
    monkeypatch.setattr(attention, "dense_attention", lambda q, k, v: q)
    attn = echosparse.EchoSparseAttention()
    q = types.SimpleNamespace(shape=(1, 128, 1, 128), device=torch.device("cuda:0"))
    attn(q, q, q, step=0, branch=0, layer=0)
    previous = object()
    attn._states[(0, 1)] = previous
    attn._buffers = attn._plan_scratch = previous
    attn(q, q, q, step=0, branch=0, layer=0)
    assert attn._states[(0, 1)] is previous
    assert attn._buffers is previous
    q.device = torch.device("cuda:1")
    attn(q, q, q, step=0, branch=0, layer=0)
    assert not attn._states
    assert attn._buffers is None and attn._plan_scratch is None


def test_enable_and_attention_agree(monkeypatch):
    seen = {}
    fake = types.ModuleType("echosparse.models.minimax_h3")
    fake.enable = lambda pipe, attention: seen.update(attention=attention)
    monkeypatch.setitem(sys.modules, "echosparse.models.minimax_h3", fake)
    monkeypatch.delattr(echosparse.models, "minimax_h3", raising=False)
    pipe = types.SimpleNamespace(transformer=type("MiniMaxH3Transformer", (), {})())
    attn = echosparse.enable(pipe, preset="minimax_h3_rtx5090", gpu="rtx5090", refine_iters=6)
    assert seen["attention"] is attn
    assert (attn.density, attn.dense_steps, attn.dense_layers, attn.gpu) == (0.24, 10, {0}, "rtx5090")
    with pytest.raises(ValueError, match="gpu must be"):
        echosparse.enable(pipe, gpu="sm120")


def test_import_does_not_load_kernels():
    code = "import sys, echosparse; print(any(m.startswith('echosparse.kernels.') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout == "False\n"


def test_offload_blocks_restores_the_cpu_masters():
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.embed = torch.nn.Linear(8, 8)
    model.blocks = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(5))
    model.register_buffer("scale", torch.ones(8))

    def forward(x):
        x = model.embed(x) * model.scale
        for block in model.blocks:
            x = block(x)
        return x

    model.forward = forward
    x = torch.randn(2, 8)
    expected = model(x)
    echosparse.offload_blocks(model, "cpu", blocks_per_group=2)
    masters = [p.data_ptr() for p in model.parameters()]
    assert torch.equal(model(x), expected)
    assert [p.data_ptr() for p in model.parameters()] == masters
    assert len(model._forward_pre_hooks) == len(model._forward_hooks) == 1             # embed: whole forward
    assert len(model.blocks[4]._forward_pre_hooks) == len(model.blocks[4]._forward_hooks) == 1    # a group of one


@pytest.mark.parametrize("L, H, layers, branches, expected", [
    (75600, 40, 39, 2, 1888174080),      # Wan2.1-14B 720p, layers 1-39, CFG
    (118839, 24, 60, 1, 1369866240),     # HunyuanVideo 720p: 118800 video + 39 text tokens
    (38247, 56, 49, 1, 630110208),       # MiniMax-H3 768p, layers 1-49
])
def test_rtx5090_state_size(L, H, layers, branches, expected):
    from echosparse.attention import _CompactState

    st = _CompactState(L, H, "meta")
    for branch in range(branches):
        st.offset(branch)
    for plan in range(branches):
        st.plan(plan)
    tensors = [*st.offsets.values(), *(t for p in st.plans for t in p)]
    assert layers * sum(t.numel() * t.element_size() for t in tensors) == expected


def test_h3_layout_packs_the_target_video_first(monkeypatch):
    fake = types.ModuleType("diffusers.models.transformers.transformer_minimax_h3")
    fake.dispatch_attention_fn = fake._apply_rotary_emb = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.delitem(sys.modules, "echosparse.models.minimax_h3", raising=False)
    from echosparse.models.common import morton_order
    from echosparse.models.minimax_h3 import _Layout

    # prefix: 3 text, 4 conditioning-video, 2 audio tokens; target video: 2 x 3 x 4 grid with a strided t axis
    t, y, x = torch.meshgrid(torch.arange(0, 4, 2), torch.arange(3), torch.arange(4), indexing="ij")
    video = torch.stack([t.flatten(), y.flatten(), x.flatten()], 1)
    pos = torch.cat([torch.zeros(9, 3, dtype=torch.long), video])
    L, vs = pos.shape[0], 9
    kw = dict(position_ids=pos, token_tags=torch.arange(L), timestep_indices=torch.arange(L) % 2,
              text_indices=torch.arange(3), video_indices=torch.cat([torch.arange(3, 7), torch.arange(vs, L)]),
              audio_indices=torch.arange(7, 9))
    layout = _Layout(pos, kw["video_indices"])
    new = layout.remap(kw)
    perm, _ = morton_order(video[:, 0] // 2, video[:, 1], video[:, 2])
    assert layout.video == L - vs and torch.equal(layout.packed_to_original[:L - vs], perm + vs)
    assert torch.equal(layout.packed_to_original[L - vs:], torch.arange(vs))           # prefix keeps its order
    assert torch.equal(new["position_ids"], pos[layout.packed_to_original])
    for name in ("text_indices", "video_indices", "audio_indices"):                     # same tokens, new positions
        assert torch.equal(layout.packed_to_original[new[name]], kw[name])


@pytest.mark.parametrize("gpu,major,packed", [("b200", 10, False), ("rtx5090", 12, True),
                                             ("auto", 10, False), ("auto", 12, True)])
def test_h3_keeps_b200_layout_and_packs_only_sm120(monkeypatch, capability, gpu, major, packed):
    import importlib

    fake = types.ModuleType("diffusers.models.transformers.transformer_minimax_h3")
    fake.dispatch_attention_fn = fake._apply_rotary_emb = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.delitem(sys.modules, "echosparse.models.minimax_h3", raising=False)
    h3 = importlib.import_module("echosparse.models.minimax_h3")
    monkeypatch.setattr(h3, "th3", fake)
    capability(major)
    monkeypatch.setattr(h3, "dense_attention", lambda q, k, v: q.clone())

    class InputOnCuda(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        @property
        def device(self):
            return torch.device("cuda:1")

    class Transformer(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    class Attention:
        def __init__(self):
            self.gpu = gpu
            self.received = None

        def reset(self):
            pass

        def __call__(self, q, k, v, *args, **kwargs):
            self.received = q.clone()
            return q.clone()

    scheduler = types.SimpleNamespace(set_timesteps=lambda: None, step=lambda: None)
    pipe = types.SimpleNamespace(transformer=Transformer(), scheduler=scheduler)
    attention = Attention()
    h3.enable(pipe, attention)
    t, y, x = torch.meshgrid(torch.arange(2), torch.arange(3), torch.arange(4), indexing="ij")
    pos = torch.cat([torch.zeros(3, 3, dtype=torch.long), torch.stack([t.flatten(), y.flatten(), x.flatten()], 1)])
    kwargs = dict(position_ids=pos, token_tags=torch.arange(len(pos)), timestep_indices=torch.zeros(len(pos)),
                  video_indices=torch.arange(3, len(pos)), text_indices=torch.arange(3),
                  audio_indices=torch.empty(0, dtype=torch.long), hidden_states=torch.empty(0).as_subclass(InputOnCuda))
    forwarded = pipe.transformer(**kwargs)
    layout = h3._impl.layout
    assert h3._impl.packed is packed
    assert torch.equal(forwarded["position_ids"], pos[layout.packed_to_original] if packed else pos)
    if not packed:
        assert forwarded["position_ids"] is pos
    assert h3.th3._apply_rotary_emb is (h3._rotary if packed else h3._original_rotary)
    q = torch.arange(len(pos)).view(1, -1, 1, 1).float()
    output = h3._impl(q, q, q)
    assert torch.equal(output, q)
    assert torch.equal(attention.received, q if packed else q.index_select(1, layout.packed_to_original))
