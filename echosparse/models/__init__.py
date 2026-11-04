from ..attention import EchoSparseAttention

# Settings used for the reported results; the defaults are the same on every GPU. "minimax_h3_rtx5090" is the
# MiniMax-H3 setting for the RTX 5090 (enable(pipe, preset="minimax_h3_rtx5090")).
PRESETS = {
    "wan": dict(density=0.10, dense_steps=10, dense_layers=(0,)),
    "hunyuan_video": dict(density=0.10, dense_steps=10, dense_layers=()),
    "minimax_h3": dict(density=0.10, dense_steps=10, dense_layers=(0,)),
    "minimax_h3_rtx5090": dict(density=0.24, dense_steps=10, dense_layers=(0,)),
}


def model_family(pipe):
    name = type(pipe.transformer).__name__
    if name.startswith("WanTransformer"):
        return "wan"
    if name.startswith("HunyuanVideoTransformer"):
        return "hunyuan_video"
    if name.startswith("MiniMaxH3"):
        return "minimax_h3"
    raise ValueError(f"unsupported transformer {name}")


def settings(family, preset=None, **overrides):
    """Attention settings of a model family: its preset (or the named one) with `overrides` applied."""
    preset = preset or family
    if preset not in PRESETS or not preset.startswith(family):
        raise ValueError(f"preset {preset!r} is not a {family} preset; have "
                         f"{sorted(p for p in PRESETS if p.startswith(family))}")
    return {**PRESETS[preset], **overrides}


def enable(pipe, preset=None, gpu="auto", **overrides):
    """Route the pipeline's self-attention through EchoSparse attention; returns the attention object.

    `preset` names an entry of PRESETS (default: the model's own); `overrides` replace its values (density,
    dense_steps, dense_layers, refine_iters). `gpu` ("auto", "h100", "b200" or "rtx5090") selects the kernels; with
    "auto" they follow the device the attention runs on, so CPU offloading works."""
    family = model_family(pipe)
    attention = EchoSparseAttention(**settings(family, preset, **overrides), gpu=gpu)
    if family == "wan":
        from . import wan
        wan.enable(pipe, attention)
    elif family == "hunyuan_video":
        from . import hunyuan_video
        hunyuan_video.enable(pipe, attention)
    else:
        from . import minimax_h3
        minimax_h3.enable(pipe, attention)
    return attention
