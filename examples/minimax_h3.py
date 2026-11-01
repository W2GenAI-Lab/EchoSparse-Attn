"""MiniMax-H3 text-to-video-and-audio with EchoSparse attention.

Needs a diffusers build with MiniMax-H3 support (see the README).

    python examples/minimax_h3.py --prompt "..." --out h3.mp4
    python examples/minimax_h3.py --prompt "..." --preset minimax_h3_rtx5090 --offload
                                                                       # the RTX 5090 setting, on a 32 GB card
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import wave

# Offloading on a 32 GB GPU leaves little free memory; expandable segments limit fragmentation of the CUDA heap.
# Set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

import echosparse


def load_pipeline(model):
    local = Path(model).is_dir()
    index = Path(model) / "model_index.json" if local else Path(hf_hub_download(model, "model_index.json"))
    config = json.loads(index.read_text())
    config.pop("transformer_ref", None)
    if local:
        for name, spec in config.items():
            if not name.startswith("_") and isinstance(spec, list) and len(spec) == 3:
                spec[2].update(pretrained_model_name_or_path=str(Path(model).resolve()), revision=None)
    with tempfile.TemporaryDirectory(prefix="echosparse-h3-") as directory:
        Path(directory, "modular_model_index.json").write_text(json.dumps(config))
        return ModularPipeline.from_pretrained(directory, components_manager=ComponentsManager())


def offload_auxiliary(module, device):
    """Keep text/VAE weights backed by the checkpoint rather than making CPU copies."""
    apply_group_offloading(module, onload_device=device, offload_type="leaf_level",
                           non_blocking=False, use_stream=False)
    masters = {id(p): p.data for p in module.parameters()}
    owners = {key: set() for key in masters}
    groups = {}
    for child in module.modules():
        registry = getattr(child, "_diffusers_hook", None)
        hook = registry.get_hook("group_offloading") if registry else None
        if hook is None or id(hook.group) in groups:
            continue
        group = hook.group
        params = {id(p): p for m in group.modules for p in m.parameters()}
        params.update((id(p), p) for p in group.parameters)
        buffers = {id(b): b for m in group.modules for b in m.buffers()}
        buffers.update((id(b), b) for b in group.buffers)

        def load(params=params, buffers=buffers, key=id(group)):
            for ident, parameter in params.items():
                if not owners[ident]:
                    parameter.data = masters[ident].to(device, non_blocking=False)
                owners[ident].add(key)
            for buffer in buffers.values():
                buffer.data = buffer.data.to(device, non_blocking=False)

        def release(params=params, buffers=buffers, key=id(group)):
            torch.cuda.synchronize(device)
            for ident, parameter in params.items():
                owners[ident].remove(key)
                if not owners[ident]:
                    parameter.data = masters[ident]
            for buffer in buffers.values():
                buffer.data = buffer.data.cpu()

        group.onload_, group.offload_ = load, release
        groups[id(group)] = group


def offload_audio(module, device):
    """Run legacy WeightNorm after its g/v parameters have reached the GPU."""
    from torch.nn.utils.weight_norm import WeightNorm
    apply_group_offloading(module, onload_device=device, offload_type="leaf_level",
                           non_blocking=False, use_stream=False)
    for child in module.modules():
        for key, norm in list(child._forward_pre_hooks.items()):
            if not isinstance(norm, WeightNorm):
                continue
            del child._forward_pre_hooks[key]
            ref = child._diffusers_hook.get_hook("group_offloading").fn_ref
            original = ref.forward

            def forward(*args, child=child, norm=norm, original=original, **kwargs):
                try:
                    norm(child, args)
                    return original(*args, **kwargs)
                finally:
                    setattr(child, norm.name, None)

            ref.forward = forward
            setattr(child, norm.name, None)


def write_wav(path, audio, rate):
    pcm = (np.clip(audio, -1, 1).T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="h3.mp4")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default=None, help="an entry of echosparse.PRESETS (default: minimax_h3)")
    ap.add_argument("--density", type=float, default=None, help="fraction of kv blocks computed exactly")
    ap.add_argument("--offload", action="store_true",
                    help="keep weights on the CPU and copy each group to the GPU for its forward")
    ap.add_argument("--group-size", type=int, default=8, help="transformer blocks per offload group")
    ap.add_argument("--pin", action="store_true", help="pin the transformer CPU weights for faster transfers")
    args = ap.parse_args()

    pipe = load_pipeline(args.model)
    if args.offload:
        cuda = torch.device("cuda")
        pipe.load_components(names=["scheduler", "audio_scheduler", "tokenizer", "processor"])
        # Load separately to keep the two large checkpoints memory-mapped.
        for name in ("text_encoder", "vae"):
            pipe.load_components(names=name, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            offload_auxiliary(getattr(pipe, name), cuda)
        pipe.load_components(names="audio_vae", dtype=torch.bfloat16)
        offload_audio(pipe.audio_vae, cuda)
        pipe.load_components(names="transformer", dtype=torch.bfloat16, low_cpu_mem_usage=True)
        echosparse.offload_blocks(pipe.transformer, cuda, blocks_per_group=args.group_size, pin=args.pin)
    else:
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to("cuda")
    echosparse.enable(pipe, preset=args.preset, **({} if args.density is None else {"density": args.density}))

    state = pipe(prompt=args.prompt, height=768, width=1344, num_frames=124, num_inference_steps=args.steps,
                 generator=torch.Generator().manual_seed(args.seed), output_type="np")
    export_to_video(list(np.asarray(state.get("videos")[0])), args.out, fps=24)
    audio = state.get("audio")
    audio = audio[0].float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)[0]
    write_wav(args.out.rsplit(".", 1)[0] + ".wav", audio, state.get("sampling_rate"))


if __name__ == "__main__":
    main()
