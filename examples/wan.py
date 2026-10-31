"""Wan2.1 text-to-video with EchoSparse attention.

    python examples/wan.py --model Wan-AI/Wan2.1-T2V-14B-Diffusers --prompt "A cat surfing at sunset" --out cat.mp4
    python examples/wan.py --prompt "A cat surfing at sunset" --offload      # 14B on a 32 GB RTX 5090
"""
import argparse

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

import echosparse

# official sampling settings of each model size
SETTINGS = {
    "1.3B": dict(height=480, width=832, guidance_scale=6.0, flow_shift=3.0),
    "14B": dict(height=720, width=1280, guidance_scale=5.0, flow_shift=5.0),
}
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="wan.mp4")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=None, help="fraction of kv blocks computed exactly")
    ap.add_argument("--compile", action="store_true", help="torch.compile the transformer blocks")
    ap.add_argument("--offload", action="store_true", help="keep the weights in CPU memory, one block at a time on the GPU")
    args = ap.parse_args()

    cfg = SETTINGS["1.3B" if "1.3B" in args.model else "14B"]
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=cfg["flow_shift"])
    if args.offload:
        cuda = torch.device("cuda")
        pipe.transformer.enable_group_offload(cuda, offload_type="block_level", num_blocks_per_group=1)
        pipe.vae.enable_group_offload(cuda, offload_type="leaf_level")
        apply_group_offloading(pipe.text_encoder, onload_device=cuda, offload_type="leaf_level")
    else:
        pipe.to("cuda")
    # after the scheduler is set: the attention follows its steps
    echosparse.enable(pipe, **({} if args.density is None else {"density": args.density}))
    if args.compile:
        pipe.transformer.compile_repeated_blocks(fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")

    frames = pipe(prompt=args.prompt, negative_prompt=NEGATIVE_PROMPT, height=cfg["height"], width=cfg["width"],
                  num_frames=81, guidance_scale=cfg["guidance_scale"], num_inference_steps=args.steps,
                  generator=torch.Generator("cuda").manual_seed(args.seed)).frames[0]
    export_to_video(frames, args.out, fps=16)


if __name__ == "__main__":
    main()
