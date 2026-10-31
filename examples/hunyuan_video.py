"""HunyuanVideo text-to-video with EchoSparse attention.

    python examples/hunyuan_video.py --prompt "A cat surfing at sunset" --out cat.mp4
    python examples/hunyuan_video.py --prompt "A cat surfing at sunset" --offload      # on a 32 GB RTX 5090
"""
import argparse

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

import echosparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hunyuanvideo-community/HunyuanVideo")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="hunyuan_video.mp4")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--density", type=float, default=None, help="fraction of kv blocks computed exactly")
    ap.add_argument("--offload", action="store_true", help="keep the weights in CPU memory, 16 blocks at a time on the GPU")
    args = ap.parse_args()

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.model, subfolder="transformer",
                                                                 torch_dtype=torch.bfloat16)
    pipe = HunyuanVideoPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    if args.offload:
        cuda = torch.device("cuda")
        echosparse.offload_blocks(pipe.transformer, cuda, blocks_per_group=16)
        for name in ("text_encoder", "text_encoder_2", "vae"):
            apply_group_offloading(getattr(pipe, name), onload_device=cuda, offload_type="leaf_level")
    else:
        pipe.to("cuda")
    echosparse.enable(pipe, **({} if args.density is None else {"density": args.density}))

    frames = pipe(prompt=args.prompt, height=720, width=1280, num_frames=129, num_inference_steps=args.steps,
                  guidance_scale=6.0, true_cfg_scale=1.0, max_sequence_length=256,
                  generator=torch.Generator("cuda").manual_seed(args.seed)).frames[0]
    export_to_video(frames, args.out, fps=24)


if __name__ == "__main__":
    main()
