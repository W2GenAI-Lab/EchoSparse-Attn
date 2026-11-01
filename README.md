<div align="center">

<h1>EchoSparse</h1>

<p><b>LSE-guided block-sparse attention for video diffusion transformers</b></p>

<p>
<a href="https://w2genai-lab.github.io/blog/EchoSparse/blog.html"><img alt="Blog (English)" src="https://img.shields.io/badge/Blog-English-2a78d6?style=flat-square"></a>
<a href="https://w2genai-lab.github.io/blog/EchoSparse/zh.html"><img alt="博客（中文）" src="https://img.shields.io/badge/%E5%8D%9A%E5%AE%A2-%E4%B8%AD%E6%96%87-2a78d6?style=flat-square"></a>
<a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-555?style=flat-square"></a>
<img alt="B200 (SM100)" src="https://img.shields.io/badge/B200-SM100-76B900?style=flat-square&logo=nvidia&logoColor=white">
<img alt="RTX 5090 (SM120)" src="https://img.shields.io/badge/RTX%205090-SM120-76B900?style=flat-square&logo=nvidia&logoColor=white">
<img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
<img alt="CUDA 12.8+" src="https://img.shields.io/badge/CUDA-12.8%2B-555?style=flat-square">
</p>

<p><b><a href="https://w2genai-lab.github.io/blog/EchoSparse/blog.html">Blog</a></b> &nbsp;·&nbsp; <b><a href="https://w2genai-lab.github.io/blog/EchoSparse/zh.html">中文博客</a></b> &nbsp;·&nbsp; <a href="README_zh.md">中文说明</a> &nbsp;·&nbsp; <a href="#results">Results</a> &nbsp;·&nbsp; <a href="#installation">Installation</a> &nbsp;·&nbsp; <a href="#usage">Usage</a></p>

<img src="assets/teaser.png" width="100%" alt="EchoSparse on Wan2.1-14B: 1.70x faster than FA4, closer to the dense video than Sol-Attn and PISA">

</div>

EchoSparse is block-sparse self-attention for video diffusion transformers. The softmax statistics of each
denoising step echo into the next one: they set its softmax offset, pick the blocks it computes and decide how
the skipped blocks are compensated. It needs no training and plugs into diffusers pipelines. It runs on NVIDIA
Blackwell GPUs: B200 (SM100) and RTX 5090 (SM120).

## News

- **2026-11-01** — The EchoSparse blog post is out: [English](https://w2genai-lab.github.io/blog/EchoSparse/blog.html) · [中文](https://w2genai-lab.github.io/blog/EchoSparse/zh.html).

## How it works

<p align="center">
  <img src="assets/echosparse-explainer.gif" width="85%" alt="Animated explainer: one attention layer of EchoSparse, from the carried-over LSE to the fused kernel">
  <br>
  <em>One attention layer on a toy example: the carried-over LSE, block selection, clustered representatives, the
single normalization and the fused B200 kernel, followed by the measured B200 results.</em>
</p>

Between two denoising steps, the log-sum-exp (LSE) of each attention row changes very little. EchoSparse keeps the
previous step's LSE of every layer and CFG branch and uses it in three places, all inside one attention kernel:

1. **Fixed softmax offset.** The previous LSE takes the place of the running row maximum, so the kernel does no
   max reduction and no rescaling. Tiles whose row sums leave a safe range (about 0.5% of them) are recomputed
   with online softmax. For the same block lists this cuts kernel time by about 47%.
2. **Block probabilities from the same pass.** With that offset the exp values are already close to
   probabilities, so the kernel sums them per 64-token block while it computes the selected blocks. The next step
   takes the top blocks of every 128-row query tile from this map; there is no separate scoring pass.
3. **Compensation of skipped blocks.** Skipped blocks still take part in the same softmax: each through its mean
   key and value, weighted by its token count, and the most probable third of them through four representatives
   each, found by k-means in the query metric.

The first 10 denoising steps run dense with FlashAttention-4. The last of them sends every block through the
kernel to build the first probability map.

On the B200 the kernels are written in CuTe DSL with tcgen05 MMAs and tensor memory. The RTX 5090 has neither, so
its kernels are written in Triton with ordinary tensor-core dots. They compute the same quantities in the same
precision. To fit in 32 GB, the RTX 5090 keeps per layer only the LSE of each CFG branch and the next step's block
selection, as bit masks; the probability map is scratch shared by all layers.

## Results

### RTX 5090

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-rtx5090-dark.png">
  <img alt="RTX 5090: speedup of EchoSparse over FlashAttention-4 (whole video and one sparse-step forward) and its PSNR against FA4, next to the B200 PSNR of the same prompt" src="assets/results-rtx5090.png">
</picture>

Complete 50-step generations with this package and the scripts in `examples/`, one prompt per model with the seed of
the B200 evaluation; EchoSparse and dense FA4 each run once on one GPU. The weights of Wan2.1-14B, HunyuanVideo and
MiniMax-H3 stay in CPU memory and are copied to the GPU group by group (`--offload`); Wan2.1-1.3B sits on the GPU.
One forward is the median over a run's sparse steps (EchoSparse) and dense steps (FA4). MiniMax-H3 runs at density
0.24 (`minimax_h3_rtx5090`). Sol-Attn and PISA were not run on the RTX 5090.

### B200

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-b200-dark.png">
  <img alt="B200: end-to-end speedup over FlashAttention-4 and PSNR against FA4 for EchoSparse, PISA and Sol-Attn on four models" src="assets/results-b200.png">
</picture>

Complete video generation with 50 denoising steps; all methods keep the first 10 steps dense. EchoSparse computes 10%
of KV blocks; Sol-Attn uses its paper's exact threshold estimator with τ = 1.0 and computes 15–18%; PISA computes
12.5% with covariance-aware selection. PSNR is against FA4, averaged over 10–25 prompts per model.

EchoSparse has the highest speedup on all four models. Its PSNR gains over Sol-Attn are 1.10 dB on Wan2.1-14B,
1.19 dB on Wan2.1-1.3B, and 2.44 dB on HunyuanVideo. On MiniMax-H3, the differences are +0.09 dB relative to
Sol-Attn and −0.21 dB relative to PISA; audio SNR is 4.9 dB for EchoSparse and 5.4 dB for Sol-Attn.

With Sol-Engine's diagonal threshold estimator, Sol-Attn computes 21–25% of KV blocks. In model order as above,
its PSNR is 22.34, 23.39, 25.47, and 18.70 dB, with speedups over FA4 of 1.24×, 1.17×, 1.27×, and 1.10×.
EchoSparse matches its quality on Wan, gains 1.44 dB on HunyuanVideo, and gives 0.8 dB lower PSNR on H3,
while running 1.12–1.54× faster.

## TODO List

- [ ] Adapt EchoSparse to NVIDIA H100 (SM90).
- [ ] Adapt EchoSparse to NVIDIA RTX 4090 (SM89).
- [ ] Add packed representative tokens and develop a producer-consumer architecture using TMA and a
  three-stage pipeline to reduce fragmented work and memory stalls.

The H100 experimental branch already measures a single-attention GPU pipeline at **163.48 ms with dense FA4,
71.65 ms with nonpacked EchoSparse, and 44.51 ms with grouped packing, TMA and three pipeline stages**.
For the same sparse plan and representative tensors, packing introduces no additional approximation and yields
the same output in real arithmetic.
These experimental timings use synthetic BF16 QKV `[1,75600,40,128]`, density 0.10 and a frozen sparse plan;
they measure the GPU attention pipeline, not video generation. Mainline integration and full benchmarks remain open.

## Installation

Requirements: a B200 (SM100) or RTX 5090 (SM120) GPU, CUDA 12.8 or newer, Python 3.10 or newer.

```bash
git clone https://github.com/W2GenAI-Lab/EchoSparse-Attn.git
cd EchoSparse-Attn
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[examples]"
```

Dependencies are pinned in `pyproject.toml`. On RTX 5090, EchoSparse applies the SM120 compatibility fix from
[flash-attention PR #2484](https://github.com/Dao-AILab/flash-attention/pull/2484) to FlashAttention-4.
MiniMax-H3 uses diffusers commit `abc5e9bf` and transformers 5.8.1.

For MiniMax-H3, keep its diffusers source alongside this repository:

```bash
git clone https://github.com/huggingface/diffusers.git ../diffusers-h3
git -C ../diffusers-h3 checkout abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
pip install transformers==5.8.1
export PYTHONPATH="$(pwd)/../diffusers-h3/src:${PYTHONPATH}"
```

## Usage

### Quick start

On a B200, with every weight on the GPU:

```python
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
import echosparse

model = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
pipe.to("cuda")

echosparse.enable(pipe)      # after the scheduler is set: the attention follows its steps
video = pipe(prompt="A cat surfing at sunset", height=720, width=1280, num_frames=81,
             guidance_scale=5.0, num_inference_steps=50).frames[0]
```

`enable` recognizes the model and applies its preset. Keyword arguments override the preset values `density`,
`dense_steps`, `dense_layers` and `refine_iters`. Complete scripts for the three models are in `examples/`; they
support CPU weight offloading for the 32 GB RTX 5090.

### On an RTX 5090

Wan2.1-1.3B at 480p fits on the 32 GB GPU as it is:

```bash
python examples/wan.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "A cat surfing at sunset"
```

For the larger models, `--offload` keeps weights on the CPU and copies each group to the GPU as it runs:

```bash
python examples/wan.py --prompt "A cat surfing at sunset" --offload
python examples/hunyuan_video.py --prompt "A cat surfing at sunset" --offload
python examples/minimax_h3.py --model /path/to/MiniMax-H3 --prompt "A cat surfing at sunset" \
    --preset minimax_h3_rtx5090 --offload
```

Offload groups contain one block for Wan, 16 for HunyuanVideo, and eight by default for H3; `--group-size 12
--pin` gives the same H3 forward times. `examples/minimax_h3.py` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to limit fragmentation of the CUDA heap on a 32 GB GPU.

The attention state kept between steps occupies about 1.89 GB for Wan2.1-14B (both CFG branches), 1.37 GB for
HunyuanVideo and 0.63 GB for MiniMax-H3 at the resolutions above, plus one shared work buffer. It stays on the GPU
throughout inference.

To insist on one GPU type, pass `gpu="b200"` or `gpu="rtx5090"`; a call on a different GPU then raises an error.

### Presets

| Model | Density | Dense steps | Dense layers | Notes |
|---|---|---|---|---|
| Wan2.1 | 0.10 | 10 | 0 | video tokens in 3D Morton order |
| HunyuanVideo | 0.10 | 10 | none | text tokens are computed exactly by every tile; text query rows use FA4 |
| MiniMax-H3 | 0.10 | 10 | 0 | the packed prefix is computed exactly by every tile; prefix query rows use FA4 |

The `minimax_h3_rtx5090` preset uses density 0.24, the setting of the RTX 5090 measurements above:

```python
echosparse.enable(pipe, preset="minimax_h3_rtx5090")      # density 0.24, 10 dense steps, layer 0 dense
```

### Other models

`echosparse.EchoSparseAttention` can be called directly:

```python
attn = echosparse.EchoSparseAttention(density=0.10, dense_steps=10, dense_layers=(0,))
attn.reset()                                           # before each video
out = attn(q, k, v, step=step, branch=branch, layer=layer)
```

`q`, `k`, `v` are `[1, L, heads, 128]` bf16 tensors of one self-attention call (batch size 1, queries and keys of
the same length; sequences shorter than 4096 tokens run dense). `branch` counts the transformer forwards inside a
denoising step (0 for the conditional pass, 1 for the unconditional one); later branches reuse the block selection
of branch 0. Tokens that every query should attend to exactly (text, conditioning) go at the end of the sequence,
with `sink` set to the index of their first 64-token block. If the caller recomputes some trailing query rows
itself (as the MiniMax-H3 integration does for the prefix), `rows=n` tells the attention that only the first `n`
output rows are used; the RTX 5090 kernels then skip the query tiles past them. `enable` patches diffusers'
attention dispatch of the model, so use one pipeline per process.

## License

Apache License 2.0. See [NOTICE](NOTICE) for third-party attributions.

## Core contributors

Tian Ye and Shitong Shao.
