"""Transformer weights in CPU memory, copied to the GPU one group of blocks at a time.

This only moves weights; the attention and its per-layer state stay on the GPU.
"""
import torch


def _group(params, device, pin):
    params = list({id(p): p for p in params}.values())
    for p in params:
        p.data = p.data.cpu().pin_memory() if pin else p.data.cpu()
    masters = [p.data for p in params]

    def load(*_):
        for p, m in zip(params, masters):
            p.data = m.to(device, non_blocking=False)

    def release(*_):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for p, m in zip(params, masters):
            p.data = m

    return params, load, release


def offload_blocks(transformer, device="cuda", blocks_per_group=1, pin=False):
    """Keep the transformer's parameters in CPU memory. Each group of `blocks_per_group` consecutive blocks (of its
    ModuleList children) is copied to `device` just before it runs, the other parameters (embeddings, output layers)
    for the whole forward. The CPU tensors stay the masters: afterwards the parameters point back to them, so nothing
    is copied back. Buffers are moved to `device` once. `pin`: page-locked masters (faster copies, more pinned host
    memory). For inference only: the weights must not change."""
    device = torch.device(device)
    grouped = set()
    for blocks in (m for m in transformer.children() if isinstance(m, torch.nn.ModuleList)):
        for i in range(0, len(blocks), blocks_per_group):
            group = blocks[i:i + blocks_per_group]
            params, load, release = _group((p for block in group for p in block.parameters()), device, pin)
            group[0].register_forward_pre_hook(load)
            group[-1].register_forward_hook(release)
            grouped.update(id(p) for p in params)
    _, load, release = _group((p for p in transformer.parameters() if id(p) not in grouped), device, pin)
    transformer.register_forward_pre_hook(load, prepend=True)
    transformer.register_forward_hook(release)
    for module in transformer.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
    return transformer
