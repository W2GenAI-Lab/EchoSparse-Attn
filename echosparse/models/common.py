import functools

import torch


class Clock:
    """Denoising step, transformer forward within the step (CFG branch) and attention layer of the current call.

    It follows the pipeline's own calls: `scheduler.set_timesteps` starts a video, `scheduler.step` ends a step and
    every transformer forward starts a branch. The attention dispatch advances `layer`."""

    def __init__(self, pipe, on_reset=()):
        self.step, self.branch, self.layer = 0, -1, 0
        self._on_reset = list(on_reset)
        scheduler = pipe.scheduler
        set_timesteps, step = scheduler.set_timesteps, scheduler.step

        @functools.wraps(set_timesteps)
        def set_timesteps_and_reset(*args, **kwargs):
            self.step, self.branch = 0, -1
            for fn in self._on_reset:
                fn()
            return set_timesteps(*args, **kwargs)

        @functools.wraps(step)
        def step_and_advance(*args, **kwargs):
            self.step, self.branch = self.step + 1, -1
            return step(*args, **kwargs)

        scheduler.set_timesteps = set_timesteps_and_reset
        scheduler.step = step_and_advance
        pipe.transformer.register_forward_pre_hook(self._begin_forward)

    def _begin_forward(self, _module, _args):
        self.branch += 1
        self.layer = 0


def morton_order(t, y, x):
    """Permutation that sorts tokens with coordinates (t, y, x) along a 3D Morton curve, and its inverse."""
    code = torch.zeros_like(x)
    for bit in range(21):
        code |= ((x >> bit) & 1) << (3 * bit) | ((y >> bit) & 1) << (3 * bit + 1) | ((t >> bit) & 1) << (3 * bit + 2)
    perm = torch.argsort(code)
    return perm, torch.argsort(perm)


@functools.lru_cache(maxsize=8)
def grid_morton_order(frames, height, width, device):
    """Morton order of a (frames, height, width) token grid stored in raster order."""
    t, y, x = torch.meshgrid(torch.arange(frames), torch.arange(height), torch.arange(width), indexing="ij")
    perm, inv = morton_order(t.flatten(), y.flatten(), x.flatten())
    return perm.to(device), inv.to(device)
