"""Metadata is in pyproject.toml. This file only adds the optional native VSA / VSA2 extension
(echosparse.vsa._C), built when ECHOSPARSE_VSA_BUILD=1 and otherwise skipped (no nvcc or torch needed):

  RTX 5090: ECHOSPARSE_VSA_BUILD=1 TORCH_CUDA_ARCH_LIST=12.0 pip install -e . --no-build-isolation
  H100:     ECHOSPARSE_VSA_BUILD=1 ECHOSPARSE_VSA_TK=1 TORCH_CUDA_ARCH_LIST=9.0a pip install -e . --no-build-isolation

ECHOSPARSE_VSA_TK=1 adds the ThunderKittens fine kernels, from ECHOSPARSE_VSA_TK_ROOT (default
third_party/ThunderKittens, checked out at TK_COMMIT). ECHOSPARSE_VSA_PTXAS_VERBOSE=1 prints register use.
"""
import os
import shutil
import subprocess

from setuptools import setup

ROOT = os.path.dirname(os.path.abspath(__file__))
TK_COMMIT = "6c27e28c8115d1839d9eeeb530913c184a75fc87"  # the ThunderKittens revision pinned by FastVideo 661d18ba


def libcuda_link_flags(cuda_home):
    """(library_dirs, libraries) for the CUDA driver API, needed only by the TK path.

    ThunderKittens calls the driver API (cuTensorMapEncodeTiled, cuGetErrorString), which
    CUDAExtension's -lcudart does not provide. Prefer the toolkit's link-time stub: its
    SONAME is libcuda.so.1, so at run time the loader binds to the installed driver's library
    with no RPATH or environment change. Do not add the stubs directory to any run-time path.
    ECHOSPARSE_VSA_LIBCUDA_DIR overrides the search.
    """
    dirs = [os.environ.get("ECHOSPARSE_VSA_LIBCUDA_DIR", "")]
    dirs += [os.path.join(cuda_home, d) for d in ("lib64/stubs", "lib/stubs")] if cuda_home else []
    dirs += ["/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib"]
    dirs = [d for d in dirs if d and os.path.isdir(d)]
    for d in dirs:
        if os.path.exists(os.path.join(d, "libcuda.so")):
            return [d], ["cuda"]
    for d in dirs:
        if os.path.exists(os.path.join(d, "libcuda.so.1")):
            return [d], [":libcuda.so.1"]
    raise RuntimeError(f"ECHOSPARSE_VSA_TK=1 needs libcuda to link; searched {dirs}. Set ECHOSPARSE_VSA_LIBCUDA_DIR.")


def check_tk(tk_root):
    """Headers and license must be present. A Git checkout of ThunderKittens itself must be at TK_COMMIT; an
    exported tree (as in a source distribution) has no Git metadata of its own and is used as it is, without
    asking an enclosing repository (e.g. EchoSparse's) for a HEAD."""
    for f in ("include/kittens.cuh", "LICENSE"):
        if not os.path.isfile(os.path.join(tk_root, f)):
            raise RuntimeError(f"ThunderKittens not found at {tk_root} (missing {f}); see the README")
    if not os.path.exists(os.path.join(tk_root, ".git")):
        return
    if not shutil.which("git"):
        raise RuntimeError(f"{tk_root} is a Git checkout, but git is not available to check it is at {TK_COMMIT}")

    def git(*args):
        return subprocess.run(["git", "-C", tk_root, *args], capture_output=True, text=True, check=True).stdout.strip()

    top = git("rev-parse", "--show-toplevel")
    if os.path.realpath(top) != os.path.realpath(tk_root):
        raise RuntimeError(f"{tk_root} has Git metadata but its repository top level is {top}")
    head = git("rev-parse", "HEAD")
    if head != TK_COMMIT:
        raise RuntimeError(f"ThunderKittens at {tk_root} is {head}, expected {TK_COMMIT}")


def vsa_extension():
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    with_tk = os.environ.get("ECHOSPARSE_VSA_TK", "0") == "1"
    src = "csrc/vsa"
    sources = [f"{src}/{f}" for f in ("vsa2_api.cpp", "generic/attn_fwd.cu", "generic/attn_bwd.cu", "router/pool.cu",
                                       "router/router.cu", "router/topk.cu", "router/scores.cu", "fuse.cu")]
    include_dirs = [os.path.join(ROOT, src)]
    cxx = ["-O3", "-std=c++17"]
    nvcc = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
            "-Xptxas=--warn-on-spills", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    if os.environ.get("ECHOSPARSE_VSA_PTXAS_VERBOSE", "0") == "1":
        nvcc.append("-Xptxas=-v")
    library_dirs, libraries = [], []
    if with_tk:
        tk_root = os.path.abspath(os.environ.get("ECHOSPARSE_VSA_TK_ROOT", os.path.join(ROOT, "third_party",
                                                                                       "ThunderKittens")))
        check_tk(tk_root)
        if "9.0a" not in os.environ.get("TORCH_CUDA_ARCH_LIST", ""):
            raise RuntimeError("ECHOSPARSE_VSA_TK=1 needs TORCH_CUDA_ARCH_LIST to contain 9.0a (wgmma/TMA)")
        sources.append(f"{src}/sm90_tk/fine_tk_h100.cu")
        include_dirs += [os.path.join(tk_root, "include"), os.path.join(tk_root, "prototype")]
        cxx.append("-DVSA2_WITH_TK")
        nvcc += ["-DVSA2_WITH_TK", "-DKITTENS_HOPPER", "-DNDEBUG"]
        library_dirs, libraries = libcuda_link_flags(os.environ.get("CUDA_HOME") or CUDA_HOME or "")

    class BuildExt(BuildExtension):
        def run(self):
            super().run()
            if not with_tk or not shutil.which("readelf"):
                return
            for ext in self.extensions:  # the driver must be found by SONAME, never through the link-time stub
                path = self.get_ext_fullpath(ext.name)
                dyn = subprocess.run(["readelf", "-d", path], capture_output=True, text=True, check=True).stdout
                if "[libcuda.so.1]" not in dyn or "stubs" in dyn:
                    raise RuntimeError(f"{path} needs DT_NEEDED libcuda.so.1 and no stub run path:\n{dyn}")

    ext = CUDAExtension(name="echosparse.vsa._C", sources=sources, include_dirs=include_dirs,
                        library_dirs=library_dirs, libraries=libraries,
                        extra_compile_args={"cxx": cxx, "nvcc": nvcc})
    return [ext], {"build_ext": BuildExt}


if os.environ.get("ECHOSPARSE_VSA_BUILD", "0") == "1":
    ext_modules, cmdclass = vsa_extension()
    setup(ext_modules=ext_modules, cmdclass=cmdclass)
else:
    setup()
