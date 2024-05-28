# Copyright (c) 2024, Tri Dao.

import sys
import os
import re
import ast
from pathlib import Path
import platform

from setuptools import setup, find_packages

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "causal_conv1d"

BASE_WHEEL_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("CAUSAL_CONV1D_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("CAUSAL_CONV1D_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("CAUSAL_CONV1D_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    print("Please note that rocm >= 6.0 is required for it to build correctly and run efficiently.")
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    ext_modules.append(
        CUDAExtension(
            name="causal_conv1d_cuda",
            sources=[
                "csrc/causal_conv1d.cpp",
                "csrc/causal_conv1d_fwd.cu",
                "csrc/causal_conv1d_bwd.cu",
                "csrc/causal_conv1d_update.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc":
                    [
                        "-O3",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "-ffast-math",
                        "-munsafe-fp-atomics"
                    ],
            },
            include_dirs=[Path(this_dir) / "csrc" / "causal_conv1d"],
            extra_link_args=["-z", "muldefs"]
        )
    )


def get_package_version():
    with open(Path(this_dir) / "causal_conv1d" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("CAUSAL_CONV1D_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class CachedWheelsCommand(_bdist_wheel):
    """
    Fragment from the CUDA version. There aren't any prebuilt wheels, 
    so this just calls run.
    """
    def run(self):
        super().run()


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "causal_conv1d.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Causal depthwise conv1d in CUDA, with a PyTorch interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/causal-conv1d",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
    ],
)
