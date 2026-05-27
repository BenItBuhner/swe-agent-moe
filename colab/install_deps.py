"""Colab dependency installation script.

Run this first in the Colab notebook to set up the environment.
"""

import sys
import subprocess
import importlib
from pathlib import Path


REQUIRED_PACKAGES = [
    "torch>=2.4.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "datasets>=2.20.0",
    "wandb>=0.17.0",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25.0",
    "einops>=0.7.0",
]


def check_tpu_available() -> bool:
    try:
        import torch_xla
        return True
    except ImportError:
        return False


def install_tpu_deps():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch-xla[tpu]", "torch", "torchvision",
        "-f", "https://storage.googleapis.com/libtpu-releases/index.html",
    ])


def install_flash_attn():
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "flash-attn", "--no-build-isolation",
        ])
    except subprocess.CalledProcessError:
        print("flash-attn build failed; using sdpa fallback")


def setup_colab():
    print("=" * 60)
    print("SWE-Agent MoE Training - Colab Setup")
    print("=" * 60)

    has_gpu = False
    has_tpu = False

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        has_tpu = check_tpu_available()

        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        if has_tpu:
            print("TPU detected")

        if not has_gpu and not has_tpu:
            print("WARNING: No accelerator found. Training will be slow on CPU.")
    except ImportError:
        print("PyTorch not found, installing...")

    print(f"\nInstalling {len(REQUIRED_PACKAGES)} required packages...")
    for pkg in REQUIRED_PACKAGES:
        pkg_name = pkg.split(">=")[0].split("==")[0]
        try:
            importlib.import_module(pkg_name.replace("-", "_"))
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    if has_tpu:
        print("Installing TPU dependencies...")
        install_tpu_deps()

    if has_gpu:
        install_flash_attn()

    project_root = Path("/content/model-training-pipeline")
    if project_root.exists():
        sys.path.insert(0, str(project_root))
        print(f"\nProject root added to sys.path: {project_root}")

    print("\nSetup complete!")
    print(f"  GPU available: {has_gpu}")
    print(f"  TPU available: {has_tpu}")
    print(f"  Python: {sys.version}")

    return has_gpu, has_tpu


if __name__ == "__main__":
    setup_colab()
