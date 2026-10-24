from setuptools import setup, find_packages

setup(
    name="swe-agent-moe",
    version="0.1.0",
    description="SWE-Agent MoE Transformer Training Pipeline",
    packages=find_packages(),
    install_requires=[
        "torch>=2.14.1",
        "torchvision>=0.29.1",
        "transformers>=5.10.0",
        "accelerate>=1.15.0",
        "datasets>=2.20.0",
        "wandb>=0.17.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "pillow>=13.0.0",
        "aiohttp>=3.14.4",
        "anyio>=4.14.2",
        "certifi>=2026.10.19",
        "idna>=3.15",
        "httpx2>=2.12.0",
        "httpcore2>=2.10.0",
    ],
    extras_require={
        "flash": ["flash-attn>=2.6.0"],
        "tpu": ["torch-xla[tpu]"],
    },
    python_requires=">=3.11",
)
