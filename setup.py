from setuptools import setup, find_packages

setup(
    name="swe-agent-moe",
    version="0.1.0",
    description="SWE-Agent MoE Transformer Training Pipeline",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "wandb>=0.17.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    ],
    extras_require={
        "flash": ["flash-attn>=2.6.0"],
        "tpu": ["torch-xla[tpu]"],
    },
    python_requires=">=3.10",
)
