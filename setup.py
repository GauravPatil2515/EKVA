from setuptools import setup, find_packages

setup(
    name="ekva",
    version="0.2.0",
    description="Expert-Aware KV Budget Allocation for Sparse MoE LLM inference",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Gaurav Patil",
    license="MIT",
    packages=find_packages(exclude=("tests", "experiments", "output")),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "models": ["transformers>=4.40.0", "datasets>=2.18.0", "accelerate>=0.29.0"],
        "kernel": ["triton>=2.3.0"],
        "dev": ["pytest>=8.0.0"],
    },
)
