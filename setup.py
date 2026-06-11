from setuptools import setup, find_packages

setup(
    name="ekva",
    version="0.1.0",
    description="Expert-Aware KV Budget Allocation for Sparse MoE LLMs",
    author="Gaurav Patil",
    packages=find_packages(),
    python_requires=">=3.10",
)
