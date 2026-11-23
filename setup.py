from setuptools import setup, find_packages

setup(
    name="ekva",
    version="0.2.0",
    packages=find_packages(exclude=("tests", "experiments", "output")),
)
