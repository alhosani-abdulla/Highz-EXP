from setuptools import setup, find_packages
from pathlib import Path

with open(Path(__file__).parent / "requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="highz_exp",
    version="0.1.0",
    description="A Python package for Highz-EXP project.",
    author="High-Z Team",
    author_email="highz.team@example.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)