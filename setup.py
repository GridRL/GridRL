#!/usr/bin/env python3

"""Package setup."""

from setuptools import find_packages,setup

required_packages=[
    "numpy",
    "pillow",
    "gymnasium>=0.29",
    "matplotlib",
]

with open("gridrl/__version__.py",mode="r",encoding="utf8") as f:
    version=f.read().rsplit(maxsplit=1)[-1].strip("'")

setup(
    name="gridrl",
    description="Customizable engine for minimalist 2D grid-based exploration games,"
        " oriented towards Reinforcement Learning.",
    long_description_content_type="text/markdown",
	version=version,
    packages=find_packages(),
    include_package_data=False,
    install_requires=required_packages,
    python_requires=">=3.9",
    license="MIT",
    author="Sky",
    author_email="gridrlgym@gmail.com",
    url="https://github.com/GridRL/GridRL",
    keywords=["GridWorld", "Exploration", "AI", "RL"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
