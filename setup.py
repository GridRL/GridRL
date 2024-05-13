#!/usr/bin/env python3

"""Package setup."""

import sys
import platform
import os
import multiprocessing
from multiprocessing import cpu_count
import numpy as np
from setuptools import Extension, setup, find_packages

try:
    raise ModuleNotFoundError # CURRENTLY NOTHING TO CYTHONIZE, RAISING EXCEPTION
    if platform.python_implementation() != "CPython":
        raise ModuleNotFoundError
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext as _build_ext

    use_cython = True
except (ModuleNotFoundError, ImportError):
    use_cython = False

try:
    script_dir = f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}"
except NameError:
    script_dir = f".{os.sep}"

required_packages = [
    "numpy",
    "pillow",
    "gymnasium>=0.29",
    "matplotlib",
    "pandas",
]
if use_cython:
    required_packages.append("cython>=3.0.6")

version = "0.0.1"
with open(f"{script_dir}gridrl{os.sep}__version__.py", mode="r", encoding="utf8") as f:
    version = f.read().rsplit(maxsplit=1)[-1].strip("'")

if use_cython:

    def prep_pxd_py_files():
        ignore_py_files = []
        for root, dirs, files in os.walk("."):
            if len([1 for m in ["data", "extras", "examples", ".egg-info"] if m in root]) > 0:
                continue
            for f in files:
                if len([1 for m in ["setup.py", "core_game.p"] in f]) > 0:
                    continue
                if os.path.splitext(f)[1] in [".py", ".pyx"] and f not in ignore_py_files:
                    yield os.path.join(root, f)
                """if os.path.splitext(f)[1] == ".pxd":
                    py_file = f"{os.path.join(root, os.path.splitext(f)[0])}.py"
                    if os.path.isfile(py_file):
                        if os.path.getmtime(os.path.join(root, f)) > os.path.getmtime(py_file):
                            os.utime(py_file)"""

    class build_ext(_build_ext):
        def initialize_options(self):
            super().initialize_options()
            self.parallel = cpu_count()
            # Fixing issue with nthreads in Cython
            if (
                (3, 8) <= sys.version_info[:2]
                and sys.platform == "darwin"
                and multiprocessing.get_start_method() == "spawn"
            ):
                multiprocessing.set_start_method("fork", force=True)
            cflags = []  # ["-O3"]
            # NOTE: For performance. Check if other platforms need an equivalent change.
            if sys.platform == "darwin":
                cflags.append("-DCYTHON_INLINE=inline __attribute__ ((__unused__)) __attribute__((always_inline))")
            py_pxd_files = prep_pxd_py_files()
            cythonize_files = map(
                lambda src: Extension(
                    src.split(".")[0].replace(os.sep, "."),
                    [src],
                    extra_compile_args=cflags,
                    extra_link_args=["-s", "-w"],
                    include_dirs=[np.get_include()],
                ),
                list(py_pxd_files),
            )
            self.distribution.ext_modules = cythonize(
                [*cythonize_files],
                nthreads=0 if sys.platform == "win32" else cpu_count(),
                annotate=False,
                gdb_debug=False,
                language_level=3,
                compiler_directives={
                    ### NOT IS FOR TESTING
                    "boundscheck": not False,
                    "cdivision": True,
                    "cdivision_warnings": False,
                    "infer_types": not True,
                    "initializedcheck": not False,
                    "legacy_implicit_noexcept": not True,
                    "nonecheck": not False,
                    "overflowcheck": not False,
                    "profile" : False,
                    "wraparound": not False,
                },
            )

    cythonized_kwargs = {
        "cmdclass": {"build_ext": build_ext},
        "ext_modules": [Extension("", [""])],
    }
else:
    cythonized_kwargs = {}

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
        "Programming Language :: Python :: 3.11",
    ],
    **cythonized_kwargs,
)
