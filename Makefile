
.PHONY: build clean run install

all: build

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CFLAGS='-w'
else
    CFLAGS='-w -DCYTHON_WITHOUT_ASSERTIONS'
endif

PY ?= python3
PYPY ?= pypy3
ROOT_DIR := $(shell git rev-parse --show-toplevel)
GITHUB_REF ?= "refs/tags/v0.0.0"
GRIDRL_VERSION ?= $(shell echo ${GITHUB_REF} | cut -d'/' -f3)
PYTEST_ARGS ?= -n auto -v

dist: clean build
	${PY} setup.py sdist bdist_wheel
	${PY} -m twine upload dist/gridrl-${version}*

build:
	@echo "Building..."
    rm -rf build
    find gridrl/ -type f -name "*.c" -delete
	CFLAGS=$(CFLAGS) ${PY} setup.py build_ext -j $(shell getconf _NPROCESSORS_ONLN) --inplace

clean:
	@echo "Cleaning..."
	rm -rf gridrl.egg-info
	rm -rf build
	rm -rf dist
	find gridrl/ -type f -name "*.pyo" -delete
	find gridrl/ -type f -name "*.pyc" -delete
	find gridrl/ -type f -name "*.pyd" -delete
	find gridrl/ -type f -name "*.so" -delete
	find gridrl/ -type f -name "*.c" -delete
	find gridrl/ -type f -name "*.h" -delete
	find gridrl/ -type f -name "*.dll" -delete
	find gridrl/ -type f -name "*.lib" -delete
	find gridrl/ -type f -name "*.exp" -delete
	find gridrl/ -type f -name "*.html" -delete
	find gridrl/ -type d -name "__pycache__" -delete

install: build
	${PY} -m pip install .

uninstall:
	${PY} -m pip uninstall gridrl
