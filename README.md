<!-- Do NOT edit the Markdown file directly - generated from the Jupyter notebook. -->

# Tumoroscope in PyMC

[![python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Pytest](https://github.com/jhrcook/tumoroscope-pymc/actions/workflows/pytest.yaml/badge.svg)](https://github.com/jhrcook/tumoroscope-pymc/actions/workflows/pytest.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**This package is a Work-in-Progress.**

This package builds the ['Tumoroscope']() (Shafighi *et al.*, 2022, bioRxiv preprint) model with the probabilistic programming library [PyMC]().
'Tumoroscope' is a "probabilistic model that accurately infers cancer clones and their high-resolution localization by integrating pathological images, whole exome sequencing, and spatial transcriptomics data."

![Tumoroscope diagram](tumoroscope-diagram.jpeg)

## Installation

> As this package provides a model produced using PyMC, I recommend first creating a virtual environment using `conda` and installing the PyMC library.
> You can follow their instructions [here](https://www.pymc.io/projects/docs/en/latest/installation.html).

You can install this package using `pip` either from PyPI

```bash
pip install tumoroscope-pymc  # not available yet
```

or from GitHub

```
pip install git+https://github.com/jhrcook/tumoroscope-pymc.git
```

## Use

(TODO)


```python
import numpy as np
import pymc as pm

from tumoroscope import TumoroscopeData, build_tumoroscope_model
from tumoroscope.mock_data import generate_random_data

np.random.seed(1)
data = generate_random_data()

model = build_tumoroscope_model(data)
pm.model_to_graphviz(model)
```





![svg](README_files/README_3_0.svg)




## Developing

Setup up the develpment envionrment using `conda` (or `mamba`)

```bash
mamba env create -f conda.yaml
conda activate tumoroscope-pymc
```

Run the test suite using `tox`

```bash
tox
```

Build the README documentation by re-executing the `README.ipynb` notebook and converting it to Markdown using the following command

```bash
tox -e readme
```

---

## Environment information



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-10-24

    Python implementation: CPython
    Python version       : 3.10.6
    IPython version      : 8.5.0

    Compiler    : Clang 13.0.1
    OS          : Darwin
    Release     : 21.6.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: JHCookMac.local

    Git branch: dev

    numpy: 1.23.4
    pymc : 4.2.2
