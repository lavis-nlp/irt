# Inductive Reasoning with Text (IRT)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This code is used to create benchmark datasets as described in
[Open-World Knowledge Graph Completion Benchmarks for Knowledge
Discovery](#) from a given knowledge graph (i.e. triple set) and
supplementary text. The two KG's evaluated in the paper (based on
[FB15k237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
and [CoDEx](https://github.com/tsafavi/codex)) are available for
download [below](#Download).


## Installation

Python 3.9 is required. We recommend [conda](https://docs.conda.io/en/latest/miniconda.html) for managing Python environments.


``` bash
conda create -n irt python=3.9
pip install -r requirements.txt
```


## Data Organisation

We offer two IRT reference datasets: The first - **IRT-FB** - is
baed on
[FB15k237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
and the second - **IRT-CDE** - utilizes
[CoDEx](https://github.com/tsafavi/codex). Each dataset offers
knowledge graph triples for the *closed world (cw)* and *open world
(ow)* split. The ow-split is partitioned into validation and test
data. Each entity of the kg is assigned a set of textual description
of mentions. We offer sets of size 1, 5, 15, and 30.

TODO describe `.load`-Interface for the IRT dataset.


### Graph Data

TODO describe file formats


### Text Data

TODO describe file formats


## Download

TBA
