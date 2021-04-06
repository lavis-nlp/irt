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


## Bring Your Own Data

If you want to utilize this code to create your own *ow/cw*-split, you
need to either bring your data in a format readable by the existing
code base or extend this code for your own data model. This section
describes how **IRT-CDE** is created using the code base for guidance.


### The CW/OW Triple Split

The algorithm to determine *concept entities* and the subsequent
selection of *open-world* entities is described in Section 3 of the
paper. An implementation of that algorithm can be found in
`irt/graph/split.py:Splitter.create`. The goal of this step is to
create a `split.Dataset`.

First, a knowledge graph needs to be loaded. We use CoDEx and the
loader defined in `irt/graph/loader.py:`

``` bash
mkdir -p lib
git clone https://github.com/tsafavi/codex lib/codex
```

``` python
from irt.graph import loader

source = loader.load_cde(
    'lib/codex/data/triples/codex-m/train.txt',
    'lib/codex/data/triples/codex-m/valid.txt',
    'lib/codex/data/triples/codex-m/test.txt',
    f_ent2id='lib/codex/data/entities/en/entities.json',
    f_rel2id='lib/codex/data/relations/en/relations.json',
)

from irt.graph import graph
g = graph.Graph(name='irt-cde', source=source)

print(str(g))
# irt graph: [irt-cde] (17050 entities)

print(g.description)
# irt graph: irt-cde
#   nodes: 17050
#   edges: 206205 (51 types)
#   degree:
#     mean 24.19
#     median 13

# now is a good time to save the graph to disk
# we have canonical places for data but you can use any place you like

import irt
g.save(irt.ENV.DATASET_DIR / 'cde')

```

If you want to roll your own loader, you can simply implement one that
returns a `irt.graph.Graph` instance. If you want to have it
integrated into `irt`, implement it in `irt.graph.loader` and register
the name in `irt.graph.loader.LOADER`.

