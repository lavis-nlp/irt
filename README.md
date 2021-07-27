# Inductive Reasoning with Text (IRT)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/irt-data.svg)](https://badge.fury.io/py/irt-data)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Inductive Reasoning with Text (IRT)](#inductive-reasoning-with-text-irt)
    - [Download](#download)
    - [Installation](#installation)
    - [Loading](#loading)
    - [Data Formats](#data-formats)
        - [Graph Data](#graph-data)
        - [Text Data](#text-data)
    - [PyKEEN Dataset](#pykeen-dataset)
    - [Pytorch Dataset](#pytorch-dataset)
    - [Bring Your Own Data](#bring-your-own-data)
    - [Legacy Data](#legacy-data)
    - [Citation](#citation)

<!-- markdown-toc end -->


This code is used to create benchmark datasets as described in
[Open-World Knowledge Graph Completion Benchmarks for Knowledge
Discovery](https://doi.org/10.1007/978-3-030-79463-7_21) from a given knowledge graph (i.e. triple set) and
supplementary text. The two KG's evaluated in the paper (based on
[FB15k237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
and [CoDEx](https://github.com/tsafavi/codex)) are available for
download [below](#Download).


## Download

We offer two IRT reference datasets: The first - **IRT-FB** - is based
on
[FB15k237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
and the second - **IRT-CDE** - utilizes
[CoDEx](https://github.com/tsafavi/codex). Each dataset offers
knowledge graph triples for the *closed world (cw)* and *open world
(ow)* split. The ow-split is partitioned into validation and test
data. Each entity of the KG is assigned a set of text contexts of
mentions of that entity.

| Name        | Description       | Download                               |
|:------------|:------------------|:---------------------------------------|
| **IRT-CDE** | Based on CoDEx    | [Link](https://bit.ly/2TE62wf-IRT-CDE) |
| **IRT-FB**  | Based on FB15k237 | [Link](https://bit.ly/376WJIh-IRT-FB)  |


## Installation

Python 3.9 is required. We recommend
[miniconda](https://docs.conda.io/en/latest/miniconda.html) for
managing Python environments.


``` bash
conda create -n irt python=3.9
conda activate irt
pip install irt-data
```

The `requirements.txt` contains additional packages used for development.


## Loading

Simply provide a path to an IRT dataset folder. The data is loaded
lazily - that is why the construction is fast, but the first invocation
of `.description` takes a while.


``` python
from irt import Dataset
dataset = Dataset('path/to/irt-fb')
print(dataset.description)
```

```
IRT DATASET

IRT GRAPH: irt-fb
  nodes: 14541
  edges: 310116 (237 types)
  degree:
    mean 42.65
    median 26

IRT SPLIT
2389 retained concepts

Config:
  seed: 26041992
  ow split: 0.7
  ow train split: 0.5
  relation threshold: 100
  git: 66fe7bd3c934311bdc3b1aa380b7c6c45fd7cd93
  date: 2021-07-21 17:29:04.339909

Closed World - TRAIN:
  owe: 12164
  entities: 12164
  heads: 11562
  tails: 11252
  triples: 238190

Open World - VALID:
  owe: 1558
  entities: 9030
  heads: 6907
  tails: 6987
  triples: 46503

Open World - TEST:
  owe: 819
  entities: 6904
  heads: 4904
  tails: 5127
  triples: 25423

IRT Text (Mode.CLEAN)
  mean contexts: 28.92
  median contexts: 30.00
  mean mentions: 2.84
  median mentions: 2.00
```


## Data Formats

The data in the respective provided dataset folders should be quite
self-explanatory. Each entity and each relation is assigned a unique
integer id (denoted `e` [entity], `h` [head], `t` [tail], and `r`
[relation]). There is folder containing the full graph data
(`graph/`), a folder containing the open-world/closed-world splits
(`split/`) and the textual data (`text/`).

### Graph Data

This concerns both data in `graph/` and `split/`. Entity and relation
identifier can be translated with the `graph/entities.txt` and
`graph/relations.txt` respectively. Triple sets come in `h t r`
order. Reference code to load graph data:

* `irt.graph.Graph.load`
* `irt.data.dataset.Split.load`


### Text Data

The upstream system that sampled our texts:
[ecc](https://github.com/TobiasUhmann/entity-context-crawler). All
text comes gzipped and can be opened using the built-in python `gzip`
library. For inspection, you can use the `zcat`, `zless`, `zgrep`,
etc. (at least on unixoid systems ;)) - or extract them using
`unzip`. Reference code to load text data:

* `irt.data.dataset.Text.load`


## PyKEEN Dataset

For users of [pykeen](https://github.com/pykeen/pykeen). There are two
"views" on the triple sets: closed-world and open-world. Both simply
offer pykeen TriplesFactories with an id-mapping to the IRT
entity-ids.

**Closed-World:**


``` python
from irt import Dataset
from irt import KeenClosedWorld

dataset = Dataset('path/to/dataset')

# 'split' is either a single float, a tuple (for an additional
# test split) or a triple which must sum to 1
kcw = KeenClosedWorld(dataset=dataset, split=.8, seed=1234)

print(kcw.description)
```

```
IRT PYKEEN DATASET
irt-cde

  training triples factory:
    entities: 12091
    relations: 51
    triples: 109910

  validation triples factory:
    entities: 12091
    relations: 51
    triples: 27478
```

It offers `.training`, `.validation`, and `.testing` TriplesFactories,
and `irt2keen`/`keen2irt` id-mappings.


**Open-World:**

``` python
from irt import Dataset
from irt import KeenClosedWorld

dataset = Dataset('path/to/dataset')
kow = KeenOpenWorld(dataset=ds)

print(kow.description)
```

```
IRT PYKEEN DATASET
irt-cde

  closed world triples factory:
    entities: 12091
    relations: 51
    triples: 137388

  open world validation triples factory:
    entities: 15101
    relations: 46
    triples: 41240

  open world testing triples factory:
    entities: 17050
    relations: 48
    triples: 27577
```

It offers `.closed_world`, `.open_world_valid`, and `.open_world_test`
TriplesFactories, and `irt2keen`/`keen2irt` id-mappings.


## Pytorch Dataset

For users of [pytorch](https://pytorch.org/)
and/or [pytorch-lightning](https://www.pytorchlightning.ai/).

We offer a `torch.utils.data.Dataset`, a `torch.utils.data.DataLoader`
and a `pytorch_lightning.DataModule`. The dataset abstracts what a
"sample" is and how to collate samples to batches:


``` python
from irt import TorchDataset

# given you have loaded a irt.Dataset instance called "dataset"
# 'model_name' is one of huggingface.co/models
torch_dataset = TorchDataset(
    model_name='bert-base-cased',
    dataset=dataset,
    part=dataset.split.closed_world,
)

# a sample is an entity-to-token-index mapping:
torch_dataset[100]
# -> Tuple[int, List[int]]
# (124, [[101, 1130, ...], ...])

# and it offers a collator for batching:
batch = TorchDataset.collate_fn([torch_dataset[0], torch_dataset[1]])
# batch: Tuple[Tuple[int], torch.Tensor]

len(batch[0])   # -> 60
batch[1].shape  # -> 60, 105
```

Note: Only the first invocation is slow, because the tokenizer needs
to run. The tokenized text is saved to the IRT folder under `torch/`
and re-used from then on.


## Bring Your Own Data

If you want to utilize this code to create your own
open-world/closed-world-split, you need to either bring your data in a
format readable by the existing code base or extend this code for your
own data model. See [ipynb/graph.split.ipynb](ipynb/graph.split.ipynb)
for a step-by-step guide.


## Legacy Data

This data is used as upstream source or was used in the original
experiments for [the paper](#). They are left here for documentation
and to allow for reproduction of the original results. You need to go
back to this
[commit](https://github.com/lavis-nlp/irtm/tree/157df680f9ee604b43a13581ab7de45d40ac81d6)
in irtm to use the data for model training.


| Name                   | Description                                                                                     | Download                                                                |
|:-----------------------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------|
| **fb-contexts-v7**     | Original dataset (our text) as used in the paper (all modes, all context sizes)                 | [Link](http://lavis.cs.hs-rm.de/storage/irt/fb.legacy.contexts-v7.tgz)  |
| **fb-owe**             | Original dataset (Wikidata descriptions provided by [shah/OWE](https://github.com/haseebs/OWE)) | [Link](http://lavis.cs.hs-rm.de/storage/irt/fb.legacy.owe.tgz)          |
| **fb-db-contexts-v7**  | Our text sampled by [ecc](https://github.com/TobiasUhmann/entity-context-crawler) for FB        | [Link](http://lavis.cs.hs-rm.de/storage/irt/fb.src.contexts-db.tgz)     |
| **cde-contexts-v7**    | Original dataset (our text) as used in the paper (all modes, all contexts sizes)                | [Link](http://lavis.cs.hs-rm.de/storage/irt/cde.legacy.contexts-v7.tgz) |
| **cde-codex.en**       | Original dataset (Texts provided by [tsafavi/codex](https://github.com/tsafavi/codex))          | [Link](http://lavis.cs.hs-rm.de/storage/irt/cde.legacy.codex-en.tgz)    |
| **cde-db-contexts-v7** | Our text sampled by [ecc](https://github.com/TobiasUhmann/entity-context-crawler) for CDE       | [Link](http://lavis.cs.hs-rm.de/storage/irt/cde.src.contexts-db.tgz)    |


## Citation

If this is useful to you, please consider a citation:

```
@inproceedings{hamann2021open,
  title={Open-World Knowledge Graph Completion Benchmarks for Knowledge Discovery},
  author={Hamann, Felix and Ulges, Adrian and Krechel, Dirk and Bergmann, Ralph},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  pages={252--264},
  year={2021},
  organization={Springer}
}
```
