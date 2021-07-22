# Inductive Reasoning with Text (IRT)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Inductive Reasoning with Text (IRT)](#inductive-reasoning-with-text-irt)
    - [Download](#download)
    - [Installation](#installation)
    - [Loading](#loading)
    - [Data Formats](#data-formats)
        - [Graph Data](#graph-data)
        - [Text Data](#text-data)
    - [PyKeen Dataset](#pykeen-dataset)
    - [Pytorch Dataset](#pytorch-dataset)
    - [Bring Your Own Data](#bring-your-own-data)
    - [Citation](#citation)

<!-- markdown-toc end -->


This code is used to create benchmark datasets as described in
[Open-World Knowledge Graph Completion Benchmarks for Knowledge
Discovery](#) from a given knowledge graph (i.e. triple set) and
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

| Name        | Description       | Download |
|:------------|:------------------|:---------|
| **IRT-CDE** | Based on CoDEx    |          |
| **IRT-FB**  | Based on FB15k237 |          |

**Legacy Data**

This data is used as upstream source or was used in the original
experiments for [the paper](#). They are left here for documentation
and to allow for reproduction of the original results. You need to go
back to this
[commit](https://github.com/lavis-nlp/irtm/tree/157df680f9ee604b43a13581ab7de45d40ac81d6)
in irtm to use the data for model training.


| Name                   | Description                                                                                        | Download                                                                |
|:-----------------------|:---------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------|
| **fb-contexts-v7**     | Original dataset with our text sampling used in the paper (all modes, all context sizes)           | [Link](http://lavis.cs.hs-rm.de/storage/irt/fb.legacy.contexts-v7.tgz)  |
| **fb-owe**             | Original dataset with Wikidata descriptions provided by [shah/OWE](https://github.com/haseebs/OWE) | [Link](http://lavis.cs.hs-rm.de/storage/irt/fb.legacy.owe.tgz)          |
| **fb-db-contexts-v7**  | FB15k aligned text sampled by [ecc](https://github.com/TobiasUhmann/entity-context-crawler)        | [Link](http://lavis.cs.hs-rm.de/storage/irt/fb.src.contexts-db.tgz)     |
| **cde-contexts-v7**    | Original dataset with our text sampling used in the paper (all modes, all contexts sizes)          | [Link](http://lavis.cs.hs-rm.de/storage/irt/cde.legacy.contexts-v7.tgz) |
| **cde-codex.en**       | Dataset with texts provided by [tsafavi/codex](https://github.com/tsafavi/codex)                   | [Link](http://lavis.cs.hs-rm.de/storage/irt/cde.legacy.codex-en.tgz)    |
| **cde-db-contexts-v7** | CoDEx aligned text sampled by [ecc](https://github.com/TobiasUhmann/entity-context-crawler)        | [Link](http://lavis.cs.hs-rm.de/storage/irt/cde.src.contexts-db.tgz)    |


## Installation

Python 3.9 is required. We recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html) for managing Python environments.


``` bash
conda create -n irt python=3.9
pip install -r requirements.txt
```


## Loading

Simply provide a path to an IRT dataset folder.

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


## PyKeen Dataset

For users of [pykeen](https://github.com/pykeen/pykeen).

``` python
TODO
```


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


## Citation

If this is useful to you, please consider a citation:


```
coming soon
```
