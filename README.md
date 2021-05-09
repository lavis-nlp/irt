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
of mentions.


## Download

TBA


## Loading

TODO describe `.load`-Interface for the IRT dataset.


## Data Formats

TODO

### Graph Data

TODO describe file formats


### Split Data

TODO describe file formats


### Text Data

TODO describe file formats


## PyKeen Dataset

TODO


## Pytorch Dataset

We offer a `torch.utils.data.Dataset`, a `torch.utils.data.DataLoader`
and a `pytorch_lightning.DataModule`. The Dataset abstracts what a
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





## Bring Your Own Data

If you want to utilize this code to create your own *ow/cw*-split, you
need to either bring your data in a format readable by the existing
code base or extend this code for your own data model. See
[ipynb/graph.split.ipynb](#) for a step-by-step guide.
