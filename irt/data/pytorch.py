# -*- coding: utf-8 -*-

"""

IRT dataset wrapper for use by pytorch and pytorch lightning

https://github.com/pytorch/pytorch
https://github.com/PyTorchLightning/pytorch-lightning

"""

import irt
from irt.data import dataset
from irt.common import helper

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm as _tqdm

import numpy as np
import transformers as tf
import torch.utils.data as td
import pytorch_lightning as pl

import enum
import logging
import pathlib
from functools import partial
from functools import lru_cache
from collections import defaultdict

from typing import Union
from typing import Optional


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


def load_tokenizer(
    model_name: str,
    dataset_path: Union[str, pathlib.Path],
) -> tf.BertTokenizer:

    dataset_path = helper.path(dataset_path, exists=True)
    path = dataset_path / "torch" / model_name / "tokenizer"

    if path.is_dir():
        log.info("loading tokenizer from disk")
        tokenizer = tf.BertTokenizer.from_pretrained(str(path))

    else:
        log.info("creating new tokenizer")
        cache_dir = str(irt.ENV.CACHE_DIR / "lib.transformers")
        tokenizer = tf.BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            additional_special_tokens=[
                irt.text.TOK_MENTION_START,
                irt.text.TOK_MENTION_END,
            ],
        )

        log.info("saving tokenizer to disk")
        tokenizer.save_pretrained(str(path))

    return tokenizer


class TorchDataset(td.Dataset):
    """

    Offers samples and a batch collator

    This dataset holds a tokenizer, and maps all tokens to indexes.
    The collate_fn creates batches based on the longest sequence of
    the batch.

    """

    model_name: str
    dataset: dataset.Dataset
    tokenizer: tf.BertTokenizer

    max_context_idx: int
    max_context_size: int

    degrees: torch.Tensor  # Index -> Degree

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx: int) -> tuple[int, list[int]]:
        return self._flat[idx]

    def _tokenize(self, part):
        fname = f"{self.dataset.text.mode.value}.{part.name}.txt"
        path = self.dataset.path / "torch" / self.model_name / fname
        path.parent.mkdir(exist_ok=True)

        if not path.is_file():
            self._flat = []

            log.info("begin tokenization")
            for e in tqdm(part.owe):
                sentences = [sample.context for sample in self.dataset.text[e]]
                if sentences:
                    tokenized = self.tokenizer(sentences)["input_ids"]
                    self._flat.append((e, tokenized))

            log.info("saving indexes to file")
            with path.open(mode="w") as fd:
                for e, tokens in self._flat:
                    prefix = f"{e}: "
                    idxstr = [" ".join(map(str, idxs)) for idxs in tokens]
                    tokstr = prefix + f"\n{prefix}".join(idxstr) + "\n"
                    fd.write(tokstr)

            log.info("finished tokenization)")

        else:
            log.info("loading indexes from file")
            with path.open(mode="r") as fd:

                accum = defaultdict(list)
                for line in fd:
                    estr, idxstr = map(str.strip, line.split(":"))
                    accum[int(estr)].append(list(map(int, idxstr.split())))

            self._flat = tuple(accum.items())

    def __init__(
        self,
        model_name: str,
        part: dataset.Part,
        dataset: dataset.Dataset,
    ):
        super().__init__()
        self.name = part.name

        self.model_name = model_name
        self.dataset = dataset

        # tokenize
        self.tokenizer = load_tokenizer(
            model_name=model_name,
            dataset_path=self.dataset.path,
        )

        assert self.tokenizer
        self._tokenize(part)
        assert self._flat

        # analyze
        shapes = [
            [len(idxs), max(len(sentence) for sentence in idxs)]
            for _, idxs in self._flat
        ]

        self.max_context_idx = np.argmax([x * y for x, y in shapes])
        self.max_context_size = shapes[self.max_context_idx]

        # note: this does not consider direction; nodes with an
        #  out_degree of 0 can still have a large in_degree
        self.degrees = torch.Tensor([part.graph.nx.degree[e] for e, _ in self._flat])
        assert len(self.degrees[self.degrees == 0]) == 0, "found disconnected nodes"

        log.info(
            f"node degrees: mean={self.degrees.mean():2.2f};"
            f" std={self.degrees.std():2.2f}"
        )

        log.info(
            f"initialized torch dataset {self.name}: samples={len(self)};"
            f" max context size: {self.max_context_size}"
        )

    @staticmethod
    def collate_fn(batch: list[tuple[int, torch.Tensor]]) -> tuple[torch.Tensor]:

        # flatten entities to match context counts
        ents = tuple(ent for ent, ctx in batch for _ in ctx)

        # flatten and pad context sentences
        ctxs = pad_sequence(
            [
                torch.Tensor(sentence).to(torch.long)
                for _, ctx in batch
                for sentence in ctx
            ],
            batch_first=True,
        )

        return (ents, ctxs)


class TorchLoader(td.DataLoader):

    subbatch_size: int

    def __init__(self, torch_dataset, subbatch_size: int, *args, **kwargs):
        super().__init__(
            torch_dataset,
            *args,
            collate_fn=torch_dataset.collate_fn,
            **kwargs,
        )

        self.subbatch_size = subbatch_size


class Sampler(enum.Enum):

    node_degree = "node degree"


class TorchModule(pl.LightningDataModule):

    # pykeen open-world dataset
    kow: irt.KeenOpenWorld

    train_set: TorchDataset
    valid_set: TorchDataset
    test_set: TorchDataset

    # ---

    @lru_cache
    def subbatch_size(self, kind: str = None) -> int:
        log.info(f"subbatch_size cache miss: {kind=}")

        if kind == "train":
            return self.train_dataloader().subbatch_size

        assert kind == "valid"
        return self.val_dataloader().subbatch_size

    @property
    def kgc_dataloader(self):
        return self.val_dataloader()

    # ---

    def __init__(
        self,
        model_name: str,
        kow: irt.KeenOpenWorld,
        dataloader_train_args: dict,
        dataloader_valid_args: dict,
        dataloader_test_args: dict,
        sampler: Optional[str] = None,
        sampler_args: Optional[dict] = None,
    ):
        """

        Create a new DataModule for an IRT dataset

        Parameters
        ----------

        model_name : str
          One of the huggingface transformer models

        kow : irt.KeenOpenWorld
          IRT encapsulated for open-world KGC

        sampler: Optional[str]
          Currently, only "node degree" is supported which selects
          samples more frequently when they are highly connected
          in the graph

        sampler_args: Optional[dict]
          For "node degree":
             num_samples: int --  Total count of samples
             replacement: bool -- Whether samples can be drawn multiple times

        Returns
        -------


        """
        super().__init__()
        self.model_name = model_name
        self.kow = kow

        self._dataloader_train_args = dataloader_train_args
        self._dataloader_valid_args = dataloader_valid_args
        self._dataloader_test_args = dataloader_test_args

        self._sampler_name = sampler
        self._sampler_args = sampler_args

    def prepare_data(self, *args, **kwargs):
        # called once on master for multi-gpu setups
        # do not set any state here
        pass

    def setup(self, *args, **kwargs):

        self.train_set = TorchDataset(
            model_name=self.model_name,
            dataset=self.kow.dataset,
            part=self.kow.dataset.split.closed_world,
        )

        self._sampler = None
        if self._sampler_name:
            self._sampler_name = Sampler(self._sampler_name)

            num_samples = self._sampler_args["num_samples"]
            if num_samples.startswith("x"):
                num_samples = len(self.train_set) * int(num_samples[1:])
            elif num_samples == "triples":
                num_samples = int(sum(self.train_set.degrees) // 2)

            replacement = self._sampler_args["replacement"]

            self._sampler = td.WeightedRandomSampler(
                weights=1 / self.train_set.degrees,
                num_samples=num_samples,
                replacement=replacement,
            )

            log.info(f"using node degreee sampler {num_samples=} {replacement=}")

        self.valid_set = TorchDataset(
            model_name=self.model_name,
            dataset=self.kow.dataset,
            part=self.kow.dataset.split.open_world_valid,
        )

        self.test_set = TorchDataset(
            model_name=self.model_name,
            dataset=self.kow.dataset,
            part=self.kow.dataset.split.open_world_test,
        )

    # FOR LIGHTNING

    def train_dataloader(self) -> TorchLoader:
        return TorchLoader(
            self.train_set,
            sampler=self._sampler,
            **self._dataloader_train_args,
        )

    def val_dataloader(self) -> TorchLoader:
        return TorchLoader(
            self.valid_set,
            **self._dataloader_valid_args,
        )

    def test_dataloader(self) -> TorchLoader:
        # see evaluator.py
        return TorchLoader(
            self.test_set,
            **self._dataloader_test_args,
        )
