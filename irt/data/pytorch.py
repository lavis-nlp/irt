# -*- coding: utf-8 -*-

"""

IRT dataset wrapper for use by pytorch and pytorch lightning

https://github.com/pytorch/pytorch
https://github.com/PyTorchLightning/pytorch-lightning

"""

import irt
from irt.data import dataset

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm as _tqdm

import numpy as np
import transformers as tf
import torch.utils.data as td
import pytorch_lightning as pl

import logging
from functools import partial
from collections import defaultdict

from typing import List
from typing import Tuple


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


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

    def __getitem__(self, idx: int) -> Tuple[int, List[int]]:
        return self._flat[idx]

    def _get_tokenizer(self):
        path = self.dataset.path / "torch" / self.model_name / "tokenizer"

        if path.is_dir():
            log.info("loading tokenizer from disk")
            self.tokenizer = tf.BertTokenizer.from_pretrained(str(path))

        else:
            log.info("creating new tokenizer")
            cache_dir = str(irt.ENV.CACHE_DIR / "lib.transformers")
            self.tokenizer = tf.BertTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                additional_special_tokens=[
                    irt.text.TOK_MENTION_START,
                    irt.text.TOK_MENTION_END,
                ],
            )

            log.info("saving tokenizer to disk")
            self.tokenizer.save_pretrained(str(path))

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

        else:

            log.info("loading indexes from file")
            with path.open(mode="r") as fd:

                accum = defaultdict(list)
                for line in fd:
                    estr, idxstr = map(str.strip, line.split(":"))
                    accum[int(estr)].append(list(map(int, idxstr.split())))

            self._flat = tuple(accum.items())

        log.info("finished tokenization")

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
        self._get_tokenizer()
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
    def collate_fn(batch: List[Tuple[int, torch.Tensor]]) -> Tuple[torch.Tensor]:

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


class TorchDataLoader(td.DataLoader):
    pass


class TorchDataModule(pl.LightningDataModule):
    pass
