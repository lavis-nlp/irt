# -*- coding: utf-8 -*-

"""

assigns text to entities

"""

import irt
from irt.text import loader
from irt.graph import graph
from irt.common import helper
from irt.common import logging

import enum
import gzip
import random
import pathlib
import contextlib

from functools import partial
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

import yaml
from tqdm import tqdm as _tqdm

from typing import IO
from typing import Union


log = logging.get("text.selector")
tqdm = partial(_tqdm, ncols=80)


SEP = "|"
MASK_TOKEN = "[MASK]"
TOK_MENTION_START = "[MENTION_START]"
TOK_MENTION_END = "[MENTION_END]"


# this matches masked sequences of the upstream sqlite context dbs
MARKED = TOK_MENTION_START + " {mention} " + TOK_MENTION_END


def _filter(sentence, mention):
    return all(
        (
            # at least 50 chars per sentence
            len(sentence) > 50,
            # no File: prefixed blobs
            not sentence.startswith("File:"),
        )
    )


class Mode(enum.Enum):
    CLEAN = "clean"
    MARKED = "marked"
    MASKED = "masked"


# ---


@dataclass
class Selector:

    g: graph.Graph

    sentences: list[int]
    shuffle: bool

    select: loader.Loader

    fd_nocontext: IO[bytes]
    fd_sentences: dict[str, IO[bytes]]

    # --

    def get_text(self):
        for e in set(self.g.nx.nodes):
            yield e, self.select.by_entity(e=e)

    def handle_noctx(self, e: int, noctx_triples: set, noctx_ents: set):
        name = self.g.source.ents[e]
        triples = self.g.find(heads={e}, tails={e})
        count = len(triples)

        noctx_triples |= triples
        noctx_ents.add(e)

        self.write(
            self.fd_nocontext,
            f"{e}{SEP}{name}{SEP}{count}",
        )

        msg = f"! no context for {e}: {name} ({count} triples)"
        log.info(msg)

    def transform_result(self, result=loader.Result):

        # list(set(x)) is not possible because of the
        # unpredictable python hash seeds
        acc, seen = defaultdict(list), set()

        for blob, mention in zip(result.blobs, result.mentions):

            # filter BEFORE mapping
            gen = (s for s in blob.split("\n") if _filter(s, mention))

            for sentence in gen:
                if sentence in seen:
                    continue

                seen.add(sentence)
                acc[mention].append(sentence)

        if self.shuffle:
            for sentences in acc.values():
                random.shuffle(sentences)

        # select n sentences each
        return {
            mention: sentences[: self.sentences]
            for mention, sentences in acc.items()
            if sentences
        }

    def transform_sentences(self, mention: str, sentences: list[str]):

        sentences = [" ".join(sentence.strip().split()) for sentence in sentences]

        ret = {Mode.CLEAN: sentences}

        if Mode.MASKED in self.fd_sentences:
            ret[Mode.MASKED] = [
                sentence.replace(mention, MASK_TOKEN) for sentence in sentences
            ]

        if Mode.MARKED in self.fd_sentences:
            ret[Mode.MARKED] = [
                sentence.replace(mention, MARKED.format(mention=mention))
                for sentence in sentences
            ]

        return ret

    def write(self, fd, s: str):
        assert "\n" not in s, f"{s=}"
        fd.write((s + "\n").encode())

    def write_header(self):
        self.fd_nocontext.write(f"# Format: <ID>{SEP}<NAME>{SEP}<TRIPLES>\n".encode())

        for fd in self.fd_sentences.values():
            fd.write(f"# Format: <ID>{SEP}<NAME>{SEP}<SENTENCE>\n".encode())

    def yield_lines(self, result: loader.Result):

        # maps result objects to mention -> [sentence1, sentence2, ...]
        for mention, sentences in self.transform_result(result=result).items():

            # maps sentences to mode -> [f(sentence1), f(sentence2), ...]
            for mode, sentences in self.transform_sentences(
                mention=mention, sentences=sentences
            ).items():

                for sentence in sentences:
                    yield mode, mention, sentence

    def write_result(self, e: int, result: loader.Result):
        for mode, mention, sentence in self.yield_lines(result=result):
            fd = self.fd_sentences[mode]
            self.write(fd, f"{e}{SEP}{mention}{SEP}{sentence}")

    def write_text(self):
        self.write_header()
        noctx_ents, noctx_triples = set(), set()

        for e, result in tqdm(self.get_text()):

            if not result:
                self.handle_noctx(
                    e=e,
                    noctx_triples=noctx_triples,
                    noctx_ents=noctx_ents,
                )
                continue

            self.write_result(e=e, result=result)

        log.info(f"finished processing {self.g.name}")
        if len(noctx_triples):
            log.error(
                f"{len(noctx_ents)} entities without context"
                f" ({len(noctx_triples)}/{len(self.g.nx.edges)} triples)"
            )


def create(
    loader: loader.Loader,
    path: Union[str, pathlib.Path],
    seed: int,
    sentences: int = None,
    shuffle: bool = True,
    mask: bool = False,
    mark: bool = False,
):

    g = graph.Graph.load(path / "graph")
    path = helper.path(path, exists=True)
    text_dir = helper.path(path / "text", create=True)

    with (text_dir / "config.yml").open(mode="w") as fd:
        yaml.dump(
            dict(
                seed=seed,
                sentences=sentences,
                shuffle=shuffle,
                mask=mask,
                mark=mark,
                git=helper.git_hash(),
                data=datetime.now(),
            ),
            fd,
        )

    def gopen(f_name: str):
        return gzip.open(str(text_dir / f_name), mode="wb")

    fd_sentences = {Mode.CLEAN: gopen("sentences.clean.txt.gz")}

    if mask:
        fd_sentences[Mode.MASKED] = gopen("sentences.masked.txt.gz")
    if mark:
        fd_sentences[Mode.MARKED] = gopen("sentences.marked.txt.gz")

    with contextlib.ExitStack() as stack:

        Selector(
            g=g,
            sentences=sentences,
            shuffle=shuffle,
            select=stack.enter_context(loader),
            fd_sentences={k: stack.enter_context(v) for k, v in fd_sentences.items()},
            fd_nocontext=stack.enter_context(gopen("nocontext.txt.gz")),
        ).write_text()
