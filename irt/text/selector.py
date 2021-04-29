# -*- coding: utf-8 -*-

"""

assigns text to entities

"""

from irt import text
from irt.text import loader
from irt.graph import graph
from irt.common import helper

import gzip
import random
import pathlib
import logging
import contextlib

from functools import partial
from datetime import datetime
from dataclasses import dataclass

import yaml
from tqdm import tqdm as _tqdm

from typing import IO
from typing import Union


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


# this matches masked sequences of the upstream sqlite context dbs
MARKED = text.TOK_MENTION_START + " {mention} " + text.TOK_MENTION_END


def _filter(context, mention):
    return all(
        (
            # at least 50 chars per context
            len(context) > 50,
            # no File: prefixed blobs
            not context.startswith("File:"),
        )
    )


# ---


@dataclass
class Selector:

    g: graph.Graph

    contexts: list[int]
    shuffle: bool

    select: loader.Loader

    fd_nocontext: IO[bytes]
    fd_contexts: dict[str, IO[bytes]]

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
            f"{e}{text.SEP}{name}{text.SEP}{count}",
        )

        msg = f"! no context for {e}: {name} ({count} triples)"
        log.info(msg)

    def transform_result(self, result=loader.Result):

        # list(set(x)) is not possible because of the
        # unpredictable python hash seeds
        tuples, seen = [], set()

        for blob, mention in zip(result.blobs, result.mentions):

            # filter BEFORE mapping
            gen = (s for s in blob.split("\n") if _filter(s, mention))

            for context in gen:
                if context in seen:
                    continue

                seen.add(context)
                tuples.append((mention, context))

        if self.shuffle:
            random.shuffle(tuples)

        # select n contexts each
        return tuples[: self.contexts]

    def transform_context(self, mention: str, context: str):

        context = " ".join(context.strip().split())

        ret = {text.Mode.CLEAN: context}

        if text.Mode.MASKED in self.fd_contexts:
            ret[text.Mode.MASKED] = (
                context.replace(mention, text.MASK_TOKEN) if mention else context
            )

        if text.Mode.MARKED in self.fd_contexts:
            ret[text.Mode.MARKED] = (
                context.replace(mention, MARKED.format(mention=mention))
                if mention
                else context
            )

        return ret

    def write(self, fd, s: str):
        assert "\n" not in s, f"{s=}"
        fd.write((s + "\n").encode())

    def write_header(self):
        self.fd_nocontext.write(
            f"# Format: <ID>{text.SEP}<NAME>{text.SEP}<TRIPLES>\n".encode()
        )

        for fd in self.fd_contexts.values():
            fd.write(f"# Format: <ID>{text.SEP}<NAME>{text.SEP}<CONTEXT>\n".encode())

    def yield_lines(self, result: loader.Result):
        for mention, context in self.transform_result(result=result):
            for mode, context in self.transform_context(
                mention=mention, context=context
            ).items():

                yield mode, mention, context

    def write_result(self, e: int, result: loader.Result):
        for mode, mention, context in self.yield_lines(result=result):
            fd = self.fd_contexts[mode]

            assert text.SEP not in mention
            assert text.SEP not in context

            self.write(fd, f"{e}{text.SEP}{mention}{text.SEP}{context}")

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
    contexts: int = None,
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
                contexts=contexts,
                shuffle=shuffle,
                mask=mask,
                mark=mark,
                git=helper.git_hash(),
                data=datetime.now(),
            ),
            fd,
        )

    def _gz_open(fname: str):
        return gzip.open(str(text_dir / fname), mode="wb")

    def _ctx_open(mode: text.Mode):
        return _gz_open(fname=text.Mode.filename(mode))

    fd_contexts = {text.Mode.CLEAN: _ctx_open(text.Mode.CLEAN)}

    if mask:
        fd_contexts[text.Mode.MASKED] = _ctx_open(text.Mode.MASKED)
    if mark:
        fd_contexts[text.Mode.MARKED] = _ctx_open(text.Mode.MARKED)

    with contextlib.ExitStack() as stack:

        Selector(
            g=g,
            contexts=contexts,
            shuffle=shuffle,
            select=stack.enter_context(loader),
            fd_contexts={k: stack.enter_context(v) for k, v in fd_contexts.items()},
            fd_nocontext=stack.enter_context(_gz_open("nocontext.txt.gz")),
        ).write_text()
