# -*- coding: utf-8 -*-

"""

migrate legacy ryn text datasets to the new irt format

"""

import irt
from irt.common import helper
from irt.text import selector
from irt.text import loader as text_loader

import re
import gzip
import logging
import pathlib
from collections import deque
from collections import defaultdict

from typing import Optional


log = logging.getLogger(__name__)


SEP = "|"


class Legacy(text_loader.Loader):

    path: pathlib.Path

    class Selector(text_loader.Selector):

        e2result: dict[int, tuple[str, str]]

        def __init__(self, e2result):
            self.e2result = e2result

        def by_entity(self, e: int) -> Optional[text_loader.Result]:
            if e not in self.e2result:
                return None

            mentions, blobs = zip(*self.e2result[e])
            res = text_loader.Result(mentions=mentions, blobs=blobs)
            return res

    def _read(self, path, fname):
        with gzip.open(str(path / fname), mode="r") as fd:
            log.info(f"reading {path.name}/{fname}")
            fd.readline()  # consume head comment

            for line in map(bytes.decode, fd.readlines()):
                e_str, label, context = map(str.strip, line.split(SEP, maxsplit=2))
                yield int(e_str), label, context

    def _read_mult(self, path, *fnames):
        for fname in fnames:
            yield from self._read(path, fname)

    def __init__(self, path):
        self.path = helper.path(path, exists=True)

    def __enter__(self):

        # optionally reverse engineer the mention from
        # the marked dataset

        files = (
            "cw.train-contexts.txt.gz",
            "ow.valid-contexts.txt.gz",
            "ow.test-contexts.txt.gz",
        )

        e2mention = defaultdict(deque)
        marked_path = self.path / "bert-base-cased.30.768.marked"

        if marked_path.exists():
            count = 0
            rex = re.compile(r"\[MENTION_START\](.+?)\[MENTION_END\]")
            for e, label, context in self._read_mult(marked_path, *files):
                match = re.search(rex, context)

                if match:
                    mention = match[1].strip()
                else:
                    mention = ""
                    count += 1

                e2mention[e].append(mention)

            log.info(f"there are {count} contexts without explicit mentions")

        e2result = defaultdict(list)
        clean_path = self.path / "bert-base-cased.30.768.clean"
        for e, label, context in self._read_mult(clean_path, *files):
            mention = e2mention[e].popleft() if e in e2mention else label
            e2result[e].append((mention, context))

        return Legacy.Selector(e2result=e2result)

    def __exit__(self, *_):
        pass


if __name__ == "__main__":

    # irt-cde conversion 21/04/25
    selector.create(
        loader=Legacy(
            irt.ENV.SRC_DIR / "legacy/cde.m_8051991_27/contexts-v7-2020-12-31.db"
        ),
        path=irt.ENV.DATASET_DIR / "irt-cde",
        contexts=30,
        seed=8051991,
        shuffle=False,
        mark=True,
        mask=True,
    )

    # irt-fb conversion 21/04/25
    pass
