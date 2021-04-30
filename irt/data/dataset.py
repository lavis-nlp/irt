# -*- coding: utf-8 -*-


from irt import text
from irt.graph import graph
from irt.graph import split as graph_split
from irt.common import helper

import gzip
import pathlib
import logging
import textwrap
import statistics

from functools import lru_cache
from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict

from typing import Set
from typing import Dict
from typing import Tuple
from typing import Union


def _ents_from_triples(triples):
    if not triples:
        return set()

    hs, ts, _ = zip(*triples)
    return set(hs) | set(ts)


log = logging.getLogger(__name__)


@dataclass(eq=False)  # id based hashing
class Part:

    name: str
    owe: Set[int]  # open world entities
    triples: Set[Tuple[int]]

    @property
    @lru_cache
    def graph(self) -> graph.Graph:
        return graph.Graph(source=graph.GraphImport(triples=self.triples))

    @property
    @lru_cache
    def entities(self):
        return _ents_from_triples(self.triples)

    @property
    @lru_cache
    def heads(self) -> Set[int]:
        if not self.triples:
            return set()

        return set(tuple(zip(*self.triples))[0])

    @property
    @lru_cache
    def tails(self) -> Set[int]:
        if not self.triples:
            return set()

        return set(tuple(zip(*self.triples))[1])

    @property
    def description(self) -> str:
        return (
            f"owe: {len(self.owe)}\n"
            f"entities: {len(self.entities)}\n"
            f"heads: {len(self.heads)}\n"
            f"tails: {len(self.tails)}\n"
            f"triples: {len(self.triples)}\n"
        )

    # ---

    def __str__(self) -> str:
        return f"{self.name}={len(self.triples)}"

    def __or__(self, other: "Part") -> "Part":
        return Part(
            name=f"{self.name}|{other.name}",
            owe=self.owe | other.owe,
            triples=self.triples | other.triples,
        )


@dataclass
class Split:
    """
    Container class for a split dataset

    """

    cfg: graph_split.Config
    concepts: Set[int]

    closed_world: Part
    open_world_valid: Part
    open_world_test: Part

    # ---

    @property
    def description(self) -> str:
        s = f"IRT SPLIT\n{len(self.concepts)} retained concepts\n\n{self.cfg}"

        # functools.partial not applicable :(
        def _indent(s):
            return textwrap.indent(s, "  ")

        s += f"\nClosed World - TRAIN:\n{_indent(self.closed_world.description)}"
        s += f"\nOpen World - VALID:\n{_indent(self.open_world_valid.description)}"
        s += f"\nOpen World - TEST:\n{_indent(self.open_world_test.description)}"

        return s

    # ---

    def __str__(self) -> str:
        return "IRT split: " + (
            " | ".join(
                f"{part}"
                for part in (
                    self.closed_world,
                    self.open_world_valid,
                    self.open_world_test,
                )
            )
        )

    def __getitem__(self, key: str):
        return {
            "cw.train": self.closed_world,
            "ow.valid": self.open_world_valid,
            "ow.test": self.open_world_test,
        }[key]

    # ---

    def check(self):
        """

        Run some self diagnosis

        """
        log.info("! running self-check for dataset split")

        # no triples must be shared between splits

        triplesets = (
            ("cw.train", self.closed_world.triples),
            ("ow.valid", self.open_world_valid.triples),
            ("ow.test", self.open_world_test.triples),
        )

        for (n1, s1), (n2, s2) in combinations(triplesets, 2):
            assert s1.isdisjoint(s2), f"{n1} and {n2} share triples"

        # no ow entities must be shared between splits

        owesets = (
            ("cw.train", self.closed_world.owe),
            ("ow.valid", self.open_world_valid.owe),
            ("ow.test", self.open_world_test.owe),
        )

        for (n1, s1), (n2, s2) in combinations(owesets, 2):
            assert s1.isdisjoint(s2), f"{n1} and {n2} share owe entities"

        # ow entities must not be seen in earlier splits and no ow
        # entities must occur in cw.valid (use .entities property
        # which gets this information directly from the triple sets)

        assert (
            self.closed_world.owe == self.closed_world.entities
        ), "cw.train owe != cw.train entities"

        seen = self.closed_world.entities
        if self.cfg.strict:
            assert self.open_world_valid.owe.isdisjoint(
                seen
            ), "entities in ow valid leaked"
        else:
            log.warning("entities in ow valid leaked!")

        seen |= self.open_world_valid.entities
        if self.cfg.strict:
            assert self.open_world_test.owe.isdisjoint(
                seen
            ), "entities in ow test leaked"
        else:
            log.warning("entities in ow test leaked!")

        # each triple of the open world splits must contain at least
        # one open world entity
        for part in (self.open_world_valid, self.open_world_test):
            undesired = set(
                (h, t, r)
                for h, t, r in part.triples
                if h not in part.owe and t not in part.owe
            )

            if self.cfg.strict:
                # deactivate for fb15k237-owe
                assert not len(undesired), f"found undesired triples: len({undesired})"

            if len(undesired):
                log.error(
                    f"there are {len(undesired)} triples containing"
                    f" only closed world entities in {part.name}"
                )

    # ---

    @classmethod
    def load(K, path: Union[str, pathlib.Path]):
        log.info("loading split data")

        path = helper.path(path, exists=True)
        cfg = graph_split.Config.load(path / "config.yml")

        with (path / "concepts.txt").open(mode="r") as fd:
            # we are only interested in the entity ids
            concepts = {int(e) for e, _ in (line.split(maxsplit=1) for line in fd)}

        parts, seen = {}, set()
        for name in ("closed_world", "open_world-valid", "open_world-test"):
            log.info(f"initializing part: {name}")
            key = name.replace("-", "_")

            with (path / f"{name}.txt").open(mode="r") as fd:
                triples = {tuple(map(int, line.split())) for line in fd}

            entities = _ents_from_triples(triples)

            part = Part(
                name=name,
                owe=entities - seen,
                triples=triples,
            )

            parts[key] = part
            seen |= entities

        self = K(cfg=cfg, concepts=concepts, **parts)
        return self


@dataclass(frozen=True)
class TextSample:

    mention: str
    context: str


class Text(defaultdict):
    def __init__(self):
        super().__init__(set)

    def __str__(self) -> str:
        fmt = "irt text: ~{mean_contexts:.2f} text contexts per entity"
        return fmt.format(**self.stats)

    @property
    def stats(self) -> Dict[str, float]:
        contexts, mentions = zip(
            *[
                (len(samples), len({sample.mention for sample in samples}))
                for samples in self.values()
            ]
        )

        return dict(
            mean_contexts=statistics.mean(contexts),
            median_contexts=statistics.median(contexts),
            mean_mentions=statistics.mean(mentions),
            median_mentions=statistics.median(mentions),
        )

    # fmt: off
    @property
    def description(self):
        s = "IRT Text (per entity)\n"

        stats = (
            "mean contexts: {mean_contexts:.2f}\n"
            "median contexts: {median_contexts:.2f}\n"
            "mean mentions: {mean_mentions:.2f}\n"
            "median mentions: {median_mentions:.2f}\n"
        ).format(**self.stats)

        s += textwrap.indent(stats, "  ")
        return s
    # fmt: on

    # ---

    @classmethod
    def load(K, path: Union[str, pathlib.Path], mode: text.Mode):
        log.info(f"loading text data ({mode.value=})")

        self = K()

        path = helper.path(path, exists=True)
        fpath = path / text.Mode.filename(mode)

        with gzip.open(str(fpath), mode="rb") as fd:

            header = next(fd).decode().strip()
            assert header.startswith("#")

            splits = (
                # strip each value of the split lines
                map(str.strip, line.decode().split(text.SEP, maxsplit=2))
                for line in fd
                if line.strip()
            )

            for e, mention, context in splits:
                sample = TextSample(mention=mention, context=context)
                self[int(e)].add(sample)

        return self


class Dataset:
    """

    IRT Dataset

    TODO: documentation

    """

    graph: graph.Graph
    split: Split
    text: Text  # i.e. Dict[int, set[TextSample]]

    # ---

    @property
    def name(self) -> str:
        return self.graph.name

    @property
    def config(self) -> graph_split.Config:
        return self.split.cfg

    @property
    def id2ent(self) -> Dict[int, str]:
        return self.graph.source.ents

    @property
    def id2rel(self) -> Dict[int, str]:
        return self.graph.source.rels

    # ---

    def __str__(self):
        return f"IRT dataset:\n{self.graph}\n{self.split}\n{self.text}"

    # fmt: off
    @property
    @lru_cache
    def description(self) -> str:
        return (
            f"IRT DATASET\n\n"
            f"{self.graph.description}\n"
            f"{self.split.description}\n"
            f"{self.text.description}"
        )
    # fmt: on

    # ---
    # initialization

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        mode: text.Mode = text.Mode.CLEAN,
        check: bool = False,
    ):
        self._check = check
        path = helper.path(path, exists=True)

        self.graph = graph.Graph.load(path=path / "graph")
        self.split = Split.load(path=path / "split")
        self.text = Text.load(path=path / "text", mode=mode)
