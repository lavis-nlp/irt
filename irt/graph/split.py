# -*- coding: utf-8 -*-

"""

Create graph splits exposing tbox proxies.

"""

from irt.graph import graph
from irt.common import helper

import yaml
import random
import pathlib
import textwrap

import logging
import dataclasses
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass
from itertools import combinations

from typing import Set
from typing import Dict
from typing import List
from typing import Union
from irt.graph.graph import Triple


log = logging.getLogger(__name__)


def _ents_from_triples(triples):
    if not triples:
        return set()

    hs, ts, _ = zip(*triples)
    return set(hs) | set(ts)


# ---


@dataclass
class Config:

    seed: int

    ow_split: float
    ow_train_split: float

    # no of relation (sorted by ratio)
    threshold: int

    # manually add or remove relations to be considered
    excludelist: Set[str]
    includelist: Set[str]

    # strict checks (deactivated for other splits such as vll.fb15k237-OWE)
    # if strict is True, no ow-entity must be in any "preceeding" split
    # (see Dataset.check for the constraints)
    strict: bool = True

    # post-init

    git: str = None  # revision hash
    date: datetime = None

    def __post_init__(self):
        self.git = helper.git_hash()
        self.date = datetime.now()

    def __str__(self) -> str:
        return "Config:\n" + textwrap.indent(
            (
                f"seed: {self.seed}\n"
                f"ow split: {self.ow_split}\n"
                f"ow train split: {self.ow_train_split}\n"
                f"relation threshold: {self.threshold}\n"
                f"git: {self.git}\n"
                f"date: {self.date}\n"
            ),
            "  ",
        )

    # ---

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(path, message="saving config to {path_abbrv}")
        raw = dataclasses.asdict(
            dataclasses.replace(
                self,
                excludelist=list(self.excludelist),
                includelist=list(self.includelist),
            )
        )

        with path.open(mode="w") as fd:
            yaml.dump(raw, fd)

    @classmethod
    def load(K, path: Union[str, pathlib.Path]) -> "Config":
        path = helper.path(
            path, exists=True, message="loading config from {path_abbrv}"
        )

        with path.open(mode="r") as fd:
            self = K(**yaml.load(fd, Loader=yaml.FullLoader))

        return dataclasses.replace(
            self, excludelist=set(self.excludelist), includelist=set(self.includelist)
        )


@dataclass(eq=False)  # id based hashing
class Part:

    name: str
    owe: Set[int]  # open world entities
    triples: Set[Triple]
    concepts: Set[int]

    @property
    @lru_cache
    def g(self) -> graph.Graph:
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
    @lru_cache
    def linked_concepts(self) -> Set[int]:
        return self.entities & self.concepts

    @property
    @lru_cache
    def concept_triples(self) -> Set[Triple]:
        g = graph.Graph(source=graph.GraphImport(triples=self.triples))
        return g.find(heads=self.concepts, tails=self.concepts)

    @property
    def str_stats(self) -> str:
        return (
            f"owe: {len(self.owe)}\n"
            f"triples: {len(self.triples)}\n"
            f"entities: {len(self.entities)}\n"
            f"heads: {len(self.heads)}\n"
            f"tails: {len(self.tails)}"
        )

    # ---

    def __str__(self) -> str:
        return f"{self.name}={len(self.triples)}"

    def __or__(self, other: "Part") -> "Part":
        return Part(
            name=f"{self.name}|{other.name}",
            owe=self.owe | other.owe,
            triples=self.triples | other.triples,
            concepts=self.concepts | other.concepts,
        )


@dataclass
class Dataset:
    """
    Container class for a split dataset

    """

    path: pathlib.Path

    cfg: Config
    g: graph.Graph

    id2ent: Dict[int, str]
    id2rel: Dict[int, str]

    concepts: Set[int]

    cw_train: Part
    ow_valid: Part
    ow_test: Part

    # ---

    @property
    def name(self):
        return self.path.name

    @property
    def str_stats(self) -> str:
        s = (
            "IRT.SPLIT DATASET\n"
            f"-----------------\n"
            f"\n{len(self.concepts)} retained concepts\n\n"
            f"{self.cfg}\n"
            f"{self.g.str_stats}\n"
            f"{self.path}\n"
        )

        # functools.partial not applicable :(
        def _indent(s):
            return textwrap.indent(s, "  ")

        s += f"\nClosed World - TRAIN:\n{_indent(self.cw_train.str_stats)}"
        s += f"\nOpen World - VALID:\n{_indent(self.ow_valid.str_stats)}"
        s += f"\nOpen World - TEST:\n{_indent(self.ow_test.str_stats)}"

        return s

    # ---

    def __str__(self) -> str:
        return f"split [{self.name}]: " + (
            " | ".join(
                f"{part}"
                for part in (
                    self.cw_train,
                    self.ow_valid,
                    self.ow_test,
                )
            )
        )

    def __getitem__(self, key: str):
        return {
            "cw.train": self.cw_train,
            "ow.valid": self.ow_valid,
            "ow.test": self.ow_test,
        }[key]

    # ---

    def check(self):
        """

        Run some self diagnosis

        """
        log.info(f"! running self-check for {self.path.name} Dataset")

        # no triples must be shared between splits

        triplesets = (
            ("cw.train", self.cw_train.triples),
            ("ow.valid", self.ow_valid.triples),
            ("ow.test", self.ow_test.triples),
        )

        for (n1, s1), (n2, s2) in combinations(triplesets, 2):
            assert s1.isdisjoint(s2), f"{n1} and {n2} share triples"

        # no ow entities must be shared between splits

        owesets = (
            ("cw.train", self.cw_train.owe),
            ("ow.valid", self.ow_valid.owe),
            ("ow.test", self.ow_test.owe),
        )

        for (n1, s1), (n2, s2) in combinations(owesets, 2):
            assert s1.isdisjoint(s2), f"{n1} and {n2} share owe entities"

        # ow entities must not be seen in earlier splits
        # and no ow entities must occur in cw.valid
        # (use .entities property which gets this information directly
        # directly from the triple sets)

        assert (
            self.cw_train.owe == self.cw_train.entities
        ), "cw.train owe != cw.train entities"

        seen = self.cw_train.entities
        if self.cfg.strict:
            assert self.ow_valid.owe.isdisjoint(seen), "entities in ow valid leaked"
        else:
            log.warning("entities in ow valid leaked!")

        seen |= self.ow_valid.entities
        if self.cfg.strict:
            assert self.ow_test.owe.isdisjoint(seen), "entities in ow test leaked)"
        else:
            log.warning("entities in ow test leaked!")

        # each triple of the open world splits must contain at least
        # one open world entity
        for part in (self.ow_valid, self.ow_test):
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

    @classmethod
    def load(K, path: Union[str, pathlib.Path]) -> "Dataset":
        """

        Load a dataset from disk

        Parameters
        ----------

        path : Union[str, pathlib.Path]
          Folder containing all necessary files

        """
        path = pathlib.Path(path)

        cfg = Config.load(path / "cfg.yml")
        g = graph.Graph.load(path / "graph")

        with (path / "concepts.txt").open(mode="r") as fd:
            num = int(fd.readline())
            concepts = set(map(int, fd.readlines()))
            assert num == len(concepts)

        def _load_dict(filep):
            with filep.open(mode="r") as fd:
                num = int(fd.readline())
                gen = map(lambda l: l.rsplit(" ", maxsplit=1), fd.readlines())
                d = dict((int(val), key.strip()) for key, val in gen)

                assert num == len(d)
                return d

        id2ent = _load_dict(path / "entity2id.txt")
        id2rel = _load_dict(path / "relation2id.txt")

        def _load_part(fp, seen: Set[int], name: str) -> Part:
            nonlocal concepts

            with fp.open(mode="r") as fd:
                num = int(fd.readline())
                triples = set(
                    tuple(map(int, line.split(" "))) for line in fd.readlines()
                )

            assert num == len(triples)

            ents = _ents_from_triples(triples)
            part = Part(
                name=name.replace("_", "."),
                triples=triples,
                owe=ents - seen,
                concepts=concepts,
            )

            return part

        parts = {}
        seen = set()

        for name, fp in (
            ("cw_train", path / "cw.train2id.txt"),
            ("ow_valid", path / "ow.valid2id.txt"),
            ("ow_test", path / "ow.test2id.txt"),
        ):

            part = _load_part(fp, seen, name)
            seen |= part.entities
            parts[name] = part

        self = K(
            path=path,
            g=g,
            cfg=cfg,
            concepts=concepts,
            id2ent=id2ent,
            id2rel=id2rel,
            **parts,
        )

        self.check()
        return self


# ---


@dataclass
class Relation:

    r: int
    name: str
    triples: Set[Triple]

    hs: Set[int]
    ts: Set[int]

    ratio: float

    @property
    def concepts(self) -> Set[int]:
        # either head or tail sets (whichever is smaller)
        reverse = len(self.hs) <= len(self.ts)
        return self.hs if reverse else self.ts

    @classmethod
    def from_graph(K, g: graph.Graph) -> List["Relation"]:
        rels = []
        for r, relname in g.source.rels.items():
            triples = g.find(edges={r})
            hs, ts = map(set, zip(*((h, t) for h, t, _ in triples)))

            lens = len(hs), len(ts)
            ratio = min(lens) / max(lens)

            rels.append(
                K(
                    r=r,
                    name=relname,
                    triples=triples,
                    hs=hs,
                    ts=ts,
                    ratio=ratio,
                )
            )

        return rels


@dataclass
class Splitter:

    name: str
    cfg: Config
    g: graph.Graph
    path: pathlib.Path

    @property
    def rels(self) -> List[Relation]:
        return self._rels

    def __post_init__(self):
        self._rels = Relation.from_graph(self.g)
        self._rels.sort(key=lambda rel: rel.ratio)

    def create(self):
        """Create a new split dataset

        Three constraints apply:

        1. All concept entities must appear at least once in
           cw.train. Triples containing concept entities are
           distributed over all splits.

        2. The number of zero-shot entities needs to be maximised in
           ow: These entities are first encountered in their
           respective split; e.g. a zero-shot entity of ow.valid must
           not be present in any triple of cw.train oder cw.valid but
           may be part of an ow.test triples.

        3. The amount of triples should be balanced by the provided
           configuration (cfg.ow_split, cfg.ow_train_split).

        """
        log.info(f"create {self.name=}")

        relations = self.rels[: self.cfg.threshold]
        relations += [r for r in self.rels if r.name in self.cfg.includelist]
        relations = [r for r in relations if r.name not in self.cfg.excludelist]

        concepts = set.union(*(rel.concepts for rel in relations))
        log.info(f"{len(concepts)=}")

        candidates = sorted(set(self.g.source.ents) - concepts)
        random.shuffle(candidates)

        _p = int(self.cfg.ow_split * 100)
        log.info(f"targeting {_p}% of all triples for cw")

        # there are two thresholds:
        # 0 < t1 < t2 < len(triples) = n
        # where:
        #  0-t1:   ow test
        #  t1-t2:  ow valid
        #  t2-n:   cw

        n = len(self.g.source.triples)
        t2 = int(n * (1 - self.cfg.ow_split))
        t1 = int(t2 * (1 - self.cfg.ow_train_split))

        log.info(f"target splits: 0 {t1=} {t2=} {n=}")

        # retain all triples where both head and tail
        # are concept entities for cw train
        retained = self.g.select(heads=concepts, tails=concepts)
        agg = retained.copy()

        cw = retained
        ow_valid = set()
        ow_test = set()

        while candidates:
            e = candidates.pop()

            # it does this (but faster):
            # found = set(
            #     (h, t, r) for h, t, r in self.g.source.triples
            #     if h == e or t == e)
            found = self.g.find(heads={e}, tails={e})

            found -= agg
            agg |= found
            curr = len(agg)

            if not found:
                continue

            elif curr < t1:
                ow_test |= found
            elif curr < t2:
                ow_valid |= found
            else:
                cw |= found

        # ---

        assert len(agg) == len(self.g.source.triples)
        assert len(cw | ow_valid | ow_test) == len(self.g.source.triples)
        log.info(f"split {len(cw)=} and " f"{len((ow_valid|ow_test))=} triples")

        # ---

        log.info("writing")
        self.write(concepts=concepts, cw=cw, ow_valid=ow_valid, ow_test=ow_test)

    def write(
        self,
        concepts: Set[int],
        cw: Set[Triple],
        ow_valid: Set[Triple],
        ow_test: Set[Triple],
    ):
        def _write(name, triples):
            with (self.path / name).open(mode="w") as fd:
                fd.writelines(f"{h} {t} {r}\n" for h, t, r in triples)

        _write("closed_world.txt", cw)
        _write("open_world-valid.txt", ow_valid)
        _write("open_world-test.txt", ow_test)

        with (self.path / "concepts.txt").open(mode="w") as fd:
            fd.writelines(f"{e} {self.g.source.ents[e]}\n" for e in concepts)

        self.cfg.save(self.path / "config.yml")


def create(g: graph.Graph, cfg: Config):
    name = f'{g.name.split("-")[0]}'
    name += f"_{cfg.seed}_{cfg.threshold}"

    log.info(f"! creating dataset {name=}")

    helper.seed(cfg.seed)
    Splitter(g=g, cfg=cfg, name=name).create()
