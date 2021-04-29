# -*- coding: utf-8 -*-

from irt.graph import graph
from irt.common import helper

import json
import logging
import pathlib

from typing import Any
from typing import Union
from typing import Generator


log = logging.getLogger(__name__)


# --- | CODEX IMPORTER
#       https://github.com/tsafavi/codex


def load_codex(
    *f_triples: list[str],
    f_rel2id: Union[str, pathlib.Path] = None,
    f_ent2id: Union[str, pathlib.Path] = None,
) -> graph.GraphImport:
    """

    Load CoDEx-like benchmark files

    Structure is as follows:

    f_triples: the graph as (
      head-wikidata-id, relation-wikidata-id, tail-wikidata-id
    ) triples

    f_rel2id: json file containing wikidata-id -> label mappings
    f_ent2id: json file containing wikidata-id -> label mappings

    """

    with helper.path(
        f_rel2id,
        exists=True,
        message="loading relation labels from {path_abbrv}",
    ).open(mode="r") as fd:
        rel2label = json.load(fd)

    with helper.path(
        f_ent2id,
        exists=True,
        message="loading entity labels from {path_abbrv}",
    ).open(mode="r") as fd:
        ent2label = json.load(fd)

    triples = []
    refs = {
        "ents": {"counter": 0, "dic": {}},
        "rels": {"counter": 0, "dic": {}},
    }

    def _get(kind: str, key: str):
        dic = refs[kind]["dic"]

        if key not in dic:
            dic[key] = refs[kind]["counter"]
            refs[kind]["counter"] += 1

        return dic[key]

    for fname in f_triples:

        p_triples = helper.path(
            fname,
            exists=True,
            message="loading CoDEx-like graph from {path_abbrv}",
        )

        with p_triples.open(mode="r") as fd:
            for line in fd:
                gen = zip(("ents", "rels", "ents"), line.strip().split())
                h, r, t = map(lambda a: _get(*a), gen)
                triples.append((h, t, r))  # mind the switch!

    gi = graph.GraphImport(
        triples=triples,
        rels={
            idx: f"{wid}:{rel2label[wid]['label']}"
            for wid, idx in refs["rels"]["dic"].items()
        },
        ents={
            idx: f"{wid}:{ent2label[wid]['label']}"
            for wid, idx in refs["ents"]["dic"].items()
        },
    )

    return gi


# --- | OPEN KE IMPORTER
#       https://github.com/thunlp/OpenKE


def _oke_fn_triples(line: str):
    h, t, r = map(int, line.split())
    return h, t, r


def _oke_fn_idmap(line: str):
    name, idx = line.rsplit(maxsplit=1)
    return int(idx), name.strip()


def _oke_parse(path: str, fn) -> Generator[Any, None, None]:
    if path is None:
        return None

    with open(path, mode="r") as fd:
        fd.readline()
        for i, line in enumerate(fd):
            yield line if fn is None else fn(line)


def load_oke(
    *f_triples: list[str], f_rel2id: str = None, f_ent2id: str = None
) -> graph.GraphImport:
    """

    Load OpenKE-like benchmark files

    Structure is as follows:

    f_triples: the graph as (eid-1, eid-2, rid) triples
    f_rel2id: relation names as (name, rid) tuples
    f_ent2id: entity labels as (label, eid) tuples

    The first line of each file is ignored (contains the number of
    data points in the original data set)

    """
    log.info(f"loading OKE-like graph from {f_triples}")

    triples = set()
    for fname in f_triples:
        triples |= set(_oke_parse(fname, _oke_fn_triples))

    rels = dict(_oke_parse(f_rel2id, _oke_fn_idmap))
    ents = dict(_oke_parse(f_ent2id, _oke_fn_idmap))

    gi = graph.GraphImport(triples=tuple(triples), rels=rels, ents=ents)

    log.info(f"finished parsing {f_triples}")

    return gi


# --- | VILLMOW IMPORTER
#       https://github.com/villmow/datasets_knowledge_embedding
#       https://gitlab.cs.hs-rm.de/jvill_transfer_group/thesis/thesis


def load_vll(f_triples: list[str]) -> graph.GraphImport:
    """

    Load Villmow's benchmark files. Structure is as follows:

    f_triples: the graph encoded as string triples (e1, r, e2)

    """
    log.info(f"loading villmow-like graph from {f_triples}")

    refs = {
        "ents": {"counter": 0, "dic": {}},
        "rels": {"counter": 0, "dic": {}},
    }

    def _get(kind: str, key: str):
        dic = refs[kind]["dic"]

        if key not in dic:
            dic[key] = refs[kind]["counter"]
            refs[kind]["counter"] += 1

        return dic[key]

    triples = set()
    for fname in f_triples:
        with open(fname, mode="r") as fd:
            for line in fd:

                gen = zip(("ents", "rels", "ents"), line.strip().split())
                h, r, t = map(lambda a: _get(*a), gen)
                triples.add((h, t, r))  # mind the switch

    gi = graph.GraphImport(
        triples=triples,
        rels={idx: name for name, idx in refs["rels"]["dic"].items()},
        ents={idx: name for name, idx in refs["ents"]["dic"].items()},
    )

    log.info(f"finished parsing {f_triples}")

    return gi
