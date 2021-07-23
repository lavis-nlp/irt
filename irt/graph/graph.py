# -*- coding: utf-8 -*-


"""

graph abstraction

"""

from irt.common import helper

import yaml
import networkx
import numpy as np

import pathlib
import logging
from functools import partial

from dataclasses import field
from dataclasses import dataclass
from dataclasses import FrozenInstanceError
from collections import defaultdict

from typing import Union
from typing import NewType
from typing import Iterable


log = logging.getLogger(__name__)
Triple = NewType("Triple", tuple[int, int, int])


# ----------| DATA


class frozendict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._frozen = True

    def __setitem__(self, *args, **kwargs):
        try:
            self._frozen

        except AttributeError:
            super().__setitem__(*args, **kwargs)
            return

        raise FrozenInstanceError("mapping is frozen")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._frozen = True


@dataclass(frozen=True)
class GraphImport:
    """

    Unified data definition used by irt

    Graph triples are of the following structure: (head, tail, relation)
    You can provide any Iterable for the triples. They are converted
    to frozenset[triple]

    Currently the graph is defined by it's edges, which means each
    node is at least connected to one other node. This might change in
    the future.

    Order of provided triples is not preserved.

    The rels and ents dictionaries are filled with all missing
    information automatically such that e.g. rels[i] = f'{i}'.
    They cannot be changed afterwards.

    """

    # (head, tail, relation)
    triples: set[Triple]

    rels: dict[int, str] = field(default_factory=dict)
    ents: dict[int, str] = field(default_factory=dict)

    # --

    def _set(self, prop: str, *args, **kwargs):
        object.__setattr__(self, prop, *args, **kwargs)

    def _set_all(self, triples, ents, rels):
        self._set("triples", frozenset(triples))
        self._set("ents", frozendict(ents))
        self._set("rels", frozendict(rels))

    def _resolve(self, idx: int, mapping: dict[int, str]):
        if idx not in mapping:
            label = str(idx)
            mapping[idx] = label

    def __post_init__(self):
        triples = set(map(tuple, self.triples))

        for h, t, r in self.triples:
            self._resolve(h, self.ents)
            self._resolve(t, self.ents)
            self._resolve(r, self.rels)

        self._set_all(triples, self.ents, self.rels)

    # --

    def join(self, other: "GraphImport") -> "GraphImport":
        ents = {**self.ents, **other.ents}
        rels = {**self.rels, **other.rels}
        triples = self.triples | other.triples

        self._set_all(triples, ents, rels)

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(
            path,
            create=True,
            message="saving graph import to {path_abbrv}",
        )

        with (path / "triples.txt").open(mode="w") as fd:
            fd.writelines(f"{h} {t} {r}\n" for h, t, r in self.triples)

        with (path / "entities.txt").open(mode="w") as fd:
            fd.writelines(f"{e} {name}\n" for e, name in self.ents.items())

        with (path / "relations.txt").open(mode="w") as fd:
            fd.writelines(f"{r} {name}\n" for r, name in self.rels.items())

    @classmethod
    def load(K, path: Union[str, pathlib.Path]):
        path = helper.path(
            path,
            create=True,
            message="loading graph import from {path_abbrv}",
        )

        with (path / "triples.txt").open(mode="r") as fd:
            triples = set(tuple(map(int, line.split())) for line in fd)

        split = partial(str.split, maxsplit=1)

        def _load_dict(fd):
            lines = (split(line) for line in fd)
            return dict((int(i), name.strip()) for i, name in lines)

        with (path / "entities.txt").open(mode="r") as fd:
            ents = _load_dict(fd)

        with (path / "relations.txt").open(mode="r") as fd:
            rels = _load_dict(fd)

        return K(triples=triples, ents=ents, rels=rels)


class Graph:
    """
    IRT Graph Abstraction

    Create a new graph object which maintains a networkx graph.
    This class serves as a provider of utilities working on
    and for initializing the networkx graph.

    Design Decisions
    ----------------

    Naming:

    Naming nodes and edges: networkx uses "nodes" and "edges". To
    not confuse on which "level" you operate on the graph, everything
    here is called "ents" (for entities) and "rels" (for relations)
    when working with IRT code and "node" and "edges" when working
    with networkx instances.

    Separate Relation and Entitiy -Mapping:

    The reasoning for not providing (e.g.) the Graph.source.rels
    mapping directly on the graph is to avoid a false expectation
    that this is automatically in sync with the graph itself.
    Consider manipulating Graph.g (deleting nodes for example) -
    this would not update the .rels-mapping. Thus this is explicitly
    separated in the .source GraphImport.

    """

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> str:
        self._name = val

    @property
    def source(self) -> GraphImport:
        return self._source

    @property
    def nx(self) -> networkx.MultiDiGraph:
        return self._nx

    @property
    def rnx(self) -> networkx.MultiDiGraph:
        return self._rnx

    @property
    def edges(self) -> dict[int, Triple]:
        return self._edges

    @property
    def description(self) -> str:
        s = (
            f"IRT GRAPH: {self.name}\n"
            f"  nodes: {self.nx.number_of_nodes()}\n"
            f"  edges: {self.nx.number_of_edges()}"
            f" ({len(self.source.rels)} types)\n"
        )

        try:
            degrees = np.array(list(self.nx.degree()))[:, 1]
            s += (
                f"  degree:\n"
                f"    mean {np.mean(degrees):.2f}\n"
                f"    median {int(np.median(degrees)):d}\n"
            )

        except IndexError:
            s += "  cannot measure degree\n"

        return s

    # --

    def __str__(self) -> str:
        return f"IRT graph: [{self.name}] ({len(self.source.ents)} entities)"

    def __init__(
        self,
        name: str = None,
        source: GraphImport = None,
    ):

        assert type(name) is str if name is not None else True, f"{name=}"

        # properties
        self._nx = networkx.MultiDiGraph()
        self._edges = defaultdict(set)

        self._name = "unknown" if name is None else name

        # GraphImport
        if source is not None:
            self._source = source
            self.add(source)
        else:
            self._source = GraphImport(triples=[])

        log.debug(f"created graph: \n{self.description}\n")

    # --

    def select(
        self,
        heads: set[int] = None,
        tails: set[int] = None,
        edges: set[int] = None,
    ):
        """

        Select edges from the graph.

        An edge is a triple (h, t, r) and the selection is either
        the union or intersection of all edges containing the
        provided nodes and edges.

        The difference between Graph.find and Graph.select is that
        .find will select any edge containing any of the provided
        heads (union) or tails and .select will only choose those
        where their any combination of all provided entities occurs
        (intersection).

        Parameters
        ----------

        heads : Set[int]
          consider the provided head nodes

        tails : Set[int]
          consider the provided head nodes

        edges : Set[int]
          consider the provided edge classes


        Returns
        -------

        A set of edges adhering to the provided constraints.


        Notes
        -----

        Not using nx.subgraph as it would contain undesired edges
        (because nx.subgraph only works on node-level)

        """
        heads = set() if heads is None else heads
        tails = set() if tails is None else tails
        edges = set() if edges is None else edges

        def _gen(nxg, heads, tails, edges, rev=False):
            for h in heads:
                if h not in nxg:
                    continue

                for t, rs in nxg[h].items():
                    if tails and t not in tails:
                        continue

                    for r in rs:
                        if edges and r not in edges:
                            continue

                        yield (h, t, r) if not rev else (t, h, r)

        dom = set(_gen(self.nx, heads, tails, edges))
        rng = set(_gen(self.rnx, tails, heads, edges, rev=True))

        return dom | rng

    # --

    def find(
        self,
        heads: set[int] = None,
        tails: set[int] = None,
        edges: set[int] = None,
    ) -> set[Triple]:
        """
        Find edges in the graph.

        An edge is a triple (h, t, r) and the selection is either
        the union or intersection of all edges containing one of the
        provided nodes and edges.

        The difference between Graph.find and Graph.select is that
        .find will select any edge containing any of the provided
        heads (union) or tails and .select will only choose those
        where their any combination of all provided entities occurs
        (intersection).

        Parameters
        ----------

        heads : Set[int]
          consider the provided head nodes

        tails : Set[int]
          consider the provided head nodes

        edges : Set[int]
          consider the provided edge classes

        Returns
        -------

        A set of edges adhering to the provided constraints.


        Notes
        -----

        Not using nx.subgraph as it would contain undesired edges
        (because nx.subgraph only works on node-level)

        """
        heads = set() if heads is None else heads
        tails = set() if tails is None else tails
        edges = set() if edges is None else edges

        def _gen(nxg, heads, rev=False):
            for h in heads:
                if h not in nxg:
                    continue

                for t, rs in nxg[h].items():
                    for r in rs:
                        yield (h, t, r) if not rev else (t, h, r)

        dom = set(_gen(self.nx, heads))
        rng = set(_gen(self.rnx, tails, rev=True))
        rel = {(h, t, r) for r in edges or [] for h, t in self.edges[r]}

        return dom | rng | rel

    #
    # --- | EXTERNAL SOURCES
    #

    def add(self, source: GraphImport) -> "Graph":
        """

        Add data to the current graph by using a GraphImport instance

        Parameters
        ----------

        source : GraphImport
          Data to feed into the graph

        """
        for i, (h, t, r) in enumerate(source.triples):
            self.nx.add_node(h, label=source.ents[h])
            self.nx.add_node(t, label=source.ents[t])
            self.nx.add_edge(h, t, r, label=source.rels[r], rid=r)
            self.edges[r].add((h, t))

        self.source.join(source)
        self._rnx = self.nx.reverse()
        return self

    def save(self, path: Union[str, pathlib.Path]) -> "Graph":
        """

        Persist graph to file.

        Parameters
        ----------

        path : Union[str, pathlib.Path]
          Folder to save the graph to

        """
        path = helper.path(
            path,
            create=True,
            message=f"saving {self.name} to {{path_abbrv}}",
        )

        kwargs = dict(name=self.name)
        with (path / "config.yml").open(mode="w") as fd:
            yaml.dump(kwargs, fd)

        self.source.save(path)

    @classmethod
    def load(K, path: Union[str, pathlib.Path]) -> "Graph":
        """

        Load graph from file

        Parameters
        ----------

        path : Union[str, pathlib.Path]
          File to load graph from

        """
        path = helper.path(
            path,
            exists=True,
            message="loading graph from {path_abbrv}",
        )

        with (path / "config.yml").open(mode="r") as fd:
            kwargs = yaml.load(fd, Loader=yaml.FullLoader)

        source = GraphImport.load(path)
        return K(source=source, **kwargs)

    #
    # ---| SUGAR
    #

    def tabulate_triples(self, triples: Iterable[Triple]):
        from tabulate import tabulate

        src = self.source

        rows = [(h, src.ents[h], t, src.ents[t], r, src.rels[r]) for h, t, r in triples]

        return tabulate(rows, headers=("", "head", "", "tail", "", "relation"))

    def str_triple(self, triple: Triple):
        h, t, r = triple
        return (
            f"{self.source.ents[h]} | "
            f"{self.source.ents[t]} | "
            f"{self.source.rels[r]}"
        )
