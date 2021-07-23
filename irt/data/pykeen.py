# -*- coding: utf-8 -*-

"""

IRT dataset wrapper for use by pykeen

https://github.com/pykeen/pykeen

"""


from irt.data import dataset
from irt.common import helper

import logging
import textwrap
from abc import ABC
from abc import abstractmethod
from functools import lru_cache

import torch
from pykeen.triples import TriplesFactory

from typing import Union
from typing import Optional
from typing import Sequence
from typing import Collection


log = logging.getLogger(__name__)


class Keen(ABC):
    """

    Dataset as required by pykeen

    Using the same split.Dataset must always result in
    exactly the same TriplesFactories configuration

    """

    dataset: dataset.Dataset
    irt2keen: dict[int, int]

    # ---

    @property
    @abstractmethod
    def factories(self) -> list[tuple[str, TriplesFactory]]:
        raise NotImplementedError()

    @property
    @lru_cache
    def keen2irt(self) -> dict[int, int]:
        return {v: k for k, v in self.irt2keen.items()}

    def __str__(self) -> str:
        return f"keen dataset: [{self.dataset.name}]: " + (
            " | ".join(
                f"{name}={factory.num_triples}" for name, factory in self.factories
            )
        )

    @property
    def description(self) -> str:
        s = "IRT PYKEEN DATASET\n"
        s += f"{self.dataset.name}\n"

        for name, factory in self.factories:
            content = textwrap.indent(
                f"entities: {factory.num_entities}\n"
                f"relations: {factory.num_relations}\n"
                f"triples: {factory.num_triples}\n"
                "",
                "  ",
            )

            s += textwrap.indent(f"\n{name} triples factory:\n{content}", "  ")

        return s

    # ---

    def __init__(self, dataset: dataset.Dataset):
        self.dataset = dataset


def remap(*entities):
    assert not len(set.intersection(*entities)) if len(entities) > 1 else True

    flat = enumerate(e for subset in entities for e in subset)
    idmap = {old: new for new, old in flat}

    return idmap


def triples2factory(triples: Collection[Sequence[int]], idmap: dict[int, int]):
    """

    Convert htr triples to a pykeen triples factory

    """

    mapped = list((idmap[h], idmap[t], r) for h, t, r in triples)
    htr = torch.Tensor(mapped).to(dtype=torch.long)

    # move from htr to hrt
    hrt = htr[:, (0, 2, 1)]
    factory = TriplesFactory.create(hrt)

    return factory


class KeenClosedWorld(Keen):
    """

    Create pykeen dataset for closed world KGC

    The necessary pykeen triple factories are created and some
    constraints are checked (see self.check). The closed world triples
    of the IRT Dataset is partitioned randomly into training,
    validation, and optionally testing.

    Entities must have consecutive ids assigned. Each Keen dataset
    comes with self.irt2keen : dict[int, int] which maps the ids.

    """

    seed: int
    split: Union[float, Sequence[float]]

    training: TriplesFactory
    validation: TriplesFactory
    testing: Optional[TriplesFactory]

    def __init__(
        self,
        dataset: dataset.Dataset,
        split: Union[float, Sequence[float]],
        seed: int,
        *args,
        **kwargs,
    ):
        log.info("creating triple factories")

        self.seed = seed
        self.split = split
        self.irt2keen = remap(dataset.split.closed_world.owe)

        super().__init__(dataset=dataset, *args, **kwargs)
        helper.seed(seed)

        # create splits
        factory = triples2factory(
            triples=dataset.split.closed_world.triples,
            idmap=self.irt2keen,
        )

        splits = factory.split(ratios=split, random_state=seed)
        assert len(splits) in (2, 3), "invalid split configuration"

        self.training, self.validation, *_ = splits
        self.testing = splits[2] if len(splits) == 3 else None

        log.info(f"initialized triples split with ratio {split}")

    @property
    def factories(self):
        factories = [
            ("training", self.training),
            ("validation", self.validation),
        ]

        if self.testing:
            factories.append(("testing", self.testing))

        return factories


class KeenOpenWorld(Keen):
    """

    Create a pykeen dataset for open world KGC

    """

    closed_world: TriplesFactory
    open_world_valid: TriplesFactory
    open_world_test: TriplesFactory

    def __init__(self, *args, dataset: dataset.Dataset, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)

        self.irt2keen = remap(
            dataset.split.closed_world.owe,
            dataset.split.open_world_valid.owe,
            dataset.split.open_world_test.owe,
        )

        self.closed_world = triples2factory(
            triples=dataset.split.closed_world.triples,
            idmap=self.irt2keen,
        )
        self.open_world_valid = triples2factory(
            triples=dataset.split.open_world_valid.triples,
            idmap=self.irt2keen,
        )
        self.open_world_test = triples2factory(
            triples=dataset.split.open_world_test.triples,
            idmap=self.irt2keen,
        )

    @property
    def factories(self):
        return [
            ("closed world", self.closed_world),
            ("open world validation", self.open_world_valid),
            ("open world testing", self.open_world_test),
        ]
