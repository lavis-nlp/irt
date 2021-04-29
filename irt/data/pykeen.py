# -*- coding: utf-8 -*-


from irt.data import dataset
from irt.common import helper

import logging
import textwrap
from abc import ABC
from abc import abstractmethod

import torch
from pykeen.triples import TriplesFactory

from typing import Dict
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

    # # ID MAPPING

    # # this maps the IRT ids to the automatically
    # # assigned ids of the pykeen triples factories
    # irt2keen: Dict[int, int]

    # # id-mapping rests upon prefixing the entity and
    # # relation labels with the IRT id and extracting
    # # those again after pykeen assigned its own ids

    # def e2s(self, e: int):
    #     return f"{e}:{self.dataset.id2ent[e]}"

    # def r2s(self, r: int):
    #     return f"{r}:{self.dataset.id2rel[r]}"

    # def s2id(self, s: str):
    #     return int(s.split(":", maxsplit=1)[0])

    # def triple_to_str(self, h: int, t: int, r: int):
    #     """

    #     Transform a irtm.graphs triple to a pykeen string representation

    #     Parameters
    #     ----------

    #     h: int
    #       head id

    #     t: int
    #       tail id

    #     r: int
    #       relation id

    #     Returns
    #     -------

    #     htr: Tuple[str, str, str]
    #       String representation of (h, t, r) prefixed with irt ids

    #     """
    #     return self.e2s(h), self.e2s(t), self.r2s(r)

    # def triples_to_ndarray(self, triples: Collection[Tuple[int]]):
    #     """

    #     Transform htr triples to ndarray of hrt string rows

    #     Parameters
    #     ----------

    #     htr: Collection[Tuple[int]]
    #       irtm graph triples

    #     Returns
    #     -------

    #     Numpy array of shape [N, 3] containing triples as
    #     strings of form hrt.

    #     """

    #     # transform triples to ndarray and re-arrange
    #     # triple columns from (h, t, r) to (h, r, t)
    #     return np.array(list(map(self.triple_to_str, triples)))[:, (0, 2, 1)]

    # ---

    @property
    @abstractmethod
    def factories(self) -> Dict[str, TriplesFactory]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"keen dataset: [{self.dataset.name}]: " + (
            " | ".join(
                f"{name}={factory.num_triples}" for name, factory in self._factories
            )
        )

    @property
    def description(self) -> str:
        s = "IRT PYKEEN DATASET\n"
        s += f"{self.dataset.name}\n"

        for name, factory in self.factories.items():
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


def triples2factory(triples: Collection[Sequence[int]]):
    """

    Convert htr triples to a pykeen triples factory

    """

    triples = list(triples)
    htr = torch.Tensor(triples).to(dtype=torch.long)

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

        super().__init__(dataset=dataset, *args, **kwargs)
        helper.seed(seed)

        # create splits
        factory = triples2factory(dataset.split.closed_world.triples)
        splits = factory.split(ratios=split, random_state=seed)
        assert len(splits) in (2, 3), "invalid split configuration"

        self.training, self.validation, *_ = splits
        if len(splits) == 3:
            self.testing = splits[2]

        log.info(f"initialized triples split with ratio {split}")

    @property
    def factories(self):
        factories = {
            "training": self.training,
            "validation": self.validation,
        }

        if self.testing:
            factories["testing"] = self.testing

        return factories


class KeenOpenWorld(Keen):
    """

    Create a pykeen dataset for open world KGC

    """

    closed_world: TriplesFactory
    open_world_valid: TriplesFactory
    open_world_testing: Optional[TriplesFactory]

    def __init__(self, *args, dataset: dataset.Dataset, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)

        self.closed_world = triples2factory(
            dataset.split.closed_world.triples,
        )
        self.open_world_valid = triples2factory(
            dataset.split.open_world_valid.triples,
        )
        self.open_world_testing = triples2factory(
            dataset.split.open_world_testing.triples,
        )

    @property
    def factories(self):
        return {
            "closed world": self.closed_world,
            "open world validation": self.open_world_valid,
            "open world testing": self.open_world_testing,
        }
