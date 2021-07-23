# -*- coding: utf-8 -*-
# fmt: off

import pathlib
import logging


log = logging.getLogger(__name__)


# always look relative from the project's root directory
_root_path = pathlib.Path(__file__).parent.parent
_DATA_DIR = 'data'


# fmt: off
class ENV:

    ROOT_DIR:    pathlib.Path = _root_path
    SRC_DIR:     pathlib.Path = _root_path / 'irt'
    LIB_DIR:     pathlib.Path = _root_path / 'lib'
    CONF_DIR:    pathlib.Path = _root_path / 'conf'
    DATA_DIR:    pathlib.Path = _root_path / _DATA_DIR
    DATASET_DIR: pathlib.Path = _root_path / _DATA_DIR / 'irt'
    SRC_DIR:     pathlib.Path = _root_path / _DATA_DIR / 'src'
    CACHE_DIR:   pathlib.Path = _root_path / _DATA_DIR / 'cache'
# fmt: on


class IRTError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


# -- register classes for convient access

from irt.data import dataset  # noqa: E402

# Vanilla Python
Dataset = dataset.Dataset

try:
    from irt.data import pykeen  # noqa: E402

    # Knowledge Graph Completion with PyKeen
    KeenClosedWorld = pykeen.KeenClosedWorld
    KeenOpenWorld = pykeen.KeenOpenWorld
except ModuleNotFoundError as err:
    log.warning(f"cannot import pykeen datasets: {err}")


try:
    from irt.data import pytorch  # noqa: E402

    # Training data for pytorch
    TorchDataset = pytorch.TorchDataset
    TorchLoader = pytorch.TorchLoader
    TorchModule = pytorch.TorchModule

except ModuleNotFoundError as err:
    log.warning(f"cannot import pytorch datasets: {err}")
