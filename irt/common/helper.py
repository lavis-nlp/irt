# -*- coding: utf-8 -*-

from irt import IRTError

import git
import logging
import numpy as np

import random
import pathlib

from typing import Union
from typing import Optional


log = logging.getLogger(__name__)


def path(
    name: Union[str, pathlib.Path],
    create: Optional[bool] = False,
    exists: Optional[bool] = False,
    is_file: Optional[bool] = False,
    message: Optional[str] = None,
) -> pathlib.Path:
    """

    Quickly create and check pathlib.Paths

    Parameters
    ----------

    name : Union[str, pathlib.Path]
      The target file or directory

    create : bool
      Create as directory

    exists : bool
      Checks whether the file or directory exists

    is_file : bool
      Checks whether the target is a file

    message : str
      Log a message for successful invocations


    """
    path = pathlib.Path(name)

    if (exists or is_file) and not path.exists():
        raise IRTError(f"{path} does not exist")

    if is_file and not path.is_file():
        raise IRTError(f"{path} exists but is not a file")

    if create:
        path.mkdir(exist_ok=True, parents=True)

    if message:
        path_abbrv = f"{path.parent.name}/{path.name}"
        log.info(message.format(path=path, path_abbrv=path_abbrv))

    return path


def seed(seed: int) -> np.random.Generator:
    log.info(f"! setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed=seed)
    # torch.manual_seed(seed=seed)
    return np.random.default_rng(seed)


def git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    # dirty = '-dirty' if repo.is_dirty else ''
    return str(repo.head.object.hexsha)
