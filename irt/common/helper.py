# -*- coding: utf-8 -*-

from irt import IRTError
from irt.common import logging

import pathlib

from typing import Union


log = logging.get("common.helper")


def path(
    name: Union[str, pathlib.Path],
    create: bool = False,
    exists: bool = False,
    is_file: bool = False,
    message: str = None,
) -> pathlib.Path:
    """

    Quickly create pathlib.Paths

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
    # TODO describe message (see kgc.config)
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
