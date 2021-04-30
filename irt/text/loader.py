# -*- coding: utf-8 -*-


import irt
from irt.common import helper

import json
import sqlite3
import pathlib
from tqdm import tqdm as _tqdm

import logging
from functools import partial
from functools import lru_cache
from dataclasses import dataclass

from typing import Optional


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


@dataclass
class Result:

    mentions: tuple[str]
    blobs: tuple[str]


class Selector:
    def by_entity(self, entity: int) -> Optional[Result]:
        raise NotImplementedError()


class Loader:
    def __enter__(self) -> Selector:
        raise NotImplementedError()

    def __exit__(self, *_):
        raise NotImplementedError()


# JSON --------------------


class JSON(Loader):
    """

    Load a json


    The JSON file needs to look like this:

    "<ENTITY>": {
        "description": "<DESCRIPTION"
    }

    And also it is required to provide an id mapping:
    { "<ENTITY>": <ID> }

    """

    class JSONSelector(Selector):
        def __init__(self, id2desc: dict[int, list[str]], id2ent: dict[int, str]):
            self.id2desc = id2desc
            self.id2ent = id2ent

        def by_entity(self, e: int) -> Optional[Result]:
            blobs = self.id2desc[e]

            if not blobs:
                # log.error(f"{e=} {self.id2ent[e]} {blobs=}")
                return None

            mentions = [self.id2ent[e] for _ in blobs]

            return Result(
                mentions=mentions,
                blobs=blobs,
            )

    # ---

    @staticmethod
    def db_name(fname: str = None, **kwargs):
        return fname

    def __init__(
        self,
        fname: str,
        # from ryn: graph.source.ents
        id2ent: dict[int, str],
        # (optionally) maps mid -> idx
        idmap: Optional[str],
    ):
        log.info(f"loading {fname=} with {len(id2ent)} mapped entities")
        self.id2ent = id2ent

        idmap = helper.path(
            idmap, exists=True, message="loading idmap from {path_abbrv}"
        )

        # translate mids to idxs
        with idmap.open(mode="r") as fd:
            fd.readline()  # consume first line with count
            idmap = dict(
                (label, int(idx)) for label, idx in map(str.split, fd.readlines())
            )

        path = helper.path(fname, exists=True, message="loading {path_abbrv}")
        with path.open(mode="r") as fd:
            raw = json.load(fd)

        self.id2desc = {}
        for entity, data in raw.items():
            description = data["description"]
            e = idmap[entity]
            self.id2desc[e] = (
                None if not description else [description.capitalize() + "."]
            )

        empty = {e: None for e in id2ent if e not in self.id2desc}
        log.error(f"no description for {len(empty)} in {path.name}")
        self.id2desc.update(empty)

        log.error(
            f"no description in total for"
            f" {len([v for v in self.id2desc.values() if not v])}"
        )

        assert all(ent in self.id2desc for ent in id2ent)

    def __enter__(self):
        return JSON.JSONSelector(id2desc=self.id2desc, id2ent=self.id2ent)

    def __exit__(self, *_):
        pass


# SQLITE --------------------


class SQLite(Loader):
    """

    Load text from a sqlite database

    Schema must be like this (v4+):

    TABLE contexts:
        entity INT,
        entity_label TEXT,
        page_title TEXT,
        context TEXT,

    "entity_label" is the mention

    """

    DB_NAME = "contexts"

    # ---

    COL_ID: int = "id"
    COL_ENTITY: int = "entity"
    COL_LABEL: str = "entity_label"
    COL_MENTION: str = "mention"
    COL_CONTEXT: str = "context"

    # ---

    @dataclass
    class SQLSelector(Selector):

        conn: sqlite3.Connection
        cursor: sqlite3.Cursor

        def by_entity(self, e: int):
            query = (
                "SELECT "
                f"{SQLite.COL_MENTION}, "
                f"{SQLite.COL_CONTEXT} "
                f"FROM {SQLite.DB_NAME} "
                f"WHERE {SQLite.COL_ENTITY}=?"
            )

            self.cursor.execute(query, (e,))
            result = self.cursor.fetchall()
            if not result:
                return None

            result.sort()  # otherwise no guarantee
            mentions, blobs = zip(*result)

            return Result(
                mentions=mentions,
                blobs=blobs,
            )

    # ---

    @staticmethod
    def db_name(database: str = None, **kwargs):
        return helper.path(database).name

    def __init__(self, *, database: str = None, to_memory: bool = False):
        log.info(f"connecting to database {database}")
        self._database = database

        if to_memory:
            log.info("copying database to memory")
            self._conn = sqlite3.connect(":memory:")
            self._cursor = self._conn.cursor()

            log.info(f"opening {database}")
            with sqlite3.connect(database) as con:
                for sql in con.iterdump():
                    self._cursor.execute(sql)

        else:
            log.info("accessing database from disk")
            self._conn = sqlite3.connect(database)
            self._cursor = self._conn.cursor()

    def __enter__(self):
        return SQLite.SQLSelector(conn=self._conn, cursor=self._cursor)

    def __exit__(self, *_):
        self._conn.close()


# CoDEx --------------------


class CoDEx(Loader):
    """

    Load CoDEx texts

    """

    class CoDExSelector(Selector):
        def __init__(
            self, *, path: pathlib.Path = None, ent2wiki: dict[int, str] = None
        ):
            from nltk.tokenize import sent_tokenize

            self.path = path
            self.ent2wiki = ent2wiki
            self.tokenize = sent_tokenize

        @lru_cache
        def by_entity(self, e: int) -> Optional[Result]:
            wiki, name = self.ent2wiki[e]

            try:
                textfile = helper.path(self.path / (wiki + ".txt"), exists=True)
                with textfile.open(mode="r") as fd:
                    raw = fd.read()

            except irt.IRTError:
                log.error(f"did not find {e=}: {self.ent2wiki[e]}")
                return None

            contexts = "\n".join(self.tokenize(raw))

            return Result(
                mentions=(name,),
                blobs=(contexts,),
            )

    # ---

    @staticmethod
    def db_name(path: str = None, **kwargs):
        path = helper.path(path)
        return f"codex.{path.parts[-2]}"

    def __init__(
        self,
        # path to folder with <ID>.txt files
        path: str,
        # from irt: graph.source.ents
        id2ent: dict[int, str] = None,
    ):
        log.info(f"loading {len(id2ent)} mapped entities")

        self._path = helper.path(path, exists=True)
        self._ent2wiki = {
            ent: (wiki, name)
            for ent, wiki, name in (
                [e] + s.split(":", maxsplit=1) for e, s in id2ent.items()
            )
        }

    def __enter__(self):
        return CoDEx.CoDExSelector(path=self._path, ent2wiki=self._ent2wiki)

    def __exit__(self, *_):
        pass
