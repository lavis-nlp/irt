# -*- coding: utf-8 -*-


import enum


SEP = "|"

MASK_TOKEN = "[MASK]"
TOK_MENTION_START = "[MENTION_START]"
TOK_MENTION_END = "[MENTION_END]"


class Mode(enum.Enum):

    CLEAN = "clean"
    MARKED = "marked"
    MASKED = "masked"

    @staticmethod
    def filename(mode: "Mode"):
        return f"contexts.{mode.value}.txt.gz"
