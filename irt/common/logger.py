# -*- coding: utf-8 -*-

import irt

import os
import pathlib
import logging
import configparser

from logging.config import fileConfig


# pass these environment variables to either
# change where the configuration file is expected
ENV_IRT_LOG_CONF = "IRT_LOG_CONF"
# or where the log file is written
ENV_IRT_LOG_OUT = "IRT_LOG_OUT"


def init():
    # probe environment for logging configuration:
    #   1. if conf/logging.conf exists use this
    #   2. if IRT_LOG_CONF is set as environment variable use its value
    #      as path to logging configuration

    fconf = None

    if ENV_IRT_LOG_CONF in os.environ:
        fconf = str(os.environ[ENV_IRT_LOG_CONF])

    else:
        path = pathlib.Path(irt.ENV.CONF_DIR / "logging.conf")

        if path.is_file():
            cp = configparser.ConfigParser()
            cp.read(path)

            if "handler_fileHandler" in cp:
                opt = cp["handler_fileHandler"]
                (fname,) = eval(opt["args"])

                if ENV_IRT_LOG_OUT in os.environ:
                    fname = pathlib.Path(os.environ[ENV_IRT_LOG_OUT])
                else:
                    fname = irt.ENV.ROOT_DIR / fname

                fname.parent.mkdir(exist_ok=True, parents=True)
                fname.touch(exist_ok=True)
                opt["args"] = repr((str(fname),))

                fconf = cp

    if fconf is not None:
        fileConfig(cp)

    log = logging.getLogger(__name__)
    log.info("initialized logging")


# be nice if used as a library - do not log to stderr as default
log = logging.getLogger("irt")
log.addHandler(logging.NullHandler())
