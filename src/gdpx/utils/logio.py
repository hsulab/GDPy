#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging


def remove_extra_stream_handlers():
    """Remove extra stream handlers added by imported packages."""
    # some imported packages change `logging.basicConfig`
    # and accidently add a StreamHandler to logging.root
    # so remove it...
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            logging.root.removeHandler(h)

    return


if __name__ == "__main__":
    ...
