#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import copy
import logging

from gdpx.core.component import BaseComponent
from gdpx.core.register import registers
from gdpx.factory.computer import canonicalise_worker


class BaseExpedition(BaseComponent):

    #: Name of the expedition.
    name: str = "expedition"

    @abc.abstractmethod
    def read_convergence(self) -> bool: ...

    @abc.abstractmethod
    def get_workers(self):

        return

    def run(self, *args, **kwargs) -> None:
        """"""
        # - some imported packages change `logging.basicConfig`
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.root.removeHandler(h)

        assert self.worker is not None, f"{self.name} has not set its worker properly."

        return

    def register_builder(self, builder: dict) -> None:
        """Register StructureBuilder for this expedition."""
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create("builder", builder_method, convert_name=False, **builder_params)
        else:
            builder = builder

        self.builder = builder

        return

    def register_worker(self, worker: dict, *args, **kwargs) -> None:
        """Register DriverBasedWorker for this expedition."""
        self.worker = canonicalise_worker(inp_worker=worker)

        return


if __name__ == "__main__":
    ...
