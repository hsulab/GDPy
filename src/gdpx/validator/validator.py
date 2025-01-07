#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import pathlib
from typing import Optional, Union

from gdpx.core.component import BaseComponent


class AbstractValidator(BaseComponent):

    def __init__(
        self,
        directory: Union[str, pathlib.Path] = "./",
        random_seed: Optional[Union[int, dict]] = None,
    ) -> None:
        """ """
        super().__init__(directory=directory, random_seed=random_seed)

        return

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return


if __name__ == "__main__":
    ...

