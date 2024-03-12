#!/usr/bin/env python3
# -*- coding: utf-8 -*

from typing import NoReturn

from .scheduler import AbstractScheduler


class LocalScheduler(AbstractScheduler):
    """Local scheduler."""

    name: str = "local"

    @AbstractScheduler.job_name.setter
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        return

    def submit(self) -> NoReturn:
        """No submit is performed."""

        return

    def is_finished(self) -> bool:
        """Check if the job were finished.

        Returns:
            Always return true.

        """

        return True


if __name__ == "__main__":
    ...
