#!/usr/bin/env python3
# -*- coding: utf-8 -*


from .scheduler import BaseScheduler


class LocalScheduler(BaseScheduler):
    """Local scheduler."""

    name: str = "local"

    @BaseScheduler.job_name.setter
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        return

    def is_finished(self) -> bool:
        """Check if the job were finished.

        Returns:
            Always return true.

        """

        return True


if __name__ == "__main__":
    ...
