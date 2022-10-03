#!/usr/bin/env python3
# -*- coding: utf-8 -*

from typing import NoReturn

from GDPy.scheduler.scheduler import AbstractScheduler


class LocalScheduler(AbstractScheduler):

    """Local scheduler.

    """

    name: str = "local"

    def submit(self) -> NoReturn:
        """No submit is performed.
        """

        return
    
    def is_finished(self) -> bool:
        """ Check if the job were finished.

        Returns:
            Always return true.
    
        """

        return True


if __name__ == "__main__":
    pass
