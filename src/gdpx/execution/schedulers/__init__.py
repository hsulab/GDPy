"""Create scheduler based on parameters

This module includes several schedulers.

Example:

    .. code-block:: python

        >>> from gdpx.execution.schedulers.direct import DirectScheduler
        >>> params = dict()
        >>> scheduler = DirectScheduler(**params)

"""

from gdpx.core.registry import Registry

REGISTER = Registry("scheduler")

from .scheduler import BaseScheduler

from .direct import DirectScheduler

REGISTER.register(DirectScheduler)

from .lsf import LsfScheduler

REGISTER.register(LsfScheduler)

from .pbs import PbsScheduler

REGISTER.register(PbsScheduler)

__all__ = [
    "REGISTER",
    "BaseScheduler",
    "DirectScheduler",
    "LsfScheduler",
    "PbsScheduler",
    "SshTransport",
    "SlurmScheduler",
]

from .slurm import SlurmScheduler

REGISTER.register(SlurmScheduler)

from .factory import canonicalise_scheduler

__all__.append("canonicalise_scheduler")


def __getattr__(name):
    if name == "SshTransport":
        from .remote import SshTransport

        return SshTransport
    raise AttributeError(name)
