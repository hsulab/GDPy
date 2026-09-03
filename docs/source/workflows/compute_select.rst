Compute and Select
==================

The ``compute`` operation consumes structures and one runtime, or an explicit
list of runtimes. The runtime already contains the potential, executor,
modifiers, and scheduler needed to execute the calculation. Its output can be
passed to ``extract`` and then to ``select``.

A typical graph is::

    structures -> compute(runtime) -> extract -> select

For a sequence such as pre-relaxation followed by a higher-accuracy
calculation, define a ``runtime_chain``. Do not express alternatives by placing
lists inside potential or executor fields; list the complete runtimes instead.

This explicit boundary ensures that every submitted task has exactly one
potential/executor pairing and can be serialized as schema version 2.
