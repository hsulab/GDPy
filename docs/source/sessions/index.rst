.. _sessions:

Sessions
========

Related Commands
----------------

.. code-block:: shell

    # - run a session defined in the file `./config.yaml`
    $ gdp session ./config.yaml

    # - run a session defined in the file `./config.yaml`
    #   fill in the placeholders in config by `--feed`
    $ gdp session ./config.yaml --feed temperatures=50,150,300


Variables
---------

scheduler.interface module
--------------------------

.. automodule:: scheduler.interface
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

.. +--------+---------+
.. | build  | modify  |
.. +--------+---------+
.. | select | drive   |
.. +--------+---------+
