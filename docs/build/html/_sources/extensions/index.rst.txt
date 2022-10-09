Extensions
==========

This section is about how to extend GDPy with custom python files.

Custom Potential
----------------

First we define a class named ``EmtManager`` that is a subclass of ``AbstractPotentialManager``
in ``emt.py``. We need to implement two attributes (``implemented_backends`` and ``valid_combinations``) 
and one method (``register_calculator``). Here, we only implement one backend that uses built-in EMT calculator 
in **ase**.

.. code-block:: python3

    #!/usr/bin/env python3
    # -*- coding: utf-8 -*

    from ase.calculators.emt import EMT

    from GDPy.potential.manager import AbstractPotentialManager

    class EmtManager(AbstractPotentialManager):

        name = "emt"
        implemented_backends = ["ase"]

        valid_combinations = [
            ["ase", "ase"]
        ]


        def register_calculator(self, calc_params, *args, **kwargs):
            super().register_calculator(calc_params)

            if self.calc_backend == "ase":
                calc = EMT()

            self.calc = calc

            return

    if __name__ == "__main__":
        pass

Then we can use EMT through ``pot.yaml``.

.. code-block:: yaml

    potential:
        name: ./emt.py # lowercase
        params:
            backend: ase
    driver:
        backend: external
        task: min
        run: 
            fmax: 0.05
            steps: 10

At last, we optimise a **H2O** molecule with **EMT**. The results are stored in the directory **cand0**.

.. code-block:: shell

    $ gdp driver ./pot.yaml -s H2O
    nframes:  1
    potter:  emt
    *** run-driver time:   0.1517 ***
    [1.8792752663147125]

