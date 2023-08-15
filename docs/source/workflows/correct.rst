Add Correction to Computed Structures
=====================================

Use `gdp session` to run a worflow add energy/forces correction to computed 
structures. The `correct` operation needs two input variables and forwards a 
`Tempdata` variable (See Dataset section for more details).

For the input variables,

- structures: A `Tempdata` variable.

- computer: A `computer` variable.

The example below adds `DFT-D3` correction to a dataset of a H2O molecule. The output 
cache is saved `./_corr/0002.corr_dftd3/merged.xyz`. The structures have energy/forces 
equal `origin+dftd3`.

Example Configuration
---------------------

.. code-block:: yaml

    variables:
      dataset:
        type: tempdata
        system_dirs:
          - ./min-H2O-molecule
      # ---
      spc_dftd3:
        type: computer
        potter:
          name: dftd3
          params:
            backend: ase
            method: PBE # xc
            damping: d3bj
    operations:
      corr_dftd3:
        type: correct
        structures: ${vx:dataset}
        computer: ${vx:spc_dftd3}
    sessions:
      _corr: corr_dftd3

.. note::

    `DFT-D3` computation requires python packge `dftd3-python`. 
    Use `conda install dftd3-python -c conda-forge` if one does not have it.
