Workflows
=========

Compute-Then-Select
-------------------

.. code-block:: yaml

    operations:
      build:
        type: build
        builders:
          - ${gdp_v:builder}
      scan:
        type: compute
        builder: ${gdp_o:build}
        worker: ${gdp_v:nvtw_computation}
        batchsize: 256
      extract:
        type: extract
        compute: ${gdp_o:scan}
      select_soap:
        type: select
        structure: ${gdp_o:extract}
        selector: ${gdp_v:desc_selector}
      select_hist:
        type: select
        structure: ${gdp_o:select_soap}
        selector: ${gdp_v:hist_selector}
      run_vasp:
        type: compute
        builder: ${gdp_o:select_hist}
        worker: ${gdp_v:vasp_computation}
        batchsize: 512
      extract_dft:
        type: extract
        compute: ${gdp_o:run_vasp}
      transfer:
        type: transfer
        structure: ${gdp_o:extract_dft}
        target_dir: /scratch/gpfs/jx1279/copper+alumina/dataset/s001
        version: r6_md
        system: surf

List of Workflows
-----------------

.. toctree::
    :maxdepth: 2

    validate.rst
