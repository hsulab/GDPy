.. GDPy documentation master file, created by
   sphinx-quickstart on Mon Aug 22 14:06:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GDPy Documentation
============================================

GDPy stands for **Generating Deep Potential with Python**, including
a set of tools and Python modules to automate the structure exploration 
and the model training for **machine learning interatomic potentials** (MLIPs). 
It is developed and maintained by `Jiayan Xu`_ under supervision of Prof. `P. Hu`_
at Queen's University Belfast.

.. _Jiayan Xu: https://scholar.google.com/citations?user=ue5SBQMAAAAJ&hl=en
.. _P. Hu: https://scholar.google.com/citations?user=GNuXfeQAAAAJ&hl=en

.. figure:: ../../assets/logo.png
    :alt: GPDy LOGO
    :width: 400
    :align: center


Supported **Potentials**
------------------------

``eann``, ``deepmd``, ``lasp``, ``nequip`` / ``allegro``

Supported **Expeditions**
-------------------------

``molecular dynamics``, ``genetic algorithm``, ``grand canonical monte carlo``,
``graph-theory adsorbate configuration``, ``artificial force induced reaction``

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   about.rst
   installation.rst

.. toctree::
   :maxdepth: 2
   :caption: Basic Guides:

   start
   potentials/index
   trainers/index
   computations/index
   builders/index
   selections/index
   routines/index
   expeditions/index
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Guides:

   sessions/index
   workflows/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guides:

   extensions/index
.. modules/modules

.. toctree::
   :maxdepth: 2
   :caption: Gallery:

   applications/index


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
