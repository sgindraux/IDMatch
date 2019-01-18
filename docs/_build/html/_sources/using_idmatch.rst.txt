.. _label-usage:

Using IDMatch
*************

Install python and set environement
===================================

The credit of the information and most of the text below goes to the developers of OGGM (website) who wrote a brilliant documentation on how to intstall their software.

IDMatch is a python package containing several dependencies. The instructions below are should work on any platform.
IDMatch  is fully tested with python version 3.6 on Windows and doesn’t work with python version 2.7.
We recommend to install python and the package dependencies with a recent version of the conda package manager.
You can get conda by installing miniconda (the package manager alone - recommended) or anaconda (the full suite - with many packages you wont need).

Once python has been installed, we recommend to create a specific environment for IDMatch. In a terminal window, type:

conda create --name idmatch_env python=3.X

where 3.X is the python version shipped with conda (currently 3.6). You can of course use any other name for your environment.

Don’t forget to activate it before going on:

activate idmatch_env


Install packages
----------------
Install the packages from the conda-forge and oggm channels:

.. note:: conda install -c oggm -c conda-forge oggm-deps

The oggm-deps package is a “meta package”. It does not contain any code but will insall all the packages oggm needs automatically.

If you are using conda, you can install OGGM as a normal conda package:

.. note:: conda install -c oggm -c conda-forge oggm

If you are using pip, you can install OGGM from PyPI:

.. note:: pip install oggm


.. _label-getthesoftware:

Get the software from GitHub
----------------------------

If you want to explore the code or participate to its development, we recommend to clone the git repository (or your own fork , see also Contributing to OGGM): You should have a recent version of git

.. note:: git clone https://github.com/OGGM/oggm.git

Then go to the project root directory:

.. note:: cd idmatch

That's it! You are now set-up for the :ref:`label-quickstart` or the :ref:`label-normalstart` sections.


.. _label-quickstart:

Quick start
===========

You want to make a quick test on your dataset (or on the one we provide) without reading the documentation; you are in the right section.

Prerequisite: you have Python 3.X and IDMatch installed on your computer.

Workflow:
1. Access the main IDMatch folder, and the idmatch subfolder. There are four files inside:

 * __init__.py
 * core.py
 * functions.py
 * params.py.

2. Open the params.py file. Choose the mode you want (mode=1 is the image matching, mode=2 is the DSM matching) and fill the corresponding paths and variables.
    If you just want to make IDMatch run with the test dataset we provide, leave the params.py file as it is.

    .. important:: The processing time depends on many things: The computer power you have, the resolution of your dataset, the number of matching methods (M1,M2,...) and windows (WX) you want to test, as well as the grid_step value.
                    The dataset we provide as well as the default setting in the params.py file should take about Xh to process.
3. Run the core.py



.. _label-normalstart:

Normal start
============


.. _architecture:

.. image:: /images/architecture.png
   :align:   center

   This is the caption of that image


This picture :ref:'_architecture' shows IDMatch's architecture.
