==================
Heat Transfer (ht)
==================

.. image:: http://img.shields.io/pypi/v/ht.svg?style=flat
   :target: https://pypi.python.org/pypi/ht
   :alt: Version_status
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://ht.readthedocs.io/en/latest/
   :alt: Documentation
.. image:: https://github.com/CalebBell/ht/workflows/Build/badge.svg
   :target: https://github.com/CalebBell/ht/actions
   :alt: Build_status
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat 
   :target: https://github.com/CalebBell/ht/blob/release/LICENSE.txt
   :alt: license
.. image:: https://img.shields.io/pypi/pyversions/ht.svg?
   :target: https://pypi.python.org/pypi/ht
   :alt: Supported_versions
.. image:: http://img.shields.io/appveyor/ci/calebbell/ht.svg?
   :target: https://ci.appveyor.com/project/calebbell/ht/branch/release
   :alt: Build_status
.. image:: https://zenodo.org/badge/48963057.svg?
   :alt: Zendo
   :target: https://zenodo.org/badge/latestdoi/48963057


.. contents::

What is ht?
-----------

ht is open-source software for engineers and technicians working in the
fields of chemical or mechanical engineering. It includes modules
for various heat transfer functions.

Among the tasks this library can be used for are:

* Sizing a Shell & Tube heat exchanger using any of the Zukauskas, ESDU 73031, or Bell methods
* Calculating pressure drop in a Hairpin heat exchanger
* Calculating heat loss of objects, including insulated objects
* Calculating heat loss from buried pipe
* Performing radiative heat transfer calculations
* Conderser and Reboiler rating
* Detailed heat exchanger evaluation; finding fouling factors
* Heat transfer in packed beds
* Sizing a Plate and Frame heat exchanger
* Modeling an Air Cooler
* Supercritical CO2 or water heat transfer

The ht library depends on the SciPy library to provide numerical constants,
interpolation, integration, and numerical solving functionality. ht runs on
all operating systems which support Python, is quick to install, and is free
of charge. ht is designed to be easy to use while still providing powerful
functionality. If you need to perform some heat transfer calculations, give
ht a try.

Installation
------------

Get the latest version of ht from
https://pypi.python.org/pypi/ht/

If you have an installation of Python with pip, simple install it with:

    $ pip install ht

Alternatively, if you are using `conda <https://conda.io/en/latest/>`_ as your package management, you can simply
install ht in your environment from `conda-forge <https://conda-forge.org/>`_ channel with:

    $ conda install -c conda-forge ht

To get the git version, run:

    $ git clone git://github.com/CalebBell/ht.git

Documentation
-------------

ht's documentation is available on the web:

    https://ht.readthedocs.io/en/latest/index.html


Latest source code
------------------

The latest development version of ht's sources can be obtained at

    https://github.com/CalebBell/ht


Bug reports
-----------

To report bugs, please use the ht's Bug Tracker at:

    https://github.com/CalebBell/ht/issues


License information
-------------------

ht is MIT licensed. See ``LICENSE.txt`` for information on the terms & 
conditions for usage of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the ht license, if it is convenient for you,
please cite ht if used in your work. Please also consider contributing
any changes you make back, such that they may be incorporated into the
main library and all of us will benefit from them.


Citation
--------

To cite ht in publications use::

    Caleb Bell (2016-2021). ht: Heat transfer component of Chemical Engineering Design Library (ChEDL)
    https://github.com/CalebBell/ht.
