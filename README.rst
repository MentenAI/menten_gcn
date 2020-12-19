==================
Technical Overview
==================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        |
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/menten_gcn/badge/?style=flat
    :target: https://readthedocs.org/projects/menten_gcn
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/MentenAI/menten_gcn.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/MentenAI/menten_gcn

.. |commits-since| image:: https://img.shields.io/github/commits-since/MentenAI/menten_gcn/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/MentenAI/menten_gcn/compare/v0.0.0...main



.. end-badges

This package decorates graph tensors with data from protein models

* Free software: MIT license

Installation
============

::

    pip install menten-gcn

You can also install the in-development version with::

    pip install https://github.com/MentenAI/menten_gcn/archive/master.zip


Documentation
=============


https://menten_gcn.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
