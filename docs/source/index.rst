.. optoanalysis documentation master file, created by
   sphinx-quickstart on Fri Apr 21 00:06:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to optoanalysis's documentation!
========================================

This is the documentation for the optoanalysis package developed primarily by `Ashley Setter <http://cmg.soton.ac.uk/people/ajs3g11/>`_ of the `Quantum Nanophysics and Matter Wave Interferometry group <http://phyweb.phys.soton.ac.uk/matterwave/html/index.html>`_ headed up by Prof. Hendrik Ulbricht at Southampton University in the UK.

The thermo module of this package was developed mainly by `Markus Rademacher <https://www.linkedin.com/in/markusrademacher/>`_ of the University of Vienna in Austria, who works in the `group of Markus Aspelmeyer and Nikolai Kiesel <http://aspelmeyer.quantum.at/>`_.

This library contains numerous functions for loading, analysing and plotting data produced from our optically levitated nanoparticle experiment. We use an optical tweezer setup to optically trap and levitate nanoparticles in intense laser light and measure the motion of these particles interferometrically from the light they scatter. This library provides all the tools to load up examples of this kind of data and analyse it. Currently data can be loaded from .trc or .raw binary files produced by Teledyne LeCroy oscilloscopes and .bin files produced from by Saleae data loggers. 

If you use this package in any academic work it would be very appretiated if you could cite it.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 1
   :caption: Documentation:

   optoanalysis
   optoanalysis.sim_data
   optoanalysis.thermo
   optoanalysis.LeCroy
   optoanalysis.Saleae

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
