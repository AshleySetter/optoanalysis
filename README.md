[![Build Status](https://travis-ci.org/AshleySetter/datahandling.png)](https://travis-ci.org/AshleySetter/datahandling)
[![codecov](https://codecov.io/gh/AshleySetter/datahandling/branch/master/graph/badge.svg)](https://codecov.io/gh/AshleySetter/datahandling)
[![Documentation Status](https://readthedocs.org/projects/datahandling/badge/?version=latest)](http://datahandling.readthedocs.org/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/datahandling.svg)](https://badge.fury.io/py/datahandling)

# optoanalysis
Matterwave group data analysis library.

This is the optoanalysis package developed primarily by [Ashley Setter](http://cmg.soton.ac.uk/people/ajs3g11/) of the [Quantum Nanophysics and Matter Wave Interferometry group](http://phyweb.phys.soton.ac.uk/matterwave/html/index.html) headed up by Prof. Hendrik Ulbricht at Southampton University in the UK.

The thermo module of this package was developed mainly by [Markus Rademacher](https://www.linkedin.com/in/markusrademacher/) of the University of Vienna in Austria, who works in the [group of Markus Aspelmeyer and Nikolai Kiesel](http://aspelmeyer.quantum.at/).

This library contains numerous functions for loading, analysing and plotting data produced from our optically levitated nanoparticle experiment. We use an optical tweezer setup to optically trap and levitate nanoparticles in intense laser light and measure the motion of these particles interferometrically from the light they scatter. This library provides all the tools to load up examples of this kind of data and analyse it. Currently data can be loaded from .trc or .raw binary files produced by Teledyne LeCroy oscilloscopes and .bin files produced from by Saleae data loggers. 

Some example usage of this library is shown in [this Jupyter notebook](Usage_Demonstration.ipynb).