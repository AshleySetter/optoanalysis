[![DOI](https://zenodo.org/badge/74761875.svg)](https://zenodo.org/badge/latestdoi/74761875)
[![Build Status](https://travis-ci.org/AshleySetter/optoanalysis.png)](https://travis-ci.org/AshleySetter/optoanalysis)
[![codecov](https://codecov.io/gh/AshleySetter/optoanalysis/branch/master/graph/badge.svg)](https://codecov.io/gh/AshleySetter/optoanalysis)
[![Documentation Status](https://readthedocs.org/projects/optoanalysis/badge/?version=latest)](http://optoanalysis.readthedocs.org/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/optoanalysis.svg)](https://badge.fury.io/py/optoanalysis)

# optoanalysis
Matterwave group data analysis library.

This is the optoanalysis package developed primarily by [Ashley Setter](http://cmg.soton.ac.uk/people/ajs3g11/) of the [Quantum Nanophysics and Matter Wave Interferometry group](http://phyweb.phys.soton.ac.uk/matterwave/html/index.html) headed up by Prof. Hendrik Ulbricht at Southampton University in the UK.

The thermo module of this package was developed mainly by [Markus Rademacher](https://www.linkedin.com/in/markusrademacher/) of the University of Vienna in Austria, who works in the [group of Markus Aspelmeyer and Nikolai Kiesel](http://aspelmeyer.quantum.at/).

The package and all requirements can be installed by simply running the following command:
```
pip install optoanalysis
```

This library contains numerous functions for loading, analysing and plotting data produced from our optically levitated nanoparticle experiment. We use an optical tweezer setup to optically trap and levitate nanoparticles in intense laser light and measure the motion of these particles interferometrically from the light they scatter. This library provides all the tools to load up examples of this kind of data and analyse it. Currently data can be loaded from .trc or .raw binary files produced by Teledyne LeCroy oscilloscopes and .bin files produced from by Saleae data loggers. 

Some example usage of this library is shown in [this Jupyter notebook](Usage_Demonstration.ipynb).

If you use this package in any academic work it would be very appretiated if you could cite it [![DOI](https://zenodo.org/badge/74761875.svg)](https://zenodo.org/badge/latestdoi/74761875).

# Development

To build and install this package locally you can navigate to the optoanalysis subdirectory and run `python -m build`, this will build the package in the `optoanalysis/dist` directory relative to the root directory, inside will be a `optoanalysis-<your_version_here>-py3-none-any.whl` file for the version of the package you have built, you can change the version number by editing the version in the `optoanalysis/optoanalysis/VERSION` file. You can then install your local version by running `pip install optoanalysis-<your_version_here>-py3-none-any.whl` which will install it to your active python environment, if you have a current or newer version of the package already installed, you will need to add the `--force-reinstall` flags to force pip to reinstall the package. You can then import and use your local version of the package. You will need to rebuild and reinstall upon making any changes to the source code, and you will need to use `pip install optoanalysis-<your_version_here>-py3-none-any.whl --force-reinstall` to force pip to reinstall the same version of the package, or pip will ignore your changes.