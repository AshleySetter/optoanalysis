"""
optoanalysis
============

Package of functions for the Matter-Wave Interferometry 
group for handling experimental data.


"""

# init file

import os

_mypackage_root_dir = os.path.dirname(__file__)
_version_file = open(os.path.join(_mypackage_root_dir, 'VERSION'))
__version__ = _version_file.read().strip()

# the following line imports all the functions from optoanalysis.py
from .optoanalysis import *
import optoanalysis.thermo
import optoanalysis.LeCroy
import optoanalysis.Saleae

