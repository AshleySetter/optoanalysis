"""
datahandling
============

Package of functions for the Matter-Wave Interferometry
 group for handling experimental data.


"""

# init file

# the following 8 lines set the __version__ variable
import pkg_resources as _pkg_resources

_resource_package = __name__  # Gets the module/package name.
_rawVersionFile = _pkg_resources.resource_string(
            _resource_package, "VERSION")
_decodedVersionFile = _rawVersionFile.decode(encoding='UTF-8')
# decocdes the bytes object into a string object
__version__ = _decodedVersionFile.splitlines()[0]

# the following line imports all the functions from datahandling.py
from .datahandling import *
