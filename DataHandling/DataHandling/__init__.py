"""
DataHandling
============

Package of functions for the Matter-Wave Interferometry
 group for handling experimental data.


"""

# init file

# the following 8 lines set the __version__ variable
import pkg_resources as _pkg_resources

resource_package = __name__  # Gets the module/package name.
rawVersionFile = _pkg_resources.resource_string(
            resource_package, "VERSION")
decodedVersionFile = rawVersionFile.decode(encoding='UTF-8')
# decocdes the bytes object into a string object
__version__ = decodedVersionFile.splitlines()[0]

# the following line imports all the functions from DataHandling.py
from .DataHandling import *
