from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# the following 8 lines set the __version__ variable
import pkg_resources as _pkg_resources

_resource_package = __name__  # Gets the module/package name.
_rawVersionFile = _pkg_resources.resource_string(
                _resource_package, "VERSION")
_decodedVersionFile = _rawVersionFile.decode(encoding='UTF-8')
# decocdes the bytes object into a string object
Version = _decodedVersionFile.splitlines()[0]
    
setup(name='datahandling',
      version=Version,
      description='Python package with functions for data analysis',
      author='Ashley Setter',
      author_email='A.Setter@soton.ac.uk',
      url=None,
      packages=['datahandling',
                'datahandling.LeCroy',
                'datahandling.SimData',
      ],
      install_requires=requirements,
)
