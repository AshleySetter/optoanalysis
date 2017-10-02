from sys import version
print(version)
if version[0] != '3':
    raise OSError("This package requires python 3")
from setuptools import setup
import os

mypackage_root_dir = os.path.dirname(__file__)
with open(os.path.join(mypackage_root_dir, 'requirements.txt')) as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(mypackage_root_dir, 'optoanalysis/VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='optoanalysis',
      version=version,
      description='Python package with functions for data analysis',
      author='Ashley Setter',
      author_email='A.Setter@soton.ac.uk',
      url="https://github.com/AshleySetter/optoanalysis",
      download_url="https://github.com/AshleySetter/optoanalysis/archive/{}.tar.gz".format(version),
      include_package_data=True,
      packages=['optoanalysis',
                'optoanalysis.LeCroy',
                'optoanalysis.Saleae',
                'optoanalysis.thermo',
      ],
      install_requires=requirements,
)
