from distutils.core import setup
import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

mypackage_root_dir = os.path.dirname(__file__)
version_file = open(os.path.join(mypackage_root_dir, 'datahandling/VERSION'))
version = version_file.read().strip()

setup(name='datahandling',
      version=version,
      description='Python package with functions for data analysis',
      author='Ashley Setter',
      author_email='A.Setter@soton.ac.uk',
      url="https://github.com/AshleySetter/datahandling",
      download_url="https://github.com/AshleySetter/datahandling/archive/0.1.tar.gz",
      packages=['datahandling',
                'datahandling.LeCroy',
                'datahandling.SimData',
      ],
      install_requires=requirements,
)
