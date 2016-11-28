from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='DataHandling',
      version='1.1',
      description='Python package with functions for data analysis',
      author='Ashley Setter',
      auther_email='A.Setter@soton.ac.uk',
      url=None,
      packages=['DataHandling', 'DataHandling.LeCroy'],
      package_dir={'DataHandling': 'DataHandling',
      },
      install_requires=requirements,
)
