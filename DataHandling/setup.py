from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='DataHandling',
      version='1.5.0',
      description='Python package with functions for data analysis',
      author='Ashley Setter',
      author_email='A.Setter@soton.ac.uk',
      url=None,
      packages=['DataHandling', 'DataHandling.LeCroy', 'Datahandling.SimData'],
      install_requires=requirements,
)
