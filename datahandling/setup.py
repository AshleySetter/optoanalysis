from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='datahandling',
      version='2.0.0',
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
