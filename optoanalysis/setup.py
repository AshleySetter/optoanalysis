from setuptools import setup
from setuptools.extension import Extension
import os
import numpy
try:
    from Cython.Build import cythonize
    from Cython.Build import build_ext
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'Cython'])
    from Cython.Build import cythonize
    from Cython.Build import build_ext

    
mypackage_root_dir = os.path.dirname(__file__)
with open(os.path.join(mypackage_root_dir, 'requirements.txt')) as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(mypackage_root_dir, 'optoanalysis/VERSION')) as version_file:
    version = version_file.read().strip()

extensions = [Extension(
    name="solve",
    sources=["optoanalysis/sde_solver/solve.pyx"],
    include_dirs=[numpy.get_include()],
    )
]

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
                'optoanalysis.sim_data',
                'optoanalysis.thermo',
                'optoanalysis.sde_solver',
      ],
      ext_modules = cythonize(extensions),
      install_requires=requirements,
)
