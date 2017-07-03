from setuptools import setup
from setuptools.extension import Extension
import os
import numpy 
from Cython.Build import cythonize
from Cython.Build import build_ext
import subprocess
from sys import argv

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

#if mypackage_root_dir == "":
#    mypackage_root_dir = "." # so that if running from current directory subprocess starts in current directory
#
#def run_process(process_string):
#    popen = subprocess.Popen(process_string,
#                             cwd=mypackage_root_dir,
#                             stdout=subprocess.PIPE,
#                             universal_newlines=True,
#                             shell=True) # builds cython code
#    for stdout_line in iter(popen.stdout.readline, ""): 
#        print(stdout_line) # prints output of cython build process
#    popen.stdout.close()
#
#if argv[-1] != "--inplace": # so that it doesn't recursively call setup.py
#    run_process("python3 setup.py build_ext --inplace")

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
