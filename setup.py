from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    return codecs.open(os.path.join(here, *parts), 'r', encoding='utf8').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import tox
        import shlex
        errno = tox.cmdline(args=shlex.split(self.tox_args))
        sys.exit(errno)

setup(
    name='nsim',
    version=find_version('nsim', '__init__.py'),
    url='http://github.com/mattja/nsim/',
    license='GPLv3+',
    author='Matthew J. Aburn',
    install_requires=['ipyparallel>=4.0,<5.0',
                      'distob>=0.3.2',
                      'sdeint>=0.2.0',
                      'numpy>=1.6',
                      'scipy>=0.9',
                      'matplotlib>=1.1'],
    tests_require=['tox'],
    cmdclass = {'test': Tox},
    author_email='mattja6@gmail.com',
    description='Simulate systems from ODEs or SDEs, analyze timeseries.',
    long_description=read('README.rst'),
    packages=['nsim', 'nsim.analyses1', 'nsim.analysesN', 'nsim.models'],
    platforms='any',
    zip_safe=False,
    keywords=['simulation', 'ODE', 'SDE', 'SODE', 'time series'],
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: System :: Distributed Computing',
        ],
    extras_require={'read_EDF_BDF_files': ['edflib>=0.7'],
                    'read_all_files': ['python-biosig>=1.3']}
)
