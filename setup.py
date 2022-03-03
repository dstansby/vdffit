from setuptools import find_packages, setup

setup(packages=find_packages(),
      use_scm_version=True,
      setup_requires=['setuptools_scm'],)
