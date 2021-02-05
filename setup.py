#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'headbang.py'
DESCRIPTION = 'Percussive beat tracking in fully mixed metal songs'
URL = 'https://github.com/sevagh/headbang.py'
EMAIL = 'sevag.hanssian@gmail.com'
AUTHOR = 'Sevag Hanssian'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

REQUIRED = []
with open("./requirements.txt") as f:
    for l in f:
        REQUIRED.append(l[:-1])

TEST_REQUIRED = []
with open("./requirements-test.txt") as f:
    for l in f:
        TEST_REQUIRED.append(l[:-1])


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    py_modules=['headbang'],
    install_requires=REQUIRED,
    test_requires=REQUIRED+TEST_REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
)