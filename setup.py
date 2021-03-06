#!/usr/bin/env python

"""Setup script for the package."""

from __future__ import (
	division,
	absolute_import,
	with_statement,
	print_function,
	unicode_literals,
	nested_scopes,
	generators
)

import os
import sys
import logging

import setuptools


def check_python_version(min_python_version):
	r"""Exit when the Python version is too low.

	Args:
		min_python_version (float): Minimum python version.

	Returns:
		bool: Get minimum version for python.
	"""
	if sys.version < min_python_version: sys.exit("Python {0}+ is required." % min_python_version)


def read_package_variable(key, package_name='WeOptPy', filename='__init__.py'):
	"""Read the value of a variable from the package without importing."""
	module_path = os.path.join(package_name, filename)
	with open(module_path) as module:
		for line in module:
			parts = line.strip().split(' ', 2)
			if parts[:-1] == [key, '=']: return parts[-1].strip("'")
	logging.warning("'%s' not found in '%s'", key, module_path)
	return None


def build_description():
	r"""Build a description for the project from documentation files.

	Returns:
		str: Description.
	"""
	try: readme = open("README.rst").read()
	except IOError: return "<placeholder>"
	return readme  # return readme + '\n' + changelog


def make_python_requires(min_version):
	r"""Get minimum required python version.

	Args:
		min_version (float): Minimal supported version.

	Returns:
		str: Formatted minimal version.
	"""
	return '>=%s' % min_version


AUTHOR = 'kb2623'
AUTHOR_EMAIL = 'roxor1992@gmail.com'
MINIMUM_PYTHON_VERSION = '2.7'

check_python_version(MINIMUM_PYTHON_VERSION)

setuptools.setup(
	name=read_package_variable('__project__'),
	version=read_package_variable('__version__'),
	description=read_package_variable('__description__'),
	url=read_package_variable('__url__'),
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	license=read_package_variable('__license__'),
	packages=setuptools.find_packages(),
	long_description=build_description(),
	python_requires=make_python_requires(MINIMUM_PYTHON_VERSION),
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'Natural Language :: English',
		'Operating System :: OS Independent',
		'Programming Language :: Python',
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Topic :: Scientific/Engineering',
		'Topic :: Software Development'
	],
	install_requires=[
		'numpy >= 1.16.2',
		'scipy >= 1.2.1',
		'pandas >= 0.21.0',
		'scikit-learn >= 0.20.4',
		'XlsxWriter >= 1.1.5',
		'enum34 >= 1.1.6',
		'future >= 0.18.2'
	],
)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
