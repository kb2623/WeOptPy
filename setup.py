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


def build_extensions():
	e = list()
	e.append(setuptools.Extension(
		name='WeOptPy.benchmarks.functions',
		sources=['functions/bfuncs.c'],
		include_dirs=['functions'],
		language='c',
		extra_compile_args=['-std=c11', '-O3', '-shared'],
		extra_link_args=['-lm', '-O3', '-shared']
	))
	return e


def check_python_version(min_python_verions):
	"""Exit when the Python version is too low."""
	if sys.version < min_python_verions: sys.exit("Python {0}+ is required.".format(min_python_verions))


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
	"""Build a description for the project from documentation files."""
	try: readme = open("README.rst").read()
	except IOError: return "<placeholder>"
	return readme  # return readme + '\n' + changelog


def make_python_requires(min_version):
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
	ext_modules=build_extensions(),
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
	tests_requires=[
		'flake8 ~= 3.7.7',
		'astroid >= 2.0.4',
		'pytest ~= 3.7.1',
		'coverage ~= 4.4.2',
		'coverage-space ~= 1.0.2'
	],
	install_requires=[
		'numpy >= 1.16.2',
		'scipy >= 1.1.0',
		'enum34 >= 1.1.6',
	],
)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
