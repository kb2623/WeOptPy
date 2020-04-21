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

from distutils.core import (
	setup,
	Extension
)
from distutils.util import convert_path


def find_packages(base_path):
	base_path = convert_path(base_path)
	found = []
	for root, dirs, files in os.walk(base_path, followlinks=True):
		dirs[:] = [d for d in dirs if d[0] != '.' and d not in ('ez_setup', '__pycache__')]
		relpath = os.path.relpath(root, base_path)
		parent = relpath.replace(os.sep, '.').lstrip('.')
		if relpath != '.' and parent not in found:
			# foo.bar package but no foo package, skip
			continue
		for dir in dirs:
			if os.path.isfile(os.path.join(root, dir, '__init__.py')):
				package = '.'.join((parent, dir)) if parent else dir
				found.append(package)
	return found


def check_python_version(MINIMUM_PYTHON_VERSION):
	r"""Exit when the Python version is too low."""
	if sys.version < MINIMUM_PYTHON_VERSION:
		sys.exit("Python {0}+ is required.".format(MINIMUM_PYTHON_VERSION))


def read_package_variable(key, filename='__init__.py'):
	"""Read the value of a variable from the package without importing."""
	module_path = os.path.join(PACKAGE_NAME, filename)
	with open(module_path) as module:
		for line in module:
			parts = line.strip().split(' ', 2)
			if parts[:-1] == [key, '=']:
				return parts[-1].strip("'")
	logging.warning("'%s' not found in '%s'", key, module_path)
	return None


def build_description():
	"""Build a description for the project from documentation files."""
	try:
		# changelog = open("CHANGELOG.rst").read()
		readme = open("README.rst").read()
	except IOError:
		return "<placeholder>"
	else:
		return readme  # return readme + '\n' + changelog


def build_extension():
	e = Extension(
		name='WeOptPy.benchmarks.functions',
		sources=['functions/bfuncs.c'],
		include_dirs=['functions'],
		language='c',
		extra_compile_args=['-std=c11', '-O3'],
		extra_link_args=['-O3', '-lm']
	)
	return [e]


PACKAGE_NAME = 'WeOptPy'
MINIMUM_PYTHON_VERSION = '2.7'
PACKAGE_VERSION = read_package_variable('__version__')
URL = 'https://github.com/kb2623/WeOptPy'
AUTHOR = 'kb2623'
AUTHOR_EMAIL = 'roxor1992@gmail.com'
DESCRIPTION = 'Python micro framework for building nature-inspired algorithms.'
LICENSE = 'MIT'

check_python_version(MINIMUM_PYTHON_VERSION)

setup(
	name=PACKAGE_NAME,
	version=PACKAGE_VERSION,
	description=DESCRIPTION,
	url=URL,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	# packages=find_packages(PACKAGE_NAME),
	long_description=build_description(),
	license=LICENSE,
	ext_modules=build_extension(),
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
	]
)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
