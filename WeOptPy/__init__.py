# encoding=utf8

"""Python micro framework for building nature-inspired algorithms."""

from __future__ import print_function


from WeOptPy import (
	util,
	algorithms,
	benchmarks,
	task
)
from WeOptPy.runner import Runner
from WeOptPy.factory import Factory

__all__ = [
	'algorithms',
	'benchmarks',
	'util',
	'task',
	'Runner',
	'Factory'
]
__project__ = 'WeOptPy'
__version__ = '1.0.0'
__license__ = 'MIT'
__description__ = 'Python micro framework for building nature-inspired algorithms.'
__url__ = 'https://github.com/kb2623/WeOptPy'

VERSION = '%s v%s' % (__project__, __version__)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
