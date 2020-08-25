# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.algorithms.interfaces.individual import (
	Individual,
	default_numpy_init,
	default_individual_init
)

__all__ = [
	'Individual',
	'default_numpy_init',
	'default_individual_init',
	'Algorithm'
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
