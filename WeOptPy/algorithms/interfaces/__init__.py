# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.algorithms.interfaces.individual import (
	Individual,
	defaultNumPyInit,
	defaultIndividualInit
)

__all__ = [
	'Individual',
	'defaultNumPyInit',
	'defaultIndividualInit',
	'Algorithm'
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
