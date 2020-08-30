# encoding=utf8

"""Implementation of various utility functions."""

import inspect

import numpy as np
from numpy import random as rand

__all__ = [
	"limit_repair",
	"limit_invers_repair",
	"objects2array",
	"wang_repair",
	"rand_repair",
	"full_array",
	"reflect_repair",
	"explore_package_for_classes"
]


def limit_repair(x, lower, upper, *args, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		lower (numpy.ndarray): lower bounds of search space.
		upper (numpy.ndarray): upper bounds of search space.
		args (list): Additional arguments.
		kwargs (dict): Additional keyword arguments.

	Returns:
		numpy.ndarray: Solution in search space.
	"""
	ir = np.where(x < lower)
	x[ir] = lower[ir]
	ir = np.where(x > upper)
	x[ir] = upper[ir]
	return x


def limit_invers_repair(x, lower, upper, *args, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		lower (numpy.ndarray): lower bounds of search space.
		upper (numpy.ndarray): upper bounds of search space.
		args (list): Additional arguments.
		kwargs (dict): Additional keyword arguments.

	Returns:
		numpy.ndarray: Solution in search space.
	"""
	ir = np.where(x < lower)
	x[ir] = upper[ir]
	ir = np.where(x > upper)
	x[ir] = lower[ir]
	return x


def wang_repair(x, lower, upper, *args, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		lower (numpy.ndarray): lower bounds of search space.
		upper (numpy.ndarray): upper bounds of search space.
		args (list): Additional arguments.
		kwargs (dict): Additional keyword arguments.

	Returns:
		numpy.ndarray: Solution in search space.
	"""
	ir = np.where(x < lower)
	x[ir] = np.amin([upper[ir], 2 * lower[ir] - x[ir]], axis=0)
	ir = np.where(x > upper)
	x[ir] = np.amax([lower[ir], 2 * upper[ir] - x[ir]], axis=0)
	return x


def rand_repair(x, lower, upper, rnd=rand, *args, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		lower (numpy.ndarray): lower bounds of search space.
		upper (numpy.ndarray): upper bounds of search space.
		rnd (mtrand.RandomState): Random generator.
		args (list): Additional arguments.
		kwargs (dict): Additional keyword arguments.

	Returns:
		numpy.ndarray: Fixed solution.
	"""
	ir = np.where(x < lower)
	x[ir] = rnd.uniform(lower[ir], upper[ir])
	ir = np.where(x > upper)
	x[ir] = rnd.uniform(lower[ir], upper[ir])
	return x


def reflect_repair(x, lower, upper, *args, **kwargs):
	r"""Repair solution and put the solution in search space with reflection of how much the solution violates a bound.

	Args:
		x (numpy.ndarray): Solution to be fixed.
		lower (numpy.ndarray): lower bounds of search space.
		upper (numpy.ndarray): upper bounds of search space.
		args (list): Additional arguments.
		kwargs (dict): Additional keyword arguments.

	Returns:
		numpy.ndarray: Fix solution.
	"""
	ir = np.where(x > upper)
	x[ir] = lower[ir] + x[ir] % (upper[ir] - lower[ir])
	ir = np.where(x < lower)
	x[ir] = lower[ir] + x[ir] % (upper[ir] - lower[ir])
	return x


def full_array(a, d):
	r"""Fill or create array of length d, from value or value form a.

	Arguments:
		a (Union[int, float, Any, numpy.ndarray, Iterable[Union[int, float, Any]]]): Input values for fill.
		d (int): Length of new array.

	Returns:
		numpy.ndarray: Array filled with passed values or value.
	"""
	A = []
	if isinstance(a, (int, float)):
		A = np.full(d, a)
	elif isinstance(a, (np.ndarray, list, tuple)):
		if len(a) == d:
			A = a if isinstance(a, np.ndarray) else np.asarray(a)
		elif len(a) > d:
			A = a[:d] if isinstance(a, np.ndarray) else np.asarray(a[:d])
		else:
			for i in range(int(np.ceil(float(d) / len(a)))):
				A.extend(a[:d if (d - i * len(a)) >= len(a) else d - i * len(a)])
			A = np.asarray(A)
	return A


def objects2array(objs):
	r"""Convert `Iterable` array or list to `NumPy` array.

	Args:
		objs (Iterable[Any]): Array or list to convert.

	Returns:
		numpy.ndarray: Array of objects.
	"""
	a = np.empty(len(objs), dtype=object)
	for i, e in enumerate(objs):
		a[i] = e
	return a


def explore_package_for_classes(module, stype=object):
	r"""Explore the python package for classes.

	Args:
		module (Any): Module to inspect for classes.
		stype (Union[class, type]): Super type of search.

	Returns:
		Dict[str, Any]: Mapping for classes in package.
	"""
	tmp = {}
	for key, data in inspect.getmembers(module, inspect.isclass):
		if isinstance(data, stype) or issubclass(data, stype):
			tmp[key] = data
	return tmp


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
