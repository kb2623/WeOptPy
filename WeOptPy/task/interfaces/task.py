# encoding=utf8

"""The implementation of tasks."""

import numpy as np
from numpy import random as rand

from WeOptPy.util.utility import (
	limit_repair,
	fullArray
)
from WeOptPy.task.optimizationtype import OptimizationType
from WeOptPy.factory import Factory


class Task:
	r"""Class representing problem to solve with optimization.

	Date:
		2019

	Author:
		Klemen Berkovič

	Attributes:
		d (int): Dimension of the problem.
		lower (numpy.ndarray): lower bounds of the problem.
		upper (numpy.ndarray): upper bounds of the problem.
		bRange (numpy.ndarray): Search range between upper and lower limits.
		optimization_type (OptimizationType): Optimization type to use.

	See Also:
		* :class:`NiaPy.util.Utility`
	"""
	D = 0
	benchmark = None
	Lower, Upper, bRange = np.inf, np.inf, np.inf
	optType = OptimizationType.MINIMIZATION

	def __init__(self, d=0, optimization_type=OptimizationType.MINIMIZATION, benchmark=None, lower=None, upper=None, frepair=limit_repair, **kwargs):
		r"""Initialize task class for optimization.

		Args:
			d (Optional[int]): Number of dimensions.
			optimization_type (Optional[OptimizationType]): Set the type of optimization.
			benchmark (Union[str, Benchmark]): Problem to solve with optimization.
			lower (Optional[numpy.ndarray]): lower limits of the problem.
			upper (Optional[numpy.ndarray]): upper limits of the problem.
			frepair (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for reparing individuals components to desired limits.
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* `func`:NiaPy.util.Utility.__init__`
			* `func`:NiaPy.util.Utility.repair`
		"""
		# dimension of the problem
		self.D = d
		# set optimization type
		self.optType = optimization_type
		# set optimization function
		self.benchmark = benchmark
		if self.benchmark is not None: self.Fun = self.benchmark.function() if self.benchmark is not None else None
		# set lower limits
		if lower is not None: self.Lower = fullArray(lower, self.D)
		elif lower is None and benchmark is not None: self.Lower = fullArray(self.benchmark.Lower, self.D)
		else: self.Lower = fullArray(0, self.D)
		# set upper limits
		if upper is not None: self.Upper = fullArray(upper, self.D)
		elif upper is None and benchmark is not None: self.Upper = fullArray(self.benchmark.Upper, self.D)
		else: self.Upper = fullArray(0, self.D)
		# set range
		self.bRange = self.Upper - self.Lower
		# set repair function
		self.frepair = frepair

	def dim(self):
		r"""Get the number of dimensions.

		Returns:
			int: Dimension of problem optimizing.
		"""
		return self.D

	def lower(self):
		r"""Get the array of lower bound constraint.

		Returns:
			numpy.ndarray: lower bound.
		"""
		return self.Lower

	def upper(self):
		r"""Get the array of upper bound constraint.

		Returns:
			numpy.ndarray: upper bound.
		"""
		return self.Upper

	def range(self):
		r"""Get the range of bound constraint.

		Returns:
			numpy.ndarray: Range between lower and upper bound.
		"""
		return self.bRange

	def repair(self, x, rnd=rand):
		r"""Repair solution and put the solution in the random position inside of the bounds of problem.

		Args:
			x (numpy.ndarray): Solution to check and repair if needed.
			rnd (mtrand.RandomState): Random number generator.

		Returns:
			numpy.ndarray: Fixed solution.

		See Also:
			* :func:`NiaPy.util.limitRepair`
			* :func:`NiaPy.util.limitInversRepair`
			* :func:`NiaPy.util.wangRepair`
			* :func:`NiaPy.util.randRepair`
			* :func:`NiaPy.util.reflectRepair`
		"""
		return self.frepair(x, self.Lower, self.Upper, rnd=rnd)

	def next_iteration(self):
		r"""Increments the number of algorithm iterations."""

	def start(self):
		r"""Start stopwatch."""

	def eval(self, x):
		r"""Evaluate the solution A.

		Args:
			x (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Fitness/function values of solution.
		"""
		return self.Fun(x) * self.optType.value

	def is_feasible(self, x):
		r"""Check if the solution is feasible.

		Arguments:
			x (Union[numpy.ndarray, Individual]): Solution to check for feasibility.

		Returns:
			bool: `True` if solution is in feasible space else `False`.
		"""
		return False not in (x >= self.Lower) and False not in (x <= self.Upper)

	def stop_cond(self):
		r"""Check if optimization task should stop.

		Returns:
			bool: `True` if stopping condition is meet else `False`.
		"""
		return False


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
