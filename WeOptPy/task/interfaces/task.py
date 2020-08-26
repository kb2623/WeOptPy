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
		D (int): Dimension of the problem.
		Lower (numpy.ndarray): lower bounds of the problem.
		Upper (numpy.ndarray): upper bounds of the problem.
		bRange (numpy.ndarray): Search range between upper and lower limits.
		optType (OptimizationType): Optimization type to use.

	See Also:
		* :class:`NiaPy.util.Utility`
	"""
	D = 0
	benchmark = None
	Lower, Upper, bRange = np.inf, np.inf, np.inf
	optType = OptimizationType.MINIMIZATION

	def __init__(self, D=0, optType=OptimizationType.MINIMIZATION, benchmark=None, Lower=None, Upper=None, frepair=limit_repair, **kwargs):
		r"""Initialize task class for optimization.

		Args:
			D (Optional[int]): Number of dimensions.
			optType (Optional[OptimizationType]): Set the type of optimization.
			benchmark (Union[str, Benchmark]): Problem to solve with optimization.
			Lower (Optional[numpy.ndarray]): lower limits of the problem.
			Upper (Optional[numpy.ndarray]): upper limits of the problem.
			frepair (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for reparing individuals components to desired limits.
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* `func`:NiaPy.util.Utility.__init__`
			* `func`:NiaPy.util.Utility.repair`
		"""
		# dimension of the problem
		self.D = D
		# set optimization type
		self.optType = optType
		# set optimization function
		self.benchmark = Factory().get_benchmark(benchmark) if benchmark is not None else None
		if self.benchmark is not None: self.Fun = self.benchmark.function() if self.benchmark is not None else None
		# set lower limits
		if Lower is not None: self.Lower = fullArray(Lower, self.D)
		elif Lower is None and benchmark is not None: self.Lower = fullArray(self.benchmark.Lower, self.D)
		else: self.Lower = fullArray(0, self.D)
		# set upper limits
		if Upper is not None: self.Upper = fullArray(Upper, self.D)
		elif Upper is None and benchmark is not None: self.Upper = fullArray(self.benchmark.Upper, self.D)
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
		return self.Upper - self.Lower

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

	def eval(self, A):
		r"""Evaluate the solution A.

		Args:
			A (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Fitness/function values of solution.
		"""
		return self.Fun(self.D, A) * self.optType.value

	def is_feasible(self, A):
		r"""Check if the solution is feasible.

		Arguments:
			A (Union[numpy.ndarray, Individual]): Solution to check for feasibility.

		Returns:
			bool: `True` if solution is in feasible space else `False`.
		"""
		return False not in (A >= self.Lower) and False not in (A <= self.Upper)

	def stop_cond(self):
		r"""Check if optimization task should stop.

		Returns:
			bool: `True` if stopping condition is meet else `False`.
		"""
		return False


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
