# encoding=utf8

"""Implementation of benchmarks utility function."""

from WeOptPy import algorithms
from WeOptPy.util.utility import explore_package_for_classes


class Factory:
	r"""Base class with string mappings to benchmarks and algorithms.

	Author:
		Klemen Berkovic

	Date:
		2020

	License:
		MIT

	Attributes:
		algorithm_classes (Dict[str, Algorithm]): Mapping for fetching Algorithm classes.
	"""
	def __init__(self):
		r"""Init benchmark classes."""
		self.algorithm_classes = Factory.__init_factory(algorithms, algorithms.interfaces.Algorithm)

	@staticmethod
	def __init_factory(module, dtype):
		r"""Initialize the factory with this helper function.

		Args:
			module (module): Module to check for classes.
			dtype (class): Types of classes to search fot in a module.

		Returns:
			Dict[str, Any]: Mapping from string to type.
		"""
		tmp = {}
		for cc in explore_package_for_classes(module, dtype).values():
			for val in cc.Name: tmp[val] = cc
		return tmp

	def get_algorithm(self, algorithm, **kwargs):
		r"""Get the algorithm for optimization.

		Args:
			algorithm (Union[str, Algorithm]): Algorithm to use.
			kwargs (dict): Additional arguments for algorithm.

		Raises:
			TypeError: If algorithm is not defined.

		Returns:
			Algorithm: Initialized algorithm.
		"""
		if isinstance(algorithm, algorithms.interfaces.Algorithm): return algorithm
		elif issubclass(type(algorithms), algorithms.interfaces.Algorithm): return algorithm(**kwargs)
		elif algorithm in self.algorithm_classes.keys(): return self.algorithm_classes[algorithm](**kwargs)
		else: raise TypeError("Passed algorithm '%s' is not defined!" % algorithm)


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
