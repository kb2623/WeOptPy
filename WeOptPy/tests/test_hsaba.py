# encoding=utf8

"""Hybrid self adaptive bat algorithm test case module."""

from WeOptPy.algorithms import HybridSelfAdaptiveBatAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class HSABATestCase(AlgorithmTestCase):
	r"""Test case for HybridSelfAdaptiveBatAlgorithm algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.modified.HybridSelfAdaptiveBatAlgorithm`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = HybridSelfAdaptiveBatAlgorithm

	def test_algorithm_info_fine(self):
		"""Test case for algorithm info."""
		i = self.algo.algorithm_info()
		self.assertIsNotNone(i)

	def test_type_parameters_fine(self):
		"""Test case for type parameters."""
		d = self.algo.type_parameters()
		# Test F parameter check
		self.assertIsNotNone(d.get('F', None))
		self.assertFalse(d['F'](-30))
		self.assertFalse(d['F'](-.3))
		self.assertTrue(d['F'](.3))
		self.assertTrue(d['F'](.39))
		# Test CR parameter check
		self.assertIsNotNone(d.get('CR', None))
		self.assertFalse(d['CR'](10))
		self.assertFalse(d['CR'](-10))
		self.assertFalse(d['CR'](-1))
		self.assertTrue(d['CR'](.3))
		self.assertTrue(d['CR'](.0))
		self.assertTrue(d['CR'](1.))

	def test_custom_works_fine(self):
		"""Test case for running algorithm on costume benchmarks."""
		hsaba_custom = self.algo(NP=10, Limit=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hsaba_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		"""Test case for running algorithm on costume benchmarks."""
		hsaba_custom = self.algo(NP=10, Limit=2, seed=self.seed)
		hsaba_customc = self.algo(NP=10, Limit=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, hsaba_custom, hsaba_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
