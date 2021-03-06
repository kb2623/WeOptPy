# encoding=utf8

"""Artificial bee colony algorithm test case module."""

from WeOptPy.algorithms import ArtificialBeeColonyAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class ABCTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ArtificialBeeColonyAlgorithm

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertEqual(len(d), 2)
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['n'](-10))
		self.assertTrue(d['Limit'](10))
		self.assertFalse(d['Limit'](0))
		self.assertFalse(d['Limit'](-10))

	def test_custom_works_fine(self):
		abc_custom = self.algo(n=10, Limit=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, abc_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		abc_custom = self.algo(n=10, Limit=2, seed=self.seed)
		abc_customc = self.algo(n=10, Limit=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, abc_custom, abc_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
