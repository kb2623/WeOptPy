# encoding=utf8

"""Firefly algorithm test case suit module."""

from WeOptPy.algorithms import FireflyAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class FATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = FireflyAlgorithm

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['alpha'](10))
		self.assertFalse(d['alpha'](-10))
		self.assertTrue(d['betamin'](10))
		self.assertFalse(d['betamin'](-10))
		self.assertTrue(d['gamma'](10))
		self.assertFalse(d['gamma'](-10))
		self.assertTrue(d['n'](1))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['n'](-1))

	def test_works_fine(self):
		fa = self.algo(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fa, Sphere())

	def test_works_fine_parallel(self):
		fa = self.algo(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
		fac = self.algo(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, fa, fac, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
