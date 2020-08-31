# encoding=utf8

"""Simulated annealing test case module."""

from WeOptPy.algorithms import SimulatedAnnealing
from WeOptPy.algorithms.sa import coolLinear
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class SATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = SimulatedAnnealing

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['delta'](1))
		self.assertFalse(d['delta'](0))
		self.assertFalse(d['delta'](-1))
		self.assertTrue(d['T'](1))
		self.assertFalse(d['T'](0))
		self.assertFalse(d['T'](-1))
		self.assertTrue(d['deltaT'](1))
		self.assertFalse(d['deltaT'](0))
		self.assertFalse(d['deltaT'](-1))
		self.assertTrue(d['epsilon'](0.1))
		self.assertFalse(d['epsilon'](-0.1))
		self.assertFalse(d['epsilon'](10))

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, Sphere())

	def test_custom1_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed, coolingMethod=coolLinear)
		ca_customc = self.algo(NP=40, seed=self.seed, coolingMethod=coolLinear)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ca_custom, ca_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
