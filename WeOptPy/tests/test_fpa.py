# encoding=utf8

from WeOptPy.algorithms import FlowerPollinationAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class FPATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = FlowerPollinationAlgorithm

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertTrue(d['beta'](10))
		self.assertFalse(d['beta'](0))
		self.assertFalse(d['beta'](-10))
		self.assertTrue(d['p'](0.5))
		self.assertFalse(d['p'](-0.5))
		self.assertFalse(d['p'](1.5))

	def test_custom_works_fine(self):
		fpa_custom = self.algo(NP=10, p=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fpa_custom, MyBenchmark())

	def test_Custom_works_fine_parallel(self):
		fpa_custom = self.algo(NP=10, p=0.5, seed=self.seed)
		fpa_customc = self.algo(NP=10, p=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, fpa_custom, fpa_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
