# encoding=utf8

from WeOptPy.algorithms import BatAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class BATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = BatAlgorithm

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['Qmax'](10))
		self.assertTrue(d['Qmin'](10))
		self.assertTrue(d['r'](10))
		self.assertFalse(d['r'](-10))
		self.assertFalse(d['r'](0))
		self.assertFalse(d['A'](0))
		self.assertFalse(d['A'](-19))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertTrue(d['A'](10))
		self.assertFalse(d['Qmin'](None))
		self.assertFalse(d['Qmax'](None))

	def test_custom_works_fine(self):
		ba_custom = self.algo(NP=20, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ba_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		ba_custom = self.algo(NP=20, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		ba_customc = self.algo(NP=20, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ba_custom, ba_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
