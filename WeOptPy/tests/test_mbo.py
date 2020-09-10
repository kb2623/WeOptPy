# encoding=utf8

"""Monarch butterfly optimization algorithm test case module."""

from WeOptPy.algorithms import MonarchButterflyOptimization
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class MBOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MonarchButterflyOptimization

	def test_type_parameters(self):
		tp = MonarchButterflyOptimization.type_parameters()
		self.assertTrue(tp['n'](1))
		self.assertFalse(tp['n'](0))
		self.assertFalse(tp['n'](-1))
		self.assertFalse(tp['n'](1.0))
		self.assertTrue(tp['PAR'](1.0))
		self.assertFalse(tp['PAR'](0.0))
		self.assertFalse(tp['PAR'](-1.0))
		self.assertTrue(tp['PER'](1.0))
		self.assertFalse(tp['PER'](0.0))
		self.assertFalse(tp['PER'](-1.0))

	def test_works_fine(self):
		mbo = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mbo, Sphere())

	def test_works_fine_parallel(self):
		mbo = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
		mboc = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mbo, mboc, Sphere())



# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
