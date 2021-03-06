# encoding=utf8

"""Bees algorithm test case module."""

from WeOptPy.algorithms import BeesAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class BEATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = BeesAlgorithm

	def test_type_parameters(self):
		tp = self.algo.type_parameters()
		self.assertTrue(tp['n'](1))
		self.assertFalse(tp['n'](0))
		self.assertFalse(tp['n'](-1))
		self.assertFalse(tp['n'](1.0))
		self.assertTrue(tp['m'](1))
		self.assertFalse(tp['m'](0))
		self.assertFalse(tp['m'](-1))
		self.assertFalse(tp['m'](1.0))
		self.assertTrue(tp['e'](1))
		self.assertFalse(tp['e'](0))
		self.assertFalse(tp['e'](-1))
		self.assertFalse(tp['e'](1.0))
		self.assertTrue(tp['nep'](1))
		self.assertFalse(tp['nep'](0))
		self.assertFalse(tp['nep'](-1))
		self.assertFalse(tp['nep'](1.0))
		self.assertTrue(tp['nsp'](1))
		self.assertFalse(tp['nsp'](0))
		self.assertFalse(tp['nsp'](-1))
		self.assertFalse(tp['nsp'](1.0))
		self.assertTrue(tp['ngh'](1.0))
		self.assertTrue(tp['ngh'](0.5))
		self.assertFalse(tp['ngh'](0.0))
		self.assertFalse(tp['ngh'](-1))

	def test_works_fine(self):
		bea = self.algo(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bea, Sphere())

	def test_works_fine_parallel(self):
		bea = self.algo(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)
		beac = self.algo(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, bea, beac, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
