# encoding=utf8

"""Hybrid bat algorithm test case module."""

from WeOptPy.algorithms import HybridBatAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class HBATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = HybridBatAlgorithm

	def test_custom_works_fine(self):
		hba_custom = self.algo(n=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hba_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		hba_custom = self.algo(n=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		hba_customc = self.algo(n=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, hba_custom, hba_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
