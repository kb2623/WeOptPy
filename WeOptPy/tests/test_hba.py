# encoding=utf8

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
		hba_custom = self.algo(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		hba_customc = self.algo(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hba_custom, hba_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
