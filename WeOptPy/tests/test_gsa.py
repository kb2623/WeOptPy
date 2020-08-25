# encoding=utf8

from WeOptPy.algorithms import GravitationalSearchAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class GSATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GravitationalSearchAlgorithm

	def test_Custom_works_fine(self):
		gsa_custom = self.algo(NP=40, seed=self.seed)
		gsa_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gsa_custom, gsa_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
