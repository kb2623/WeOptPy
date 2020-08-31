# encoding=utf8

from WeOptPy.algorithms import GravitationalSearchAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class GSATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GravitationalSearchAlgorithm

	def test_Custom_works_fine(self):
		gsa_custom = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gsa_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		gsa_custom = self.algo(NP=40, seed=self.seed)
		gsa_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, gsa_custom, gsa_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
