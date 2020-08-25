# encoding=utf8

from WeOptPy.algorithms import CatSwarmOptimization
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class CSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CatSwarmOptimization

	def test_custom_works_fine(self):
		cso_custom = self.algo(NP=20, seed=self.seed)
		cso_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cso_custom, cso_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
