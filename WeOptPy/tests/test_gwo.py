# encoding=utf8

from WeOptPy.algorithms import GreyWolfOptimizer
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class GWOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GreyWolfOptimizer

	def test_custom_works_fine(self):
		gwo_custom = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gwo_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		gwo_custom = self.algo(NP=20, seed=self.seed)
		gwo_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, gwo_custom, gwo_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
