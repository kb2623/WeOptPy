# encoding=utf8

from WeOptPy.algorithms import CoralReefsOptimization
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class CROTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CoralReefsOptimization

	def test_custom_works_fine(self):
		cro_custom = self.algo(N=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cro_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		cro_custom = self.algo(N=20, seed=self.seed)
		cro_customc = self.algo(N=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, cro_custom, cro_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
