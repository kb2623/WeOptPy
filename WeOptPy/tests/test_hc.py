# encoding=utf8

"""Hill climb algorithm test case module."""

from WeOptPy.algorithms import HillClimbAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class HCTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = HillClimbAlgorithm

	def test_custom_works_fine(self):
		ihc_custom = self.algo(delta=0.4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ihc_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		ihc_custom = self.algo(delta=0.4, seed=self.seed)
		ihc_customc = self.algo(delta=0.4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ihc_custom, ihc_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
