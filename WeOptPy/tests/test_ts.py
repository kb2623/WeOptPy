# encoding=utf8

"""Tabu search algorithm test case."""

from unittest import skip

from WeOptPy.algorithms import TabuSearch
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)

# TODO algorithm in development stage


class TSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = TabuSearch

	@skip('Not implemented jet!!!')
	def test_custom_works_fine(self):
		ts_custom = self.algo(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed)
		ts_customc = self.algo(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ts_custom, ts_customc, benc=Sphere)

	@skip('Not implemented jet!!!')
	def test_griewank_works_fine(self):
		ts_griewank = self.algo(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed)
		ts_griewankc = self.algo(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ts_griewank, ts_griewankc)


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
