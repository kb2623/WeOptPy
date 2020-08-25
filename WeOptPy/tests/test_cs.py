# encoding=utf8

from WeOptPy.algorithms import CuckooSearch
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class CSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CuckooSearch

	def test_custom_works_fine(self):
		cs_custom = self.algo(NP=20, seed=self.seed)
		cs_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cs_custom, cs_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
