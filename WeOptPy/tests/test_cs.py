# encoding=utf8

"""Cuckoo search algorithm test case module."""

from WeOptPy.algorithms import CuckooSearch
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class CSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CuckooSearch

	def test_custom_works_fine(self):
		cs_custom = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cs_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		cs_custom = self.algo(NP=20, seed=self.seed)
		cs_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, cs_custom, cs_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
