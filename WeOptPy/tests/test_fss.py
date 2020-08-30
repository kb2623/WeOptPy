# encoding=utf8

from WeOptPy.algorithms import FishSchoolSearch
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class FSSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = FishSchoolSearch

	def test_custom_works_fine(self):
		fss_custom = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fss_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		fss_custom = self.algo(NP=20, seed=self.seed)
		fss_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, fss_custom, fss_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
