# encoding=utf8

from WeOptPy.algorithms import SineCosineAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class SCATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = SineCosineAlgorithm

	def test_algorithm_info_fine(self):
		self.assertIsNotNone(self.algo.algorithm_info())

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertIsNotNone(d.get('n', None))
		self.assertIsNotNone(d.get('a', None))
		self.assertIsNotNone(d.get('Rmin', None))
		self.assertIsNotNone(d.get('Rmax', None))

	def test_custom_works_fine(self):
		sca_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, sca_custom, MyBenchmark())

	def test_Custom_works_fine_parallel(self):
		sca_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		sca_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, sca_custom, sca_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
