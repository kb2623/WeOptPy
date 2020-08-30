# encoding=utf8

from WeOptPy.algorithms import (
	FireworksAlgorithm,
	DynamicFireworksAlgorithm,
	EnhancedFireworksAlgorithm,
	BareBonesFireworksAlgorithm,
	DynamicFireworksAlgorithmGauss
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class BBFWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = BareBonesFireworksAlgorithm

	def test_custom_works_fine(self):
		bbfwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bbfwa_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		bbfwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		bbfwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, bbfwa_custom, bbfwa_customc, MyBenchmark())


class FWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = FireworksAlgorithm

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, fwa_custom, fwa_customc, MyBenchmark())


class EFWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = EnhancedFireworksAlgorithm

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())


class DFWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynamicFireworksAlgorithm

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())


class DFWAGTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynamicFireworksAlgorithmGauss

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
