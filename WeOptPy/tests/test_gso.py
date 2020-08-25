# encoding=utf8

from WeOptPy.algorithms import (
	GlowwormSwarmOptimization,
	GlowwormSwarmOptimizationV1,
	GlowwormSwarmOptimizationV2,
	GlowwormSwarmOptimizationV3
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class GSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimization

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())


class GSOv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimizationV1

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())


class GSOv2TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimizationV2

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())


class GSOv3TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimizationV3

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
