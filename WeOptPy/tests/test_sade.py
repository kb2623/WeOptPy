# encoding=utf8

from unittest import skip

from WeOptPy.algorithms import (
	StrategyAdaptationDifferentialEvolution,
	StrategyAdaptationDifferentialEvolutionV1
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class SADETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = StrategyAdaptationDifferentialEvolution

	@skip('Not implementd jet!!!')
	def test_custom_works_fine(self):
		sade_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		sade_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, sade_custom, sade_customc, MyBenchmark())


class SADEv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = StrategyAdaptationDifferentialEvolutionV1

	@skip('Not implementd jet!!!')
	def test_custom_works_fine(self):
		sadev1_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		sadev1_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, sadev1_custom, sadev1_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
