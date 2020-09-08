# encoding=utf8

"""Hybrid differential evolution algorithm test case module."""

from WeOptPy.algorithms import (
	DifferentialEvolutionMTS,
	DifferentialEvolutionMTSv1,
	DynNpDifferentialEvolutionMTS,
	DynNpDifferentialEvolutionMTSv1,
	MultiStrategyDifferentialEvolutionMTS,
	MultiStrategyDifferentialEvolutionMTSv1,
	DynNpMultiStrategyDifferentialEvolutionMTS,
	DynNpMultiStrategyDifferentialEvolutionMTSv1
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class DEMTSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DifferentialEvolutionMTS

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class DEMTSv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DifferentialEvolutionMTSv1

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class DynNpDEMTSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpDifferentialEvolutionMTS

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class DynNpDEMTSv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpDifferentialEvolutionMTSv1

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class MSDEMTSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultiStrategyDifferentialEvolutionMTS

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class MSDEMTSv1STestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultiStrategyDifferentialEvolutionMTSv1

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class DynNpMSDEMTSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpMultiStrategyDifferentialEvolutionMTS

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


class DynNpMSDEMTSv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpMultiStrategyDifferentialEvolutionMTSv1

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
