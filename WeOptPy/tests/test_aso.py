# encoding=utf8

from WeOptPy.algorithms import AnarchicSocietyOptimization
from WeOptPy.algorithms.aso import (
	elitism,
	sequential,
	crossover
)

from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class ASOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AnarchicSocietyOptimization

	def test_parameter_types(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['n'](1))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['n'](-1))
		self.assertTrue(d['F'](10))
		self.assertFalse(d['F'](0))
		self.assertFalse(d['F'](-10))
		self.assertTrue(d['CR'](0.1))
		self.assertFalse(d['CR'](-19))
		self.assertFalse(d['CR'](19))
		self.assertTrue(d['alpha'](10))
		self.assertTrue(d['gamma'](10))
		self.assertTrue(d['theta'](10))


class ASOElitismTestCase(ASOTestCase):
	def test_custom_works_fine(self):
		aso_custom = self.algo(NP=40, Combination=elitism, seed=self.seed)
		aso_customc = self.algo(NP=40, Combination=elitism, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())


class ASOSequentialTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = self.algo(NP=40, Combination=sequential, seed=self.seed)
		aso_customc = self.algo(NP=40, Combination=sequential, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())


class ASOCrossoverTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = self.algo(NP=40, Combination=crossover, seed=self.seed)
		aso_customc = self.algo(NP=40, Combination=crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
