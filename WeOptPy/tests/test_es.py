# encoding=utf8

"""Evolution strategy test case module."""

from unittest import TestCase

from WeOptPy.algorithms.es import IndividualES
from WeOptPy.algorithms import (
	EvolutionStrategy1p1,
	EvolutionStrategyMp1,
	EvolutionStrategyMpL,
	EvolutionStrategyML,
	CovarianceMatrixAdaptionEvolutionStrategy,
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class IndividualESTestCase(TestCase):
	def test_init_ok_one(self):
		i = IndividualES()
		self.assertEqual(i.rho, 1.0)

	def test_init_ok_two(self):
		i = IndividualES(rho=10)
		self.assertEqual(i.rho, 10)


class ES1p1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = EvolutionStrategy1p1

	def test_custom_works_fine(self):
		es_custom = self.algo(k=10, c_a=1.5, c_r=0.42, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		es_custom = self.algo(k=10, c_a=1.5, c_r=0.42, seed=self.seed)
		es_customc = self.algo(k=10, c_a=1.5, c_r=0.42, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es_custom, es_customc, Sphere())


class ESMp1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = EvolutionStrategyMp1

	def test_custom_works_fine(self):
		es_custom = self.algo(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		es_custom = self.algo(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es_customc = self.algo(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es_custom, es_customc, Sphere())


class ESMpLTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = EvolutionStrategyMpL

	def test_typeParametes(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['lam'](10))
		self.assertFalse(d['lam'](10.10))
		self.assertFalse(d['lam'](0))
		self.assertFalse(d['lam'](-10))

	def test_custom_works_fine(self):
		es_custom = self.algo(mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		es_custom = self.algo(mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es_customc = self.algo(mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es_custom, es_customc, Sphere())

	def test_custom1_works_fine(self):
		es1_custom = self.algo(mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es1_custom, Sphere())

	def test_custom1_works_fine_parallel(self):
		es1_custom = self.algo(mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es1_customc = self.algo(mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es1_custom, es1_customc, Sphere())


class ESMLTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = EvolutionStrategyML

	def test_custom_works_fine(self):
		es_custom = self.algo(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		es_custom = self.algo(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es_customc = self.algo(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es_custom, es_customc, Sphere())

	def test_custom1_works_fine(self):
		es1_custom = self.algo(mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es1_custom, Sphere())

	def test_custom1_works_fine_parallel(self):
		es1_custom = self.algo(mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es1_customc = self.algo(mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es1_custom, es1_customc, Sphere())


class CMAESTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CovarianceMatrixAdaptionEvolutionStrategy

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['epsilon'](0.234))
		self.assertFalse(d['epsilon'](-0.234))
		self.assertFalse(d['epsilon'](10000.234))
		self.assertFalse(d['epsilon'](10))

	def test_custom_works_fine(self):
		es_custom = self.algo(seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, es_custom, Sphere())

	def test_custom1_works_fine(self):
		es1_custom = self.algo(seed=self.seed)
		es1_customc = self.algo(seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, es1_custom, es1_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
