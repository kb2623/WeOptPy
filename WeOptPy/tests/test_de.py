# encoding=utf8

"""Differential algorithm test case module."""

from WeOptPy.algorithms import (
	DifferentialEvolution,
	DynNpDifferentialEvolution,
	AgingNpDifferentialEvolution,
	MultiStrategyDifferentialEvolution,
	DynNpMultiStrategyDifferentialEvolution,
	AgingNpMultiMutationDifferentialEvolution
)
from WeOptPy.algorithms.de import (
	cross_rand1,
	cross_rand2,
	cross_best1,
	cross_best2,
	cross_curr2rand1,
	cross_curr2best1
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class DETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = self.algo(F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		de_custom = self.algo(F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_custom, de_customc, Sphere())

	def test_CrossRand1(self):
		de_rand1 = self.algo(CrossMutt=cross_rand1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_rand1, Sphere())

	def test_CrossRand1_parallel(self):
		de_rand1 = self.algo(CrossMutt=cross_rand1, seed=self.seed)
		de_rand1c = self.algo(CrossMutt=cross_rand1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_rand1, de_rand1c, Sphere())

	def test_CrossBest1(self):
		de_best1 = self.algo(CrossMutt=cross_best1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_best1, Sphere())

	def test_CrossBest1_parallel(self):
		de_best1 = self.algo(CrossMutt=cross_best1, seed=self.seed)
		de_best1c = self.algo(CrossMutt=cross_best1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_best1, de_best1c, Sphere())

	def test_CrossRand2(self):
		de_rand2 = self.algo(CrossMutt=cross_rand2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_rand2, Sphere())

	def test_CrossRand2_parallel(self):
		de_rand2 = self.algo(CrossMutt=cross_rand2, seed=self.seed)
		de_rand2c = self.algo(CrossMutt=cross_rand2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_rand2, de_rand2c, Sphere())

	def test_CrossBest2(self):
		de_best2 = self.algo(CrossMutt=cross_best2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_best2, Sphere())

	def test_CrossBest2_parallel(self):
		de_best2 = self.algo(CrossMutt=cross_best2, seed=self.seed)
		de_best2c = self.algo(CrossMutt=cross_best2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_best2, de_best2c, Sphere())

	def test_CrossCurr2Rand1(self):
		de_curr2rand1 = self.algo(CrossMutt=cross_curr2rand1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_curr2rand1, Sphere())

	def test_CrossCurr2Rand1_parallel(self):
		de_curr2rand1 = self.algo(CrossMutt=cross_curr2rand1, seed=self.seed)
		de_curr2rand1c = self.algo(CrossMutt=cross_curr2rand1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_curr2rand1, de_curr2rand1c, Sphere())

	def test_CrossCurr2Best1(self):
		de_curr2best1 = self.algo(CrossMutt=cross_curr2best1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_curr2best1, Sphere())

	def test_CrossCurr2Best1_parallel(self):
		de_curr2best1 = self.algo(CrossMutt=cross_curr2best1, seed=self.seed)
		de_curr2best1c = self.algo(CrossMutt=cross_curr2best1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_curr2best1, de_curr2best1c, Sphere())


class DynNpDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpDifferentialEvolution

	def test_typeParameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_Custom_works_fine(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_custom, de_customc, Sphere())


class ANpDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AgingNpDifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_custom, de_customc, Sphere())


class MsDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultiStrategyDifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		de_custom = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_custom, de_customc, Sphere())


class DynNpMsDeTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpMultiStrategyDifferentialEvolution

	def test_typeParameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_Custom_works_fine(self):
		de_custom = self.algo(n=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		de_custom = self.algo(n=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(n=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_custom, de_customc, Sphere())


class ANpMsDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AgingNpMultiMutationDifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = self.algo(n=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		de_custom = self.algo(n=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(n=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, de_custom, de_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
