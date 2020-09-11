# encoding=utf8

"""Self adaptive differential evolution test case module."""

from unittest import TestCase, skip

from numpy import random as rnd

from WeOptPy.task.interfaces import Task
from WeOptPy.algorithms import (
	SelfAdaptiveDifferentialEvolution,
	MultiStrategySelfAdaptiveDifferentialEvolution,
	DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
	DynNpMultiStrategySelfAdaptiveDifferentialEvolution
)
from WeOptPy.algorithms.jde import SolutionjDE
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class SolutionjDETestCase(TestCase):
	def setUp(self):
		self.D, self.F, self.CR = 10, 0.9, 0.3
		self.x, self.task = rnd.uniform(10, 50, self.D), Task(self.D, nFES=230, nGEN=None, benchmark=Sphere())
		self.s1, self.s2 = SolutionjDE(task=self.task, e=False), SolutionjDE(x=self.x, CR=self.CR, F=self.F)

	def test_F_fine(self):
		self.assertAlmostEqual(self.s1.F, 2)
		self.assertAlmostEqual(self.s2.F, self.F)

	def test_cr_fine(self):
		self.assertAlmostEqual(self.s1.CR, 0.5)
		self.assertAlmostEqual(self.s2.CR, self.CR)


class jDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = SelfAdaptiveDifferentialEvolution

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['F_l'](10))
		self.assertFalse(d['F_l'](-10))
		self.assertFalse(d['F_l'](-0))
		self.assertTrue(d['F_u'](10))
		self.assertFalse(d['F_u'](-10))
		self.assertFalse(d['F_u'](-0))
		self.assertTrue(d['Tao1'](0.32))
		self.assertFalse(d['Tao1'](-1.123))
		self.assertFalse(d['Tao1'](1.123))
		self.assertTrue(d['Tao2'](0.32))
		self.assertFalse(d['Tao2'](-1.123))
		self.assertFalse(d['Tao2'](1.123))

	def test_custom_works_fine(self):
		jde_custom = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		jde_custom = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_customc = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, jde_custom, jde_customc, Sphere())


class DyNPjDeTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm

	def test_typeParameters(self):
		d = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm.type_parameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	@skip("Not working")
	def test_custom_works_fine(self):
		dynnpjde_custom = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, dynnpjde_custom, Sphere())

	@skip("Not working")
	def test_custom_works_fine_parallel(self):
		dynnpjde_custom = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		dynnpjde_customc = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, dynnpjde_custom, dynnpjde_customc, Sphere())


class MsjDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultiStrategySelfAdaptiveDifferentialEvolution

	def test_custom_works_fine(self):
		jde_custom = self.algo(n=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		jde_custom = self.algo(n=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_customc = self.algo(n=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, jde_custom, jde_customc, Sphere())


class DynNpMsjDeTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpMultiStrategySelfAdaptiveDifferentialEvolution

	@skip("Not working")
	def test_custom_works_fine(self):
		jde_custom = self.aglo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_custom, Sphere())

	@skip("Not working")
	def test_custom_works_fine_parallel(self):
		jde_custom = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_customc = self.algo(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, jde_custom, jde_customc, Sphere())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
