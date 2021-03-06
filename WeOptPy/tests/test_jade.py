# encoding=utf8

"""Adaptive archive different evolution test case module."""

from unittest import TestCase, skip

import numpy as np

from WeOptPy.util import full_array
from WeOptPy.algorithms.jade import (
	AdaptiveArchiveDifferentialEvolution,
	CrossRandCurr2Pbest
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class CrossRandCurr2pbestTestCase(TestCase):
	def setUp(self):
		self.D, self.NP, self.F, self.CR, self.p = 10, 100, 0.5, 0.5, 0.25
		self.Upper, self.Lower = full_array(100, self.D), full_array(-100, self.D)
		self.evalFun = Sphere().function()

	def init_pop(self):
		pop = self.Lower + np.random.rand(self.NP, self.D) * (self.Upper - self.Lower)
		return pop, np.asarray([self.evalFun(self.D, x) for x in pop])

	@skip('Needs fixing!!!')
	def test_function_fine(self):
		pop, fpop = self.init_pop()
		apop, _ = self.init_pop()
		ib = np.argmin(fpop)
		xb = pop[ib].copy()
		for i, x in enumerate(pop):
			xn = CrossRandCurr2Pbest(pop, i, xb, self.F, self.CR, self.p, apop)
			self.assertFalse(np.array_equal(x, xn))


class JADETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AdaptiveArchiveDifferentialEvolution

	@skip('Not implemented jet!!!')
	def test_custom_works_fine(self):
		jade_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jade_custom, Sphere())

	@skip('Not implemented jet!!!')
	def test_custom_works_fine_parallel(self):
		jade_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		jade_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, jade_custom, jade_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
