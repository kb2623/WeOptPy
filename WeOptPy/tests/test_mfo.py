# encoding=utf8

"""Moth flame optimizer algorithm test case module."""

from WeOptPy.algorithms import MothFlameOptimizer
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class MFOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MothFlameOptimizer

	def test_custom_works_fine(self):
		mfo_custom = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mfo_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		mfo_custom = self.algo(NP=20, seed=self.seed)
		mfo_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mfo_custom, mfo_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
