# encoding=utf8

"""Nelder mead method algorithm test case module."""

from WeOptPy.algorithms import NelderMeadMethod
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class NMMTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = NelderMeadMethod

	def test_algorithm_info(self):
		self.assertIsNotNone(NelderMeadMethod.algorithm_info())

	def test_type_parameters(self):
		d = NelderMeadMethod.type_parameters()
		self.assertIsNotNone(d.get('n', None))
		self.assertIsNotNone(d.get('alpha', None))
		self.assertIsNotNone(d.get('gamma', None))
		self.assertIsNotNone(d.get('rho', None))
		self.assertIsNotNone(d.get('sigma', None))

	def test_custom_works_fine(self):
		nmm_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, nmm_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		nmm_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		nmm_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, nmm_custom, nmm_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
