# encoding=utf8

from WeOptPy.algorithms import AdaptiveBatAlgorithm
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class ABATestCase(AlgorithmTestCase):
	r"""Test case for AdaptiveBatAlgorithm algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`WeOptPy.algorithms.AdaptiveBatAlgorithm`
	"""
	def test_algorithm_info(self):
		"""Test algorithm info method of class AdaptiveBatAlgorithm."""
		self.assertIsNotNone(AdaptiveBatAlgorithm.algorithm_info())

	def test_type_parameters(self):
		"""Test type parameters method of class AdaptiveBatAlgorithm."""
		d = AdaptiveBatAlgorithm.type_parameters()
		# Test epsilon parameter check
		self.assertIsNotNone(d.get('epsilon', None))
		self.assertFalse(d['epsilon'](-100))
		self.assertFalse(d['epsilon'](-.3))
		self.assertTrue(d['epsilon'](3))
		self.assertTrue(d['epsilon'](.3))
		self.assertTrue(d['epsilon'](300))
		# Test alpha parameter check
		self.assertIsNotNone(d.get('alpha', None))
		self.assertFalse(d['alpha'](-100))
		self.assertFalse(d['alpha'](-.3))
		self.assertTrue(d['alpha'](3))
		self.assertTrue(d['alpha'](.3))
		self.assertTrue(d['alpha'](300))
		# Test r parameter check
		self.assertIsNotNone(d.get('r', None))
		self.assertFalse(d['r'](-100))
		self.assertFalse(d['r'](-.3))
		self.assertTrue(d['r'](3))
		self.assertTrue(d['r'](.3))
		self.assertTrue(d['r'](300))
		# Test Qmin parameter check
		self.assertIsNotNone(d.get('Qmin', None))
		self.assertTrue(d['Qmin'](3))
		# Test Qmax parameter check
		self.assertIsNotNone(d.get('Qmax', None))
		self.assertTrue(d['Qmax'](300))

	def test_custom_works_fine(self):
		aba_custom = AdaptiveBatAlgorithm(n=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aba_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		aba_custom = AdaptiveBatAlgorithm(n=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0, seed=self.seed)
		aba_customc = AdaptiveBatAlgorithm(n=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, aba_custom, aba_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
