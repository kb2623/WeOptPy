# encoding=utf8

"""Multiple trajectory search algorithm test case module."""

from WeOptPy.algorithms import (
	MultipleTrajectorySearch,
	MultipleTrajectorySearchV1
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class MTSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultipleTrajectorySearch

	def test_type_parameters(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['NoLsTests'](10))
		self.assertTrue(d['NoLsTests'](0))
		self.assertFalse(d['NoLsTests'](-10))
		self.assertTrue(d['NoLs'](10))
		self.assertTrue(d['NoLs'](0))
		self.assertFalse(d['NoLs'](-10))
		self.assertTrue(d['NoLsBest'](10))
		self.assertTrue(d['NoLsBest'](0))
		self.assertFalse(d['NoLsBest'](-10))
		self.assertTrue(d['NoEnabled'](10))
		self.assertFalse(d['NoEnabled'](0))
		self.assertFalse(d['NoEnabled'](-10))

	def test_custom_works_fine(self):
		mts_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mts_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		mts_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mts_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mts_custom, mts_customc, Sphere())


class MTSv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultipleTrajectorySearchV1

	def test_custom_works_fine(self):
		mts_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mts_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		mts_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mts_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mts_custom, mts_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
