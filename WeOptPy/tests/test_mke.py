# encoding=utf8

"""Monkey king evolution algorithm test case module."""

from unittest import TestCase

import numpy as np
from numpy import random as rnd

from WeOptPy.task.interfaces import Task
from WeOptPy.algorithms import (
	MonkeyKingEvolutionV1,
	MonkeyKingEvolutionV2,
	MonkeyKingEvolutionV3
)
from WeOptPy.algorithms.mke import MkeSolution
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class MkeSolutionTestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-2, 2, self.D), Task(self.D, nGEN=230, nFES=np.inf, benchmark=Sphere())
		self.sol1, self.sol2, self.sol3 = MkeSolution(x=self.x, e=False), MkeSolution(task=self.task), MkeSolution(x=self.x, e=False)

	def test_uPersonalBest_fine(self):
		self.sol2.update_personal_best()
		self.assertTrue(np.array_equal(self.sol2.x, self.sol2.x_pb))
		self.assertEqual(self.sol2.f_pb, self.sol2.f)
		self.sol3.evaluate(self.task)
		self.sol3.x = np.full(self.task.D, -5.11)
		self.sol3.evaluate(self.task)
		self.sol3.update_personal_best()
		self.assertTrue(np.array_equal(self.sol3.x, self.sol3.x_pb))
		self.assertEqual(self.sol3.f_pb, self.sol3.f)


class MKEv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MonkeyKingEvolutionV1

	def test_custom_works_fine(self):
		mke_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mke_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		mke_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mke_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mke_custom, mke_customc, Sphere())


class MKEv2TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MonkeyKingEvolutionV2

	def test_custom_works_fine(self):
		mke_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mke_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		mke_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mke_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mke_custom, mke_customc, Sphere())


class MKEv3TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MonkeyKingEvolutionV3

	def test_custom_works_fine(self):
		mke_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mke_custom, Sphere())

	def test_custom_works_fine_parallel(self):
		mke_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mke_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mke_custom, mke_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
