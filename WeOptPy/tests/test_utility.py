# encoding=utf8

from unittest import TestCase

import numpy as np
from numpy import random as rnd

from WeOptPy.util import (
	fullArray,
	limit_repair,
	limitInversRepair,
	wangRepair,
	randRepair,
	reflectRepair,
	FesException,
	GenException
	# TimeException,
	# RefException
)
from WeOptPy.task import (
	StoppingTask,
	ThrowingTask
)


class FullArrayTestCase(TestCase):
	def test_a_float_fine(self):
		A = fullArray(25.25, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

	def test_a_int_fine(self):
		A = fullArray(25, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25)))

	def test_a_float_list_fine(self):
		a = [25.25 for i in range(10)]
		A = fullArray(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

	def test_a_int_list_fine(self):
		a = [25 for i in range(10)]
		A = fullArray(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25)))

	def test_a_float_array_fine(self):
		a = np.asarray([25.25 for i in range(10)])
		A = fullArray(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

	def test_a_int_array_fine(self):
		a = np.asarray([25 for i in range(10)])
		A = fullArray(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25)))

	def test_a_float_list1_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(a, 15)
		a.extend([25.25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_list1_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(a, 15)
		a.extend([25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_array1_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(np.asarray(a), 15)
		a.extend([25.25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_array1_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(np.asarray(a), 15)
		a.extend([25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_list2_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(a, 13)
		a.extend([25.25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_list2_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(a, 13)
		a.extend([25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_array2_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(np.asarray(a), 13)
		a.extend([25.25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_array2_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(np.asarray(a), 13)
		a.extend([25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_list3_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(a, 9)
		a.remove(34.25)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_list3_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(a, 9)
		a.remove(34)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_array3_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(np.asarray(a), 9)
		a.remove(34.25)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_array3_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(np.asarray(a), 9)
		a.remove(34)
		self.assertTrue(np.array_equal(A, np.asarray(a)))


class StoppingTaskBaseTestCase(TestCase):
	def setUp(self):
		self.D = 6
		self.Lower, self.Upper = [2, 1, 1], [10, 10, 2]
		self.task = StoppingTask(Lower=self.Lower, Upper=self.Upper, D=self.D)

	def test_dim_ok(self):
		self.assertEqual(self.D, self.task.D)
		self.assertEqual(self.D, self.task.dim())

	def test_lower(self):
		self.assertTrue(np.array_equal(fullArray(self.Lower, self.D), self.task.Lower))
		self.assertTrue(np.array_equal(fullArray(self.Lower, self.D), self.task.lower()))

	def test_upper(self):
		self.assertTrue(np.array_equal(fullArray(self.Upper, self.D), self.task.Upper))
		self.assertTrue(np.array_equal(fullArray(self.Upper, self.D), self.task.upper()))

	def test_range(self):
		self.assertTrue(np.array_equal(fullArray(self.Upper, self.D) - fullArray(self.Lower, self.D), self.task.bRange))
		self.assertTrue(np.array_equal(fullArray(self.Upper, self.D) - fullArray(self.Lower, self.D), self.task.range()))

	def test_ngens(self):
		self.assertEqual(np.inf, self.task.nGEN)

	def test_nfess(self):
		self.assertEqual(np.inf, self.task.nFES)

	def test_stop_cond(self):
		self.assertFalse(self.task.stop_cond())

	def test_stop_condi(self):
		self.assertFalse(self.task.stop_cond_i())

	def test_eval(self):
		self.assertRaises(AttributeError, lambda: self.task.eval(np.asarray([])))

	def test_evals(self):
		self.assertEqual(0, self.task.evals())

	def test_iters(self):
		self.assertEqual(0, self.task.iters())

	def test_next_iter(self):
		self.assertEqual(None, self.task.next_iteration())

	def test_is_feasible(self):
		self.assertFalse(self.task.is_feasible(fullArray([1, 2, 3], self.D)))


class MyBenchmark(Benchmark):
	def __init__(self):
		Benchmark.__init__(self, -10, 10)

	def function(self):
		def evaluate(D, x): return np.sum(x ** 2)
		return evaluate


class StoppingTaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = StoppingTask(D=self.D, no_fes=self.nFES, no_gen=self.nGEN, benchmark=MyBenchmark())

	def test_isFeasible_fine(self):
		x = np.full(self.D, 10)
		self.assertTrue(self.t.is_feasible(x))
		x = np.full(self.D, -10)
		self.assertTrue(self.t.is_feasible(x))
		x = rnd.uniform(-10, 10, self.D)
		self.assertTrue(self.t.is_feasible(x))
		x = np.full(self.D, -20)
		self.assertFalse(self.t.is_feasible(x))
		x = np.full(self.D, 20)
		self.assertFalse(self.t.is_feasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.stop_cond())
			self.t.next_iteration()
		self.assertTrue(self.t.stop_cond())

	def test_stopCondI(self):
		for i in range(self.nGEN): self.assertFalse(self.t.stop_cond_i(), msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.stop_cond_i())

	def test_eval_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), 0.0, msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.stop_cond())

	def test_eval_over_nFES_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES): self.t.eval(x)
		self.assertEqual(np.inf, self.t.eval(x))
		self.assertTrue(self.t.stop_cond())

	def test_eval_over_nGEN_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nGEN): self.t.next_iteration()
		self.assertEqual(np.inf, self.t.eval(x))
		self.assertTrue(self.t.stop_cond())

	def test_nFES_count_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nGEN):
			self.t.next_iteration()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stop_cond())
		self.t.eval(x)
		self.assertTrue(self.t.stop_cond())

	def test_stopCond_iters_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nGEN - 1):
			self.t.next_iteration()
			self.assertFalse(self.t.stop_cond())
		self.t.next_iteration()
		self.assertTrue(self.t.stop_cond())


class ThrowingTaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = ThrowingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_isFeasible_fine(self):
		x = np.full(self.D, 10)
		self.assertTrue(self.t.is_feasible(x))
		x = np.full(self.D, -10)
		self.assertTrue(self.t.is_feasible(x))
		x = rnd.uniform(-10, 10, self.D)
		self.assertTrue(self.t.is_feasible(x))
		x = np.full(self.D, -20)
		self.assertFalse(self.t.is_feasible(x))
		x = np.full(self.D, 20)
		self.assertFalse(self.t.is_feasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.stop_cond())
			self.t.next_iteration()
		self.assertTrue(self.t.stop_cond())

	def test_stopCondI(self):
		for i in range(self.nGEN): self.assertFalse(self.t.stop_cond_i())
		self.assertTrue(self.t.stop_cond_i())

	def test_eval_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), 0.0, msg='Error at %s iteration!!!' % (i))
		self.assertRaises(FesException, lambda: self.t.eval(x))

	def test_eval_over_nFES_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES):
			self.t.eval(x)
		self.assertRaises(FesException, lambda: self.t.eval(x))

	def test_eval_over_nGEN_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nGEN): self.t.next_iteration()
		self.assertRaises(GenException, lambda: self.t.eval(x))

	def test_nFES_count_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nGEN):
			self.t.next_iteration()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stop_cond())
		self.t.eval(x)
		self.assertTrue(self.t.stop_cond())

	def test_stopCond_iters_fine(self):
		x = np.full(self.D, 0.0)
		for i in range(self.nGEN - 1):
			self.t.next_iteration()
			self.assertFalse(self.t.stop_cond())
		self.t.next_iteration()
		self.assertTrue(self.t.stop_cond())


class LimitRepairTestCase(TestCase):
	def setUp(self):
		self.D = 10
		self.Upper, self.Lower = fullArray(10, self.D), fullArray(-10, self.D)
		self.met = limit_repair

	def generateIndividual(self, D, upper, lower):
		u, l = fullArray(upper, D), fullArray(lower, D)
		return l + rnd.rand(D) * (u - l)

	def test_limit_repair_good_solution_fine(self):
		x = self.generateIndividual(self.D, self.Upper, self.Lower)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())

	def test_limit_repair_bad_upper_solution_fine(self):
		x = self.generateIndividual(self.D, 12, 11)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())

	def test_limit_repair_bad_lower_soluiton_fine(self):
		x = self.generateIndividual(self.D, -11, -12)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())

	def test_limit_repair_bad_upper_lower_soluiton_fine(self):
		x = self.generateIndividual(self.D, 100, -100)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())


class LimitInverseRepairTestCase(LimitRepairTestCase):
	def setUp(self):
		LimitRepairTestCase.setUp(self)
		self.met = limitInversRepair


class WangRepairTestCase(LimitRepairTestCase):
	def setUp(self):
		LimitRepairTestCase.setUp(self)
		self.met = wangRepair


class RandRepairTestCase(LimitRepairTestCase):
	def setUp(self):
		LimitRepairTestCase.setUp(self)
		self.met = randRepair


class ReflectRepairTestCase(LimitRepairTestCase):
	def setUp(self):
		LimitRepairTestCase.setUp(self)
		self.met = reflectRepair


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
