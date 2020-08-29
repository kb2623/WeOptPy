# encoding=utf8

import logging
from queue import Queue
from threading import Thread
from unittest import TestCase

import numpy as np
from numpy import random as rnd

from WeOptPy.util import objects2array
from WeOptPy.task.interfaces import (
	Task,
	UtilityFunction
)
from WeOptPy.task import StoppingTask
from WeOptPy.algorithms.interfaces import (
	Algorithm,
	Individual
)

logging.basicConfig()
logger = logging.getLogger('NiaPy.test')
logger.setLevel('INFO')


class MyBenchmark(UtilityFunction):
	r"""Testing benchmark class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`
	"""
	def __init__(self):
		UtilityFunction.__init__(self, -5.12, 5.12)

	def function(self):
		return lambda x: np.sum(x ** 2)


class IndividualTestCase(TestCase):
	r"""Test case for testing Individual class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-100, 100, self.D), StoppingTask(d=self.D, no_fes=230, no_gen=np.inf, benchmark=MyBenchmark())
		self.s1, self.s2, self.s3 = Individual(x=self.x, e=False), Individual(task=self.task, rand=rnd), Individual(task=self.task)

	def test_generate_solution_fine(self):
		self.assertTrue(self.task.is_feasible(self.s2))
		self.assertTrue(self.task.is_feasible(self.s3))

	def test_evaluate_fine(self):
		self.s1.evaluate(self.task)
		self.assertAlmostEqual(self.s1.f, self.task.eval(self.x))

	def test_repair_fine(self):
		s = Individual(x=np.full(self.D, 100))
		self.assertFalse(self.task.is_feasible(s.x))

	def test_eq_fine(self):
		self.assertFalse(self.s1 == self.s2)
		self.assertTrue(self.s1 == self.s1)
		s = Individual(x=self.s1.x)
		self.assertTrue(s == self.s1)

	def test_str_fine(self):
		self.assertEqual(str(self.s1), '%s -> %s' % (self.x, np.inf))

	def test_getitem_fine(self):
		for i in range(self.D): self.assertEqual(self.s1[i], self.x[i])

	def test_len_fine(self):
		self.assertEqual(len(self.s1), len(self.x))


def init_pop_numpy(task, NP, **kwargs):
	r"""Custom population initialization function for numpy individual type.

	Args:
		task (Task): Optimization task.
		np (int): Population size.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]):
			1. Initialized population.
			2. Initialized populations fitness/function values.
	"""
	pop = np.full((NP, task.D), 0.0)
	fpop = np.apply_along_axis(task.eval, 1, pop)
	return pop, fpop


def init_pop_individual(task, NP, itype, **kwargs):
	r"""Custom population initialization function for numpy individual type.

	Args:
		task (Task): Optimization task.
		np (int): Population size.
		itype (Individual): Type of individual in population.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]):
			1. Initialized population.
			2. Initialized populations fitness/function values.
	"""
	pop = objects2array([itype(x=np.full(task.D, 0.0), task=task) for _ in range(NP)])
	return pop, np.asarray([x.f for x in pop])


class AlgorithmBaseTestCase(TestCase):
	r"""Test case for testing Algorithm class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	Attributes:
		seed (int): Starting seed of random generator.
		rnd (mtrand.RandomState): Random generator.
		a (Algorithm): Algorithm to use for testing.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def setUp(self):
		self.seed = 1
		self.rnd = rnd.RandomState(self.seed)
		self.a = Algorithm(seed=self.seed)

	def test_algorithm_info_fine(self):
		r"""Check if method works fine."""
		i = Algorithm.algorithm_info()
		self.assertIsNotNone(i)

	def test_algorithm_getParameters_fine(self):
		r"""Check if method works fine."""
		algo = Algorithm()
		params = algo.get_parameters()
		self.assertIsNotNone(params)

	def test_type_parameters_fine(self):
		d = Algorithm.type_parameters()
		self.assertIsNotNone(d)

	def test_init_population_numpy_fine(self):
		r"""Test if custome generation initialization works ok."""
		a = Algorithm(n=10, InitPopFunc=init_pop_numpy)
		t = Task(d=20, benchmark=MyBenchmark())
		self.assertTrue(np.array_equal(np.full((10, t.D), 0.0), a.init_population(t)[0]))

	def test_init_population_individual_fine(self):
		r"""Test if custome generation initialization works ok."""
		a = Algorithm(n=10, InitPopFunc=init_pop_individual, itype=Individual)
		t = Task(d=20, benchmark=MyBenchmark())
		i = Individual(x=np.full(t.D, 0.0), task=t)
		pop, fpop, d = a.init_population(t)
		for e in pop: self.assertEqual(i, e)

	def test_set_parameters(self):
		self.a.set_parameters(t=None, a=20)
		self.assertRaises(AttributeError, lambda: self.assertEqual(self.a.a, None))

	def test_randint_fine(self):
		o = self.a.randint(maximum=20, minimum=10, d=[10, 10])
		self.assertEqual(o.shape, (10, 10))
		self.assertTrue(np.array_equal(self.rnd.randint(10, 20, (10, 10)), o))
		o = self.a.randint(maximum=20, minimum=10, d=(10, 5))
		self.assertEqual(o.shape, (10, 5))
		self.assertTrue(np.array_equal(self.rnd.randint(10, 20, (10, 5)), o))
		o = self.a.randint(maximum=20, minimum=10, d=10)
		self.assertEqual(o.shape, (10,))
		self.assertTrue(np.array_equal(self.rnd.randint(10, 20, 10), o))

	def test_randn_fine(self):
		a = self.a.randn([1, 2])
		self.assertEqual(a.shape, (1, 2))
		self.assertTrue(np.array_equal(self.rnd.randn(1, 2), a))
		a = self.a.randn(1)
		self.assertEqual(len(a), 1)
		self.assertTrue(np.array_equal(self.rnd.randn(1), a))
		a = self.a.randn(2)
		self.assertEqual(len(a), 2)
		self.assertTrue(np.array_equal(self.rnd.randn(2), a))
		a = self.a.randn()
		self.assertIsInstance(a, float)
		self.assertTrue(np.array_equal(self.rnd.randn(), a))

	def test_uniform_fine(self):
		a = self.a.uniform(-10, 10, [10, 10])
		self.assertEqual(a.shape, (10, 10))
		self.assertTrue(np.array_equal(self.rnd.uniform(-10, 10, (10, 10)), a))
		a = self.a.uniform(4, 10, (4, 10))
		self.assertEqual(len(a), 4)
		self.assertEqual(len(a[0]), 10)
		self.assertTrue(np.array_equal(self.rnd.uniform(4, 10, (4, 10)), a))
		a = self.a.uniform(1, 4, 2)
		self.assertEqual(len(a), 2)
		self.assertTrue(np.array_equal(self.rnd.uniform(1, 4, 2), a))
		a = self.a.uniform(10, 100)
		self.assertIsInstance(a, float)
		self.assertEqual(self.rnd.uniform(10, 100), a)

	def test_normal_fine(self):
		a = self.a.normal(-10, 10, [10, 10])
		self.assertEqual(a.shape, (10, 10))
		self.assertTrue(np.array_equal(self.rnd.normal(-10, 10, (10, 10)), a))
		a = self.a.normal(4, 10, (4, 10))
		self.assertEqual(len(a), 4)
		self.assertEqual(len(a[0]), 10)
		self.assertTrue(np.array_equal(self.rnd.normal(4, 10, (4, 10)), a))
		a = self.a.normal(1, 4, 2)
		self.assertEqual(len(a), 2)
		self.assertTrue(np.array_equal(self.rnd.normal(1, 4, 2), a))
		a = self.a.normal(10, 100)
		self.assertIsInstance(a, float)
		self.assertEqual(self.rnd.normal(10, 100), a)


class TestingTask(StoppingTask, TestCase):
	r"""Testing task.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`WeOptPy.task.StoppingTask`
	"""
	def names(self):
		r"""Get names of benchmark.

		Returns:
			List[str]: List of task names.
		"""
		return self.benchmark.Name

	def eval(self, x):
		r"""Check if is algorithm trying to evaluate solution out of bounds."""
		self.assertTrue(self.is_feasible(x), 'Solution %s is not in feasible space!!!' % x)
		return StoppingTask.eval(self, x)


class AlgorithmTestCase(TestCase):
	r"""Base class for testing other algorithms.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	Attributes:
		D (List[int]): Dimension of problem.
		nGEN (int): Number of generations/iterations.
		nFES (int): Number of function evaluations.
		seed (int): Starting seed of random generator.

	See Also:
		* :class:`WeOptPy.algorithms.Algorithm`
	"""
	def setUp(self):
		r"""Setup basic parameters of the algorithm run."""
		self.D, self.nGEN, self.nFES, self.seed = [10, 40], 1000, 1000, 1
		self.algo = Algorithm

	def test_algorithm_type_parameters(self):
		r"""Test if type parametes for algorithm work fine."""
		tparams = self.algo.type_parameters()
		self.assertIsNotNone(tparams)

	def test_algorithm_info_fine(self):
		r"""Test if algorithm info works fine."""
		info = self.algo.algorithm_info()
		self.assertIsNotNone(info)

	def test_algorithm_get_parameters_fine(self):
		r"""Test if algorithms parameters values are fine."""
		params = self.algo().get_parameters()
		self.assertIsNotNone(params)

	def __set_up_task(self, d=10, bech=MyBenchmark, nFES=None, nGEN=None, verbose=False):
		r"""Setup optimization tasks for testing.

		Args:
			d (int): Dimension of the problem.
			bech (UtilityFunction): Optimization problem to use.
			nFES (int): Number of fitness/objective function evaluations.
			nGEN (int): Number of generations.
			verbose (bool): Verbose output.

		Returns:
			Task: Testing task.
		"""
		return TestingTask(d=d, no_fes=self.nFES if nFES is None else nFES, no_gen=self.nGEN if nGEN is None else nGEN, benchmark=bech, verbose=verbose)

	def test_algorithm_run(self, a=None, benc=MyBenchmark):
		r"""Run main testing of algorithm.

		Args:
			a (Algorithm): First instance of algorithm.
			benc (UtilityFunction): Benchmark to use for testing.
		"""
		if a is None: return False
		for D in self.D:
			no_fes, no_gen = D * 1000, D * 1000
			task = self.__set_up_task(D, benc, nFES=no_fes, nGEN=no_gen)
			x = a.run(task)
			self.assertFalse(a.bad_run(), "Something went wrong at runtime of the algorithm --> %s" % a.exception)
			self.assertIsNotNone(x)
			logger.info('%s\n%s -> %s' % (task.names(), x[0], x[1]))
			self.assertAlmostEqual(task.benchmark.function()(x[0].x if isinstance(x[0], Individual) else x[0]), x[1], msg='Best individual fitness values does not mach the given one')
			self.assertAlmostEqual(task.x_f, x[1], msg='While running the algorithm, algorithm got better individual with fitness: %s' % task.x_f)
			self.assertTrue(no_fes >= task.Evals, msg='nfes: %d < evals: %d' % (no_fes, task.Evals))
			self.assertTrue(no_gen >= task.Iters, msg='ngen: %d < iters: %d' % (no_gen, task.Iters))
		return True

	def test_algorithm_run_parallel(self, a=None, b=None, benc=MyBenchmark):
		r"""Run main testing of algorithm in parallel.

		Args:
			a (Algorithm): First instance of algorithm.
			b (Algorithm): Second instance of algorithm.
			benc (UtilityFunction): Benchmark to use for testing.
		"""
		if a is None or b is None: return False
		for D in self.D:
			no_fes, no_gen = D * 1000, D * 1000
			task1, task2 = self.__set_up_task(D, benc, nFES=no_fes, nGEN=no_gen), self.__set_up_task(D, benc, nFES=no_fes, nGEN=no_gen)
			q = Queue(maxsize=2)
			thread1, thread2 = Thread(target=lambda a, t, q: q.put(a.run(t)), args=(a, task1, q)), Thread(target=lambda a, t, q: q.put(a.run(t)), args=(b, task2, q))
			thread1.start(), thread2.start()
			thread1.join(), thread2.join()
			x, y = q.get(block=True), q.get(block=True)
			self.assertFalse(a.bad_run() or b.bad_run(), "Something went wrong at runtime of the algorithm --> %s" % a.exception)
			self.assertIsNotNone(x), self.assertIsNotNone(y)
			logger.info('%s\n%s -> %s\n%s -> %s' % (task1.names(), x[0], x[1], y[0], y[1]))
			self.assertAlmostEqual(task1.benchmark.function()(x[0].x if isinstance(x[0], Individual) else x[0]), x[1], msg='Best individual fitness values does not mach the given one')
			self.assertAlmostEqual(task1.x_f, x[1], msg='While running the algorithm, algorithm got better individual with fitness: %s' % task1.x_f)
			self.assertTrue(np.array_equal(x[0], y[0]), 'Results can not be reproduced, check usages of random number generator')
			self.assertAlmostEqual(x[1], y[1], msg='Results can not be reproduced or bad function value')
			self.assertTrue(no_fes >= task1.Evals, msg='nfes: %s < evals: %s' % (no_fes, task1.Evals)), self.assertEqual(task1.Evals, task2.Evals, msg='task1: %d != task2: %d' % (task1.Evals, task2.Evals))
			self.assertTrue(no_gen >= task1.Iters, msg='ngen: %s < iters: %s' % (no_gen, task1.Iters)), self.assertEqual(task1.Iters, task2.Iters, msg='task1: %d != task2: %d' % (task1.Iters, task2.Iters))
		return True


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
