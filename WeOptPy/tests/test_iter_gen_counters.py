# encoding=utf8

from unittest import TestCase

from WeOptPy.algorithms import (
	DifferentialEvolution,
	BatAlgorithm,
	FireflyAlgorithm
)
from WeOptPy.task import (
	StoppingTask,
	OptimizationType
)


class DETestCase(TestCase):

	r"""Test cases for evaluating different stopping conditions.

		**Date:** November 2018

		**Author:** Iztok Fister

		**Author:** This is a very important test!
	"""

	def test_DE_evals_fine(self):
		task = StoppingTask(
			D=10,
			no_fes=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
		algo.run_task(task)
		evals = task.evals()
		self.assertEqual(1000, evals)

	def test_DE_iters_fine(self):
		task = StoppingTask(
			D=10,
			no_gen=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
		algo.run_task(task)
		iters = task.iters()
		self.assertEqual(1000, iters)


class BATestCase(TestCase):

	r"""Test cases for evaluating different stopping conditions.

		**Date:** November 2018

		**Author:** Iztok Fister

		**Author:** This is a very important test!
	"""

	def test_BA_evals_fine(self):
		task = StoppingTask(
			D=10,
			no_fes=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = BatAlgorithm(NP=25)
		algo.run_task(task)
		evals = task.evals()
		self.assertEqual(1000, evals)

	def test_BA_iters_fine(self):
		task = StoppingTask(
			D=10,
			no_gen=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = BatAlgorithm(NP=25)
		algo.run_task(task)
		iters = task.iters()
		self.assertEqual(1000, iters)

	# 1000 BA iterations spend 10010 FES (10 + 10 * 1000)
	def test_BA_iters_to_fes(self):
		task = StoppingTask(
			D=10,
			no_gen=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = BatAlgorithm(NP=10)
		algo.run_task(task)
		evals = task.evals()
		self.assertEqual(10000, evals)


class FATestCase(TestCase):

	def test_FA_evals_fine(self):
		task = StoppingTask(
			D=10,
			no_fes=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = FireflyAlgorithm(NP=25)
		algo.run_task(task)
		evals = task.evals()
		self.assertEqual(1000, evals)

	def test_FA_iters_fine(self):
		task = StoppingTask(
			D=10,
			no_gen=1000,
			optType=OptimizationType.MINIMIZATION,
			benchmark=Sphere())
		algo = FireflyAlgorithm(NP=25)
		algo.run_task(task)
		iters = task.iters()
		self.assertEqual(1000, iters)


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
