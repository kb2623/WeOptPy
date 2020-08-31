# encoding=utf8

"""Runner test case module."""

from unittest import TestCase

import WeOptPy
from WeOptPy.task.interfaces import UtilityFunction


class Sphere(UtilityFunction):
	def __init__(self):
		UtilityFunction.__init__(self, -11, 11)

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate


class Special(UtilityFunction):
	def __init__(self):
		UtilityFunction.__init__(self, -11, 11)

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val ** 2 + sol[i] / sol[i]
			return val
		return evaluate


class RunnerTestCase(TestCase):
	def setUp(self):
		self.algorithms = ['DifferentialEvolution', 'GreyWolfOptimizer', 'GeneticAlgorithm', 'ParticleSwarmAlgorithm', 'HybridBatAlgorithm', 'SelfAdaptiveDifferentialEvolution', 'CamelAlgorithm', 'BareBonesFireworksAlgorithm', 'MonkeyKingEvolutionV1', 'MonkeyKingEvolutionV2', 'MonkeyKingEvolutionV3', 'EvolutionStrategy1p1', 'EvolutionStrategyMp1', 'SineCosineAlgorithm', 'GlowwormSwarmOptimization', 'GlowwormSwarmOptimizationV1', 'GlowwormSwarmOptimizationV2', 'GlowwormSwarmOptimizationV3', 'KrillHerdV1', 'KrillHerdV2', 'KrillHerdV3', 'KrillHerdV4', 'KrillHerdV11', 'HarmonySearch', 'HarmonySearchV1', 'FireworksAlgorithm', 'EnhancedFireworksAlgorithm', 'DynamicFireworksAlgorithm', 'MultipleTrajectorySearch', 'MultipleTrajectorySearchV1', 'NelderMeadMethod', 'HillClimbAlgorithm', 'SimulatedAnnealing', 'GravitationalSearchAlgorithm', 'AnarchicSocietyOptimization']
		self.benchmarks = [Sphere(), Special()]

	def test_runner_works_fine(self):
		self.assertTrue(WeOptPy.Runner(7, 100, 2, self.algorithms, self.benchmarks).run())

	def test_runner_bad_algorithm_throws_fine(self):
		self.assertRaises(TypeError, lambda: WeOptPy.Runner(4, 10, 3, 'EvolutionStrategy', self.benchmarks).run())

	def test_runner_bad_benchmark_throws_fine(self):
		self.assertRaises(TypeError, lambda: WeOptPy.Runner(4, 10, 3, 'EvolutionStrategy1p1', 'TesterMan').run())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
