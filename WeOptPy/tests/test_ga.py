# encoding=utf8

"""Genetic algorithm test case module."""

from WeOptPy.algorithms import GeneticAlgorithm
from WeOptPy.algorithms.ga import (
	two_point_crossover,
	multi_point_crossover,
	creep_mutation,
	roulette_selection,
	crossover_uros,
	mutation_uros
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class GATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GeneticAlgorithm

	def test_custom_works_fine(self):
		ga_custom = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_custom, Sphere())

	def test_Custom_works_fine_parallel(self):
		ga_custom = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		ga_customc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_custom, ga_customc, Sphere())

	def test_two_point_crossover_fine_c(self):
		ga_tpcr = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=two_point_crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_tpcr, Sphere())

	def test_two_point_crossover_fine_c_parallel(self):
		ga_tpcr = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=two_point_crossover, seed=self.seed)
		ga_tpcrc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=two_point_crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_tpcr, ga_tpcrc, Sphere())

	def test_multi_point_crossover_fine_c(self):
		ga_mpcr = self.algo(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=multi_point_crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_mpcr, Sphere())

	def test_multi_point_crossover_fine_c_parallel(self):
		ga_mpcr = self.algo(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=multi_point_crossover, seed=self.seed)
		ga_mpcrc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=multi_point_crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_mpcr, ga_mpcrc, Sphere())

	def test_creep_mutation_fine_c(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=creep_mutation, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_crmt, Sphere())

	def test_creep_mutation_fine_c_parallel(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=creep_mutation, seed=self.seed)
		ga_crmtc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=creep_mutation, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_crmt, ga_crmtc, Sphere())

	def test_reulete_selection_c(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=roulette_selection, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_crmt, Sphere())

	def test_reulete_selection_c_parallel(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=roulette_selection, seed=self.seed)
		ga_crmtc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=roulette_selection, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_crmt, ga_crmtc, Sphere())

	def test_crossover_urso_c(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=crossover_uros, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_crmt, Sphere())

	def test_crossover_urso_c_parallel(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=crossover_uros, seed=self.seed)
		ga_crmtc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=crossover_uros, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_crmt, ga_crmtc, Sphere())

	def test_mutation_urso_c(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=mutation_uros, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ga_crmt, Sphere())

	def test_mutation_urso_c_parallel(self):
		ga_crmt = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=mutation_uros, seed=self.seed)
		ga_crmtc = self.algo(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=mutation_uros, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, ga_crmt, ga_crmtc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
