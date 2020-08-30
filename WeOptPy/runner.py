# encoding=utf8

"""Implementation of Runner utility class."""

import datetime
import json
import os
import logging

import xlsxwriter

from WeOptPy.task import (
	StoppingTask,
	OptimizationType
)
from WeOptPy.factory import Factory

logging.basicConfig()
logger = logging.getLogger('WeOptPy.runner.Runner')
logger.setLevel('INFO')

__all__ = ["Runner"]


class Runner:
	r"""Runner utility feature.

	Feature which enables running multiple algorithms with multiple benchmarks.
	It also support exporting results in various formats (e.g. LaTeX, Excel, JSON)

	Attributes:
		d (int): Dimension of problem
		NP (int): Population size
		no_fes (int): Number of function evaluations
		no_runs (int): Number of repetitions
		algorithms (Union[List[str], List[Algorithm]]): List of algorithms to run

	Returns:
		results (Dict[str, Dict]): Returns the results.

	"""
	def __init__(self, d=10, no_fes=1000000, no_runs=1, algorithms='ArtificialBeeColonyAlgorithm', *args, **kwargs):
		r"""Initialize Runner.

		Args:
			d (int): Dimension of problem
			no_fes (int): Number of function evaluations
			no_runs (int): Number of repetitions
			algorithms (List[Algorithm]): List of algorithms to run
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.
		"""
		self.D = d
		self.nFES = no_fes
		self.nRuns = no_runs
		self.useAlgorithms = algorithms
		self.results = {}
		self.factory = Factory()

	def benchmark_factory(self, name):
		r"""Create optimization task.

		Args:
			name (str): Benchmark name.

		Returns:
			Task: Optimization task to use.
		"""
		return StoppingTask(D=self.D, no_fes=self.nFES, optType=OptimizationType.MINIMIZATION, benchmark=name)

	@classmethod
	def __create_export_dir(cls):
		r"""Create export directory if not already createed."""
		if not os.path.exists("export"):
			os.makedirs("export")

	@classmethod
	def __generate_export_name(cls, extension):
		r"""Generate export file name.

		Args:
			extension (str): File format.

		Returns:
			str:
		"""
		return "export/" + str(datetime.datetime.now()).replace(":", ".") + "." + extension

	def __export_to_log(self):
		r"""Print the results to terminal."""
		print(self.results)

	def __export_to_json(self):
		r"""Export the results in the JSON form.

		See Also:
			* :func:`NiaPy.Runner.__createExportDir`
		"""
		self.__create_export_dir()
		with open(self.__generate_export_name("json"), "w") as outFile:
			json.dump(self.results, outFile)
			logger.info("Export to JSON completed!")

	def __export_to_xlsx(self):
		r"""Export the results in the xlsx form.

		See Also:
			* :func:`NiaPy.Runner.__generateExportName`
		"""
		self.__create_export_dir()
		workbook = xlsxwriter.Workbook(self.__generate_export_name("xlsx"))
		worksheet = workbook.add_worksheet()
		row, col, nRuns = 0, 0, 0

		for alg in self.results:
			_, col = worksheet.write(row, col, alg), col + 1
			for bench in self.results[alg]:
				worksheet.write(row, col, bench)
				nRuns = len(self.results[alg][bench])
				for i in range(len(self.results[alg][bench])):
					_, row = worksheet.write(row, col, self.results[alg][bench][i]), row + 1
				row, col = row - len(self.results[alg][bench]), col + 1
			row, col = row + 1 + nRuns, col - 1 + len(self.results[alg])

		workbook.close()
		logger.info("Export to XLSX completed!")

	def run(self, export="log", verbose=False):
		"""Execute runner.

		Args:
			export (str): Takes export type (e.g. log, json, xlsx, latex) (default: "log")
			verbose (bool): Switch for verbose logging (default: {False})

		Raises:
			TypeError: Raises TypeError if export type is not supported

		Returns:
			dict: Returns dictionary of results

		See Also:
			* :func:`NiaPy.Runner.algorithms`
			* :func:`NiaPy.Runner.useBenchmarks`
			* :func:`NiaPy.Runner.__algorithmFactory`
		"""
		for alg in self.useAlgorithms:
			if not isinstance(alg, "".__class__):
				alg_name = str(type(alg).__name__)
			else:
				alg_name = alg

			self.results[alg_name] = {}

			if verbose:
				logger.info("Running %s...", alg_name)

			for bench in self.useBenchmarks:
				if not isinstance(bench, "".__class__):
					bench_name = str(type(bench).__name__)
				else:
					bench_name = bench

				if verbose:
					logger.info("Running %s algorithm on %s benchmark...", alg_name, bench_name)

				self.results[alg_name][bench_name] = []
				for _ in range(self.nRuns):
					algorithm = self.factory.get_algorithm(alg)
					benchmark_stopping_task = self.benchmark_factory(bench)
					self.results[alg_name][bench_name].append(algorithm.run(benchmark_stopping_task))
			if verbose:
				logger.info("---------------------------------------------------")
		if export == "log":
			self.__export_to_log()
		elif export == "json":
			self.__export_to_json()
		elif export == "xlsx":
			self.__export_to_xlsx()
		elif export == "latex":
			self.__export_to_latex()
		else:
			raise TypeError("Passed export type is not supported!")
		return self.results


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
