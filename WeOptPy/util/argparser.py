# encoding=utf8

"""Argparser class."""

import sys
import logging
from argparse import ArgumentParser

import numpy as np

from WeOptPy.task.optimizationtype import OptimizationType

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.argparse')
logger.setLevel('INFO')

__all__ = [
	'get_args',
	'get_dict_args',
	'make_arg_parser'
]


def optimization_type(x):
	r"""Map function for optimization type.

	Args:
		x (str): String representing optimization type.

	Returns:
		OptimizationType: Optimization type based on type that is defined as enum.
	"""
	if x not in ['min', 'max']: logger.info('You can use only [min, max], using min')
	return OptimizationType.MAXIMIZATION if x == 'max' else OptimizationType.MINIMIZATION


def make_arg_parser():
	r"""Create/Make parser for parsing string.

	Author:
		Klemen Berkovič

	Date:
		2019

	Parser:
		* `-a` or `--algorithm` (str):
			Name of algorithm to use. Default value is `jDE`.
		* `-b` or `--bech` (str):
			Name of benchmark to use. Default values is `Benchmark`.
		* `-d` (int):
			Number of dimensions/components usd by benchmark. Default values is `10`.
		* `-nFES` (int):
			Number of maximum function evaluations. Default values is `inf`.
		* `-nGEN` (int):
			Number of maximum algorithm iterations/generations. Default values is `inf`.
		* `-n` (int):
			Number of individuals in population. Default values is `43`.
		* `-r` or `--runType` (str): Run type of run. Value can be:
			* '': No output during the run. Output is shown only at the end of algorithm run.
			* `log`: Output is shown every time new global best solution is found
			* `plot`: Output is shown only at the end of run. Output is shown as graph ploted in matplotlib. Graph represents convegance of algorithm over run time of algorithm.
			Default value is `''`.
		* `-seed` (list of int or int):
			Set the starting seed of algorithm run. If multiple runs, user can provide list of ints, where each int usd use at new run. Default values is `None`.
		* `-optType` (str):
			Optimization type of the run. Values can be:
				* `min`: For minimization problems
				* `max`: For maximization problems
			Default value is `min`.

	Returns:
		ArgumentParser: Parser for parsing arguments from string.

	See Also:
		* :class:`ArgumentParser`
		* :func:`ArgumentParser.add_argument`
	"""
	parser = ArgumentParser(description='Runer example.')
	parser.add_argument('-a', '--algorithm', dest='algo', default='jDE', type=str)
	parser.add_argument('-d', dest='d', default=10, type=int)
	parser.add_argument('-nFES', dest='nFES', default=np.inf, type=int)
	parser.add_argument('-nGEN', dest='nGEN', default=np.inf, type=int)
	parser.add_argument('-n', dest='n', default=43, type=int)
	parser.add_argument('-r', '--runType', dest='runType', choices=['', 'log', 'plot'], default='', type=str)
	parser.add_argument('-seed', dest='seed', nargs='+', default=[None], type=int)
	parser.add_argument('-optType', dest='optType', default=optimization_type('min'), type=optimization_type)
	return parser


def get_args(av):
	r"""Parse arguments form input string.

	Args:
		av (str): String to parse.

	Returns:
		Dict[str, Union[float, int, str, OptimizationType]]: Where key represents argument name and values it's value.

	See Also:
		* :func:`NiaPy.util.argparser.MakeArgParser`.
		* :func:`ArgumentParser.parse_args`
	"""
	parser = make_arg_parser()
	a = parser.parse_args(av)
	return a


def get_dict_args(argv):
	r"""Parse input string.

	Args:
		argv (str): Input string to parse for argumets

	Returns:
		dict: Parsed input string

	See Also:
		* :func:`NiaPy.utils.getArgs`
	"""
	return vars(get_args(argv))


if __name__ == '__main__':
	r"""Run the algorithms based on parameters from the command line interface."""
	args = get_args(sys.argv[1:])
	logger.info(str(args))

	
# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
