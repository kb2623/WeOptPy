# encoding=utf8

"""Multiple trajectory search algorithm module."""

from numpy import argsort

from WeOptPy.algorithms.interfaces.individual import Individual
from WeOptPy.algorithms import (
	DifferentialEvolution,
	DynNpDifferentialEvolution,
	MultiStrategyDifferentialEvolution
)
from WeOptPy.algorithms.mts import (
	MTS_LS1,
	MTS_LS1v1,
	MTS_LS2,
	MTS_LS3,
	MTS_LS3v1,
	MultipleTrajectorySearch
)

__all__ = [
	'DifferentialEvolutionMTS',
	'DifferentialEvolutionMTSv1',
	'DynNpDifferentialEvolutionMTS',
	'DynNpDifferentialEvolutionMTSv1',
	'MultiStrategyDifferentialEvolutionMTS',
	'MultiStrategyDifferentialEvolutionMTSv1',
	'DynNpMultiStrategyDifferentialEvolutionMTS',
	'DynNpMultiStrategyDifferentialEvolutionMTSv1'
]


class MtsIndividual(Individual):
	r"""Individual for MTS local searches.

	Attributes:
		SR (numpy.ndarray): Search range.
		grade (int): Grade of individual.
		enable (bool): If enabled.
		improved (bool): If improved.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Individual`
	"""
	def __init__(self, SR=None, grade=0, enable=True, improved=False, task=None, **kwargs):
		r"""Initialize the individual.

		Args:
			SR (numpy.ndarray): Search range.
			grade (Optional[int]): Grade of individual.
			enable (Optional[bool]): If enabled individual.
			improved (Optional[bool]): If individual improved.
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Individual.__init__`
		"""
		Individual.__init__(self, task=task, **kwargs)
		self.grade, self.enable, self.improved = grade, enable, improved
		if SR is None and task is not None: self.SR = task.bRange / 4
		else: self.SR = SR


class DifferentialEvolutionMTS(DifferentialEvolution, MultipleTrajectorySearch):
	r"""Implementation of Differential Evolution with MTS local searches.

	Algorithm:
		Differential Evolution withm MTS local searches

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		LSs (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.
		BONUS1 (int): Bonus for improving global best solution.
		BONUS2 (int): Bonus for improving solution.
		NoLsTests (int): Number of test runs on local search algorithms.
		NoLs (int): Number of local search algorithm runs.
		NoEnabled (int): Number of best solution for testing.

	See Also:
		* :class:`NiaPy.algorithms.basic.de.DifferentialEvolution`
		* :class:`NiaPy.algorithms.other.mts.MultipleTrajectorySearch`
	"""
	Name = ['DifferentialEvolutionMTS', 'DEMTS']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* NoLsTests (Callable[[int], bool]): TODO
				* NoLs (Callable[[int], bool]): TODO
				* NoEnabled (Callable[[int], bool]): TODO

		See Also:
			* :func:`NiaPy.algorithms.basic.de.DifferentialEvolution.typeParameters`
		"""
		d = DifferentialEvolution.type_parameters()
		d.update({
			'NoLsTests': lambda x: isinstance(x, int) and x >= 0,
			'NoLs': lambda x: isinstance(x, int) and x >= 0,
			'NoEnabled': lambda x: isinstance(x, int) and x > 0
		})
		return d

	def set_parameters(self, NoLsTests=1, NoLs=2, NoEnabled=2, BONUS1=10, BONUS2=2, LSs=(MTS_LS1, MTS_LS2, MTS_LS3), **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			SR (numpy.ndarray): Search range.

		See Also:
			* :func:`NiaPy.algorithms.basic.de.DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.set_parameters(self, itype=ukwargs.pop('itype', MtsIndividual), **ukwargs)
		self.LSs, self.NoLsTests, self.NoLs, self.NoEnabled = LSs, NoLsTests, NoLs, NoEnabled
		self.BONUS1, self.BONUS2 = BONUS1, BONUS2

	def get_parameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO

		See Also:
			* :func:`WeOptPy.algorithms.interfaces.Algorithm.getParameters`
		"""
		d = DifferentialEvolution.get_parameters(self)
		# TODO add parameter values to dictionary
		return d

	def post_selection(self, X, task, xb, fxb, **kwargs):
		r"""Post selection operator.

		Args:
			X (numpy.ndarray): Current populaiton.
			task (Task): Optimization task.
			xb (numpy.ndarray): Global best individual.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]: New population.
		"""
		for x in X:
			if not x.enable: continue
			x.enable, x.grades = False, 0
			x.x, x.f, xb, fxb, k = self.grading_run(x.x, x.f, xb, fxb, x.improved, x.SR, task)
			x.x, x.f, xb, fxb, x.improved, x.SR, x.grades = self.local_search(k, x.x, x.f, xb, fxb, x.improved, x.SR, 0, task)
		for i in X[argsort([x.grade for x in X])[:self.NoEnabled]]: i.enable = True
		return X, xb, fxb


class DifferentialEvolutionMTSv1(DifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution with MTSv1 local searches.

	Algorithm:
		Differential Evolution with MTSv1 local searches

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.modified.DifferentialEvolutionMTS`
	"""
	Name = ['DifferentialEvolutionMTSv1', 'DEMTSv1']

	def set_parameters(self, **ukwargs):
		r"""Set core parameters of DifferentialEvolutionMTSv1 algorithm.

		Args:
			ukwargs (dict): Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.DifferentialEvolutionMTS.setParameters`
		"""
		DifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **ukwargs)


class DynNpDifferentialEvolutionMTS(DifferentialEvolutionMTS, DynNpDifferentialEvolution):
	r"""Implementation of Differential Evolution with MTS local searches dynamic and population size.

	Algorithm:
		Differential Evolution with MTS local searches and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name

	See Also:
		* :class:`NiaPy.algorithms.modified.DifferentialEvolutionMTS`
		* :class:`NiaPy.algorithms.basic.de.DynNpDifferentialEvolution`
	"""
	Name = ['DynNpDifferentialEvolutionMTS', 'dynNpDEMTS']

	def set_parameters(self, pmax=10, rp=3, **ukwargs):
		r"""Set core parameters or DynNpDifferentialEvolutionMTS algorithm.

		Args:
			pmax (Optional[int]):
			rp (Optional[float]):
			ukwargs (dict): Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms..DifferentialEvolutionMTS.setParameters`
			* :func:`WeOptPy.algorithms.DynNpDifferentialEvolution.setParameters`
		"""
		DynNpDifferentialEvolution.set_parameters(self, pmax=pmax, rp=rp, **ukwargs)
		DifferentialEvolutionMTS.set_parameters(self, **ukwargs)

	def post_selection(self, X, task, xb, fxb, **kwargs):
		r"""Post selection work.

		Args:
			X:
			task:
			xb:
			fxb:
			**kwargs:

		Returns:

		"""
		nX, xb, fxb = DynNpDifferentialEvolution.post_selection(self, X, task, xb, fxb)
		nX, xb, fxb = DifferentialEvolutionMTS.post_selection(self, nX, task, xb, fxb)
		return nX, xb, fxb


class DynNpDifferentialEvolutionMTSv1(DynNpDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution with MTSv1 local searches and dynamic population size.

	Algorithm:
		Differential Evolution with MTSv1 local searches and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`WeOptPy.algorithms.DifferentialEvolutionMTS`
	"""
	Name = ['DynNpDifferentialEvolutionMTSv1', 'dynNpDEMTSv1']

	def set_parameters(self, **ukwargs):
		r"""Set core arguments of DynNpDifferentialEvolutionMTSv1 algorithm.

		Args:
			ukwargs (dict): Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.DifferentialEvolutionMTS.setParameters`
		"""
		DynNpDifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **ukwargs)


class MultiStrategyDifferentialEvolutionMTS(DifferentialEvolutionMTS, MultiStrategyDifferentialEvolution):
	r"""Implementation of Differential Evolution with MTS local searches and multiple mutation strategies.

	Algorithm:
		Differential Evolution with MTS local searches and multiple mutation strategies.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`WeOptPy.algorithms.DifferentialEvolutionMTS`
		* :class:`WeOptPy.algorithms.MultiStrategyDifferentialEvolution`
	"""
	Name = ['MultiStrategyDifferentialEvolutionMTS', 'MSDEMTS']

	def set_parameters(self, **ukwargs):
		r"""TODO.

		Args:
			ukwargs (dict): Additional keywrod arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.DifferentialEvolutionMTS.setParameters`
			* :func:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution.setParameters`
		"""
		DifferentialEvolutionMTS.set_parameters(self, **ukwargs)
		MultiStrategyDifferentialEvolution.set_parameters(self, itype=ukwargs.pop('itype', MtsIndividual), **ukwargs)

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve population.

		Args:
			pop (numpy.ndarray): Current population of individuals.
			xb (Individual): Global best individual.
			task (Task): Optimization task.
			kwargs (dict): Additional keyword arguments.

		Returns:
			numpy.ndarray[Individual]: Evolved population.
		"""
		return MultiStrategyDifferentialEvolution.evolve(self, pop, xb, task, **kwargs)


class MultiStrategyDifferentialEvolutionMTSv1(MultiStrategyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution with MTSv1 local searches and multiple mutation strategies.

	Algorithm:
		Differential Evolution with MTSv1 local searches and multiple mutation strategies.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of stings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS`
	"""
	Name = ['MultiStrategyDifferentialEvolutionMTSv1', 'MSDEMTSv1']

	def set_parameters(self, **ukwargs):
		r"""Set core parameters of MultiStrategyDifferentialEvolutionMTSv1 algorithm.

		Args:
			ukwargs (dict): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS.setParameters`
		"""
		MultiStrategyDifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **ukwargs)


class DynNpMultiStrategyDifferentialEvolutionMTS(MultiStrategyDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution with MTS local searches, multiple mutation strategies and dynamic population size.

	Algorithm:
		Differential Evolution with MTS local searches, multiple mutation strategies and dynamic population size.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name

	See Also:
		* :class:`WeOptPy.algorithms.MultiStrategyDifferentialEvolutionMTS`
		* :class:`WeOptPy.algorithms.DynNpDifferentialEvolutionMTS`
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolutionMTS', 'dynNpMSDEMTS']

	def set_parameters(self, **ukwargs):
		r"""Set core arguments of DynNpMultiStrategyDifferentialEvolutionMTS algorithm.

		Args:
			ukwargs (dict): Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.MultiStrategyDifferentialEvolutionMTS.setParameters`
			* :func:`WeOptPy.algorithms.DynNpDifferentialEvolutionMTS.setParameters`
		"""
		DynNpDifferentialEvolutionMTS.set_parameters(self, **ukwargs)
		MultiStrategyDifferentialEvolutionMTS.set_parameters(self, **ukwargs)


class DynNpMultiStrategyDifferentialEvolutionMTSv1(DynNpMultiStrategyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution with MTSv1 local searches, multiple mutation strategies and dynamic population size.

	Algorithm:
		Differential Evolution with MTSv1 local searches, multiple mutation strategies and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`WeOptPy.algorithm.DynNpMultiStrategyDifferentialEvolutionMTS`
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolutionMTSv1', 'dynNpMSDEMTSv1']

	def set_parameters(self, **kwargs):
		r"""Set core parameters of DynNpMultiStrategyDifferentialEvolutionMTSv1 algorithm.

		Args:
			kwargs (dict): Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithm.DynNpMultiStrategyDifferentialEvolutionMTS.setParameters`
		"""
		DynNpMultiStrategyDifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **kwargs)
