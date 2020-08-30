# encoding=utf8

import numpy as np
from numpy import random as rand

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.algorithms.interfaces.individual import Individual
from WeOptPy.util.utility import objects2array

__all__ = ['CamelAlgorithm']


class Camel(Individual):
	r"""Implementation of population individual that is a camel for Camel algorithm.

	Algorithm:
		Camel algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		E (float): Camel endurance.
		S (float): Camel supply.
		x_past (numpy.ndarray): Camel's past position.
		f_past (float): Camel's past funciton/fitness value.
		steps (int): Age of camel.

	See Also:
		* :class:`WeOptPy.algorithms.Individual`
	"""
	def __init__(self, E_init=None, S_init=None, **kwargs):
		r"""Initialize the Camel.

		Args:
			E_init (Optional[float]): Starting endurance of Camel.
			S_init (Optional[float]): Stating supply of Camel.
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`WeOptPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		self.E, self.E_past = E_init, E_init
		self.S, self.S_past = S_init, S_init
		self.x_past, self.f_past = self.x, self.f
		self.steps = 0

	def nextt(self, T_min, T_max, rnd=rand):
		r"""Apply nextT function on Camel.

		Args:
			T_min (float): TODO
			T_max (float): TODO
			rnd (Optional[mtrand.RandomState]): Random number generator.
		"""
		self.T = (T_max - T_min) * rnd.rand() + T_min

	def nexts(self, omega, n_gens):
		r"""Apply nextS on Camel.

		Args:
			omega (float): TODO.
			n_gens (int): Number of Camel Algorithm iterations/generations.
		"""
		self.S = self.S_past * (1 - omega * self.steps / n_gens)

	def nexte(self, n_gens, T_max):
		r"""Apply function nextE on function on Camel.

		Args:
			n_gens (int): Number of Camel Algorithm iterations/generations
			T_max (float): Maximum temperature of environment
		"""
		self.E = self.E_past * (1 - self.T / T_max) * (1 - self.steps / n_gens)

	def nextx(self, cb, E_init, S_init, task, rnd=rand):
		r"""Apply function nextX on Camel.

		This method/function move this Camel to new position in search space.

		Args:
			cb (Camel): Best Camel in population.
			E_init (float): Starting endurance of camel.
			S_init (float): Starting supply of camel.
			task (Task): Optimization task.
			rnd (Optional[mtrand.RandomState]): Random number generator.
		"""
		delta = -1 + rnd.rand() * 2
		self.x = self.x_past + delta * (1 - (self.E / E_init)) * np.exp(1 - self.S / S_init) * (cb - self.x_past)
		if not task.is_feasible(self.x): self.x = self.x_past
		else: self.f = task.eval(self.x)

	def next(self):
		r"""Save new position of Camel to old position."""
		self.x_past, self.f_past, self.E_past, self.S_past = self.x.copy(), self.f, self.E, self.S
		self.steps += 1
		return self

	def refill(self, S=None, E=None):
		r"""Apply this function to Camel.

		Args:
			S (float): New value of Camel supply.
			E (float): New value of Camel endurance.
		"""
		self.S, self.E = S, E


class CamelAlgorithm(Algorithm):
	r"""Implementation of Camel traveling behavior.

	Algorithm:
		Camel algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.iasj.net/iasj?func=fulltext&aId=118375

	Reference paper:
		Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177.

	Attributes:
		Name (List[str]): List of strings representing name of the algorithm.
		T_min (float): Minimal temperature of environment.
		T_max (float): Maximal temperature of environment.
		E_init (float): Starting value of energy.
		S_init (float): Starting value of supplys.

	See Also:
		* :class:`WeOptPy.algorithms.Algorithm`
	"""
	Name = ['CamelAlgorithm', 'CA']

	@staticmethod
	def algorithm_info():
		r"""Get information about algorithm.

		Returns:
			str: Algorithm information
		"""
		return r'''Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177.'''

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* omega (Callable[[Union[int, float]], bool])
				* mu (Callable[[float], bool])
				* alpha (Callable[[float], bool])
				* S_init (Callable[[Union[float, int]], bool])
				* E_init (Callable[[Union[float, int]], bool])
				* T_min (Callable[[Union[float, int], bool])
				* T_max (Callable[[Union[float, int], bool])

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'omega': lambda x: isinstance(x, (float, int)),
			'mu': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'S_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'E_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_min': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_max': lambda x: isinstance(x, (float, int)) and x > 0
		})
		return d

	def set_parameters(self, n=50, omega=0.25, mu=0.5, alpha=0.5, S_init=10, E_init=10, T_min=-10, T_max=10, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (Optional[int]): Population size :math:`\in [1, \infty)`.
			T_min (Optional[float]): Minimum temperature, must be true :math:`$T_{min} < T_{max}`.
			T_max (Optional[float]): Maximum temperature, must be true :math:`T_{min} < T_{max}`.
			omega (Optional[float]): Burden factor :math:`\in [0, 1]`.
			mu (Optional[float]): Dying rate :math:`\in [0, 1]`.
			S_init (Optional[float]): Initial supply :math:`\in (0, \infty)`.
			E_init (Optional[float]): Initial endurance :math:`\in (0, \infty)`.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, itype=Camel, init_pop_func=ukwargs.pop('init_pop_func', self.init_pop), **ukwargs)
		self.omega, self.mu, self.alpha, self.S_init, self.E_init, self.T_min, self.T_max = omega, mu, alpha, S_init, E_init, T_min, T_max

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'omega': self.omega,
			'mu': self.mu,
			'alpha': self.alpha,
			'S_init': self.S_init,
			'E_init': self.E_init,
			'T_min': self.T_min,
			'T_max': self.T_max
		})
		return d

	def init_pop(self, task, n, rnd, itype, *args, **kwargs):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.
			n (int): Number of camels in population.
			rnd (mtrand.RandomState): Random number generator.
			itype (Individual): Individual type.
			wargs (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialize population of camels.
				2. Initialized populations function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments.
		"""
		caravan = objects2array([itype(E_init=self.E_init, S_init=self.S_init, task=task, rnd=rnd, e=True) for _ in range(n)])
		return caravan, np.asarray([c.f for c in caravan]), args, kwargs

	def walk(self, c, cb, task):
		r"""Move the camel in search space.

		Args:
			c (Camel): Camel that we want to move.
			cb (Camel): Best know camel.
			task (Task): Optimization task.

		Returns:
			Camel: Camel that moved in the search space.
		"""
		c.nextt(self.T_min, self.T_max, self.Rand)
		c.nexts(self.omega, task.nGEN)
		c.nexte(task.nGEN, self.T_max)
		c.nextx(cb, self.E_init, self.S_init, task, self.Rand)
		return c

	def oasis(self, c, rn, alpha):
		r"""Apply oasis function to camel.

		Args:
			c (Camel): Camel to apply oasis on.
			rn (float): Random number.
			alpha (float): View range of Camel.

		Returns:
			Camel: Camel with applied oasis on.
		"""
		if rn > 1 - alpha and c.f < c.f_past: c.refill(self.S_init, self.E_init)
		return c

	def life_cycle(self, c, mu, task):
		r"""Apply life cycle to Camel.

		Args:
			c (Camel): Camel to apply life cycle.
			mu (float): Vision range of camel.
			task (Task): Optimization task.

		Returns:
			Camel: Camel with life cycle applied to it.
		"""
		if c.f_past < mu * c.f: return Camel(self.E_init, self.S_init, rnd=self.Rand, task=task)
		else: return c.next()

	def init_population(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population of Camels.
				2. New population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.initPopulation`
		"""
		caravan, fcaravan, args, kwargs = Algorithm.init_population(self, task)
		return caravan, fcaravan, args, kwargs

	def run_iteration(self, task, caravan, fcaravan, cb, fcb, *args, **dparams):
		r"""Core function of Camel Algorithm.

		Args:
			task (Task): Optimization task.
			caravan (numpy.ndarray[Camel]): Current population of Camels.
			fcaravan (numpy.ndarray[float]): Current population fitness/function values.
			cb (Camel): Current best Camel.
			fcb (float): Current best Camel fitness/function value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New population function/fitness value.
				3. New global best solution.
				4. New global best fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments.
		"""
		ncaravan = objects2array([self.walk(c, cb, task) for c in caravan])
		ncaravan = objects2array([self.oasis(c, self.rand(), self.alpha) for c in ncaravan])
		ncaravan = objects2array([self.life_cycle(c, self.mu, task) for c in ncaravan])
		fncaravan = np.asarray([c.f for c in ncaravan])
		cb, fcb = self.get_best(ncaravan, fncaravan, cb, fcb)
		return ncaravan, fncaravan, cb, fcb, args, dparams


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
