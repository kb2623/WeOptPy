# encoding=utf8

from math import ceil

import numpy as np

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.algorithms.interfaces.individual import (
	Individual,
	default_numpy_init,
	default_individual_init
)

__all__ = [
	'MkeSolution',
	'MonkeyKingEvolutionV1',
	'MonkeyKingEvolutionV2',
	'MonkeyKingEvolutionV3'
]


def neg(x):
	r"""Transform function.

	Args:
		x (Union[int, float]): Should be 0 or 1.

	Returns:
		float: If 0 theta 1 else 1 then 0.
	"""
	return 0.0 if x == 1.0 else 1.0


class MkeSolution(Individual):
	r"""Implementation of Monkey King Evolution individual.

	Data:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		x_pb (array of (float or int)): Personal best position of Monkey particle.
		f_pb (float): Personal best fitness/function value.
		MonkeyKing (bool): Boolean value indicating if particle is Monkey King particle.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, **kwargs):
		r"""Initialize Monkey particle.

		Args:
			**kwargs: Additional arguments

		See Also:
			* :class:`NiaPy.algorithms.Individual.__init__()`
		"""
		Individual.__init__(self, **kwargs)
		self.f_pb, self.x_pb = self.f, self.x
		self.MonkeyKing = False

	def update_personal_best(self):
		r"""Update personal best position of particle."""
		if self.f < self.f_pb: self.x_pb, self.f_pb = self.x, self.f


class MonkeyKingEvolutionV1(Algorithm):
	r"""Implementation of monkey king evolution algorithm version 1.

	Algorithm:
		Monkey King Evolution version 1

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705116000198

	Reference paper:
		Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		F (float): Scale factor for normal particles.
		R (float): TODO.
		C (int): Number of new particles generated by Monkey King particle.
		FC (float): Scale factor for Monkey King particles.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['MonkeyKingEvolutionV1', 'MKEv1']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* F (Callable[[int], bool])
				* R (Callable[[Union[int, float]], bool])
				* C (Callable[[Union[int, float]], bool])
				* FC (Callable[[Union[int, float]], bool])
		"""
		d = Algorithm.type_parameters()
		d.update({
			'n': lambda x: isinstance(x, int) and x > 0,
			'F': lambda x: isinstance(x, (float, int)) and x > 0,
			'R': lambda x: isinstance(x, (float, int)) and x > 0,
			'C': lambda x: isinstance(x, int) and x > 0,
			'FC': lambda x: isinstance(x, (float, int)) and x > 0
		})
		return d

	def set_parameters(self, n=40, f=0.7, r=0.3, c=3, fc=0.5, **ukwargs):
		r"""Set Monkey King Evolution v1 algorithms static parameters.

		Args:
			n (int): Population size.
			f (float): Scale factor for normal particle.
			r (float): Percentage value of now many new particle Monkey King particle creates. Value in rage [0, 1].
			c (int): Number of new particles generated by Monkey King particle.
			fc (float): Scale factor for Monkey King particles.
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, itype=ukwargs.pop('itype', MkeSolution), init_pop_func=ukwargs.pop('init_pop_func', default_individual_init), **ukwargs)
		self.F, self.R, self.C, self.FC = f, r, c, fc

	def get_parameters(self):
		r"""Get algorithms parametes values.

		Returns:
			Dict[str, Any]: Dictionary of parameters name and value.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.getParameters`
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'F': self.F,
			'R': self.R,
			'C': self.C,
			'FC': self.FC
		})
		return d

	def move_p(self, x, x_pb, x_b, task):
		r"""Move normal particle in search space.

		For moving particles algorithm uses next formula:
		:math:`\mathbf{x_{pb} - \mathit{F} \odot \mathbf{r} \odot (\mathbf{x_b} - \mathbf{x})`
		where
		:math:`\mathbf{r}` is one dimension array with `d` components. Components in this vector are in range [0, 1].

		Args:
			x (numpy.ndarray): Paticle position.
			x_pb (numpy.ndarray): Particle best position.
			x_b (numpy.ndarray): Best particle position.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Particle new position.
		"""
		return x_pb + self.F * self.rand(task.D) * (x_b - x)

	def move_mk(self, x, task):
		r"""Move Monkey King particle.

		For moving Monkey King particles algorithm uses next formula:
		:math:`\mathbf{x} + \mathit{FC} \odot \mathbf{R} \odot \mathbf{x}`
		where
		:math:`\mathbf{R}` is two dimensional array with shape `{C * d, d}`. Componentes of this array are in range [0, 1]

		Args:
			x (numpy.ndarray): Monkey King patricle position.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New particles generated by Monkey King particle.
		"""
		return x + self.FC * self.rand([int(self.C * task.D), task.D]) * x

	def move_particle(self, p, p_b, task):
		r"""Move particles.

		Args:
			p (MkeSolution): Monkey particle.
			p_b (MkeSolution): Population best particle.
			task (Task): Optimization task.
		"""
		p.x = self.move_p(p.x, p.x_pb, p_b, task)
		p.evaluate(task, rnd=self.Rand)

	def move_monkey_king_particle(self, p, task):
		r"""Move Monkey King Particles.

		Args:
			p (MkeSolution): Monkey King particle to apply this function on.
			task (Task): Optimization task
		"""
		p.MonkeyKing = False
		a = np.apply_along_axis(task.repair, 1, self.move_mk(p.x, task), self.Rand)
		a_f = np.apply_along_axis(task.eval, 1, a)
		ib = np.argmin(a_f)
		p.x, p.f = a[ib], a_f[ib]

	def move_population(self, pop, xb, task):
		r"""Move population.

		Args:
			pop (numpy.ndarray[MkeSolution]): Current population.
			xb (MkeSolution): Current best solution.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[MkeSolution]: New particles.
		"""
		for p in pop:
			if p.MonkeyKing: self.move_monkey_king_particle(p, task)
			else: self.move_particle(p, xb, task)
			p.update_personal_best()
		return pop

	def init_population(self, task):
		r"""Init population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple(numpy.ndarray[MkeSolution], numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized solutions
				2. Fitness/function values of solution
				3. Additional arguments
		"""
		pop, fpop, _ = Algorithm.init_population(self, task)
		for i in self.Rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
		return pop, fpop, {}

	def run_iteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of Monkey King Evolution v1 algorithm.

		Args:
			task (Task): Optimization task
			pop (numpy.ndarray[MkeSolution]): Current population
			fpop (numpy.ndarray[float]): Current population fitness/function values
			xb (MkeSolution): Current best solution.
			fxb (float): Current best solutions function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple(numpy.ndarray[MkeSolution], numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized solutions.
				2. Fitness/function values of solution.
				3. Additional arguments.
		"""
		pop = self.move_population(pop, xb, task)
		for i in self.Rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
		fpop = np.asarray([m.f for m in pop])
		xb, fxb = self.get_best(pop, fpop, xb, fxb)
		return pop, fpop, xb, fxb, {}


class MonkeyKingEvolutionV2(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 2.

	Algorithm:
		Monkey King Evolution version 2

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705116000198

	Reference paper:
		Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.basic.mke.MonkeyKingEvolutionV1`
	"""
	Name = ['MonkeyKingEvolutionV2', 'MKEv2']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009."""

	def move_mk(self, x, dx, task):
		r"""Move Monkey King particle.

		For movment of particles algorithm uses next formula:
		:math:`\mathbf{x} - \mathit{FC} \odot \mathbf{dx}`

		Args:
			x (numpy.ndarray): Particle to apply movment on.
			dx (numpy.ndarray): Difference between to random paricles in population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Moved particles.
		"""
		return x - self.FC * dx

	def move_monkey_king_particle(self, p, pop, task):
		r"""Move Monkey King particles.

		Args:
			p (MkeSolution): Monkey King particle to move.
			pop (numpy.ndarray[MkeSolution]): Current population.
			task (Task): Optimization task.
		"""
		p.MonkeyKing = False
		p_b, p_f = p.x, p.f
		for _i in range(int(self.C * self.NP)):
			r = self.Rand.choice(self.NP, 2, replace=False)
			a = task.repair(self.move_mk(p.x, pop[r[0]].x - pop[r[1]].x, task), self.Rand)
			a_f = task.eval(a)
			if a_f < p_f: p_b, p_f = a, a_f
		p.x, p.f = p_b, p_f

	def move_population(self, pop, xb, task):
		r"""Move population.

		Args:
			pop (numpy.ndarray[MkeSolution]): Current population.
			xb (MkeSolution): Current best solution.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[MkeSolution]: Moved population.
		"""
		for p in pop:
			if p.MonkeyKing: self.move_monkey_king_particle(p, pop, task)
			else: self.move_particle(p, xb, task)
			p.update_personal_best()
		return pop


class MonkeyKingEvolutionV3(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 3.

	Algorithm:
		Monkey King Evolution version 3

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705116000198

	Reference paper:
		Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.

	Attributes:
		Name (List[str]): List of strings that represent algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.basic.mke.MonkeyKingEvolutionV1`
	"""
	Name = ['MonkeyKingEvolutionV3', 'MKEv3']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009."""

	def set_parameters(self, **ukwargs):
		r"""Set core parameters of MonkeyKingEvolutionV3 algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.MonkeyKingEvolutionV1.setParameters`
		"""
		MonkeyKingEvolutionV1.set_parameters(self, itype=ukwargs.pop('itype', None), InitPopFunc=ukwargs.pop('init_pop_func', default_numpy_init), **ukwargs)

	def init_population(self, task):
		r"""Initialize the population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized population function/fitness values.
				3. Additional arguments:
					* k (int): TODO.
					* c (int): TODO.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		x, x_f, d = Algorithm.init_population(self, task)
		k, c = int(ceil(self.NP / task.D)), int(ceil(self.C * task.D))
		d.update({'k': k, 'c': c})
		return x, x_f, d

	def run_iteration(self, task, x, x_f, xb, fxb, k, c, **dparams):
		r"""Core function of Monkey King Evolution v3 algorithm.

		Args:
			task (Task): Optimization task
			x (numpy.ndarray): Current population
			x_f (numpy.ndarray[float]): Current population fitness/function values
			xb (numpy.ndarray): Current best individual
			fxb (float): Current best individual function/fitness value
			k (int): TODO
			c (int: TODO
			**dparams: Additional arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized population function/fitness values.
				3. Additional arguments:
					* k (int): TODO.
					* c (int): TODO.
		"""
		x_gb = np.apply_along_axis(task.repair, 1, xb + self.FC * x[self.Rand.choice(len(x), c)] - x[self.Rand.choice(len(x), c)], self.Rand)
		x_gb_f = np.apply_along_axis(task.eval, 1, x_gb)
		xb, fxb = self.get_best(x_gb, x_gb_f, xb, fxb)
		m = np.full([self.NP, task.D], 1.0)
		for i in range(k): m[i * task.D:(i + 1) * task.D] = np.tril(m[i * task.D:(i + 1) * task.D])
		for i in range(self.NP): self.Rand.shuffle(m[i])
		x = np.apply_along_axis(task.repair, 1, m * x + np.vectorize(neg)(m) * xb, self.Rand)
		x_f = np.apply_along_axis(task.eval, 1, x)
		xb, fxb = self.get_best(x, x_f, xb, fxb)
		iw, ib_gb = np.argmax(x_f), np.argmin(x_gb_f)
		if x_gb_f[ib_gb] <= x_f[iw]: x[iw], x_f[iw] = x_gb[ib_gb], x_gb_f[ib_gb]
		return x, x_f, xb, fxb, {'k': k, 'c': c}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
