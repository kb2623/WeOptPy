# encoding=utf8

from math import ceil

import numpy as np
from numpy import random as rand
from numpy.linalg import (
	norm,
	cholesky as chol,
	eig,
	solve,
	lstsq
)

from WeOptPy.algorithms.interfaces import (
	Algorithm,
	Individual,
	default_individual_init
)
from WeOptPy.util.utility import objects2array

__all__ = [
	'EvolutionStrategy1p1',
	'EvolutionStrategyMp1',
	'EvolutionStrategyMpL',
	'EvolutionStrategyML',
	'CovarianceMatrixAdaptionEvolutionStrategy',
	'IndividualES'
]


class IndividualES(Individual):
	r"""Individual for Evolution Strategies.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, **kwargs):
		r"""Initialize individual.

		Args:
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		self.rho = kwargs.get('rho', 1)


class EvolutionStrategy1p1(Algorithm):
	r"""Implementation of (1 + 1) evolution strategy algorithm. Uses just one individual.

	Algorithm:
		(1 + 1) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		mu (int): Number of parents.
		k (int): Number of iterations before checking and fixing rho.
		c_a (float): Search range amplification factor.
		c_r (float): Search range reduction factor.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['EvolutionStrategy1p1', 'EvolutionStrategy(1+1)', 'ES(1+1)']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* mu (Callable[[int], bool])
				* k (Callable[[int], bool])
				* c_a (Callable[[Union[float, int]], bool])
				* c_r (Callable[[Union[float, int]], bool])
				* epsilon (Callable[[float], bool])
		"""
		return {
			'mu': lambda x: isinstance(x, int) and x > 0,
			'k': lambda x: isinstance(x, int) and x > 0,
			'c_a': lambda x: isinstance(x, (float, int)) and x > 1,
			'c_r': lambda x: isinstance(x, (float, int)) and 0 < x < 1,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		}

	def set_parameters(self, mu=1, k=10, c_a=1.1, c_r=0.5, epsilon=1e-20, **ukwargs):
		r"""Set the arguments of an algorithm.

		Args:
			mu (Optional[int]): Number of parents
			k (Optional[int]): Number of iterations before checking and fixing rho
			c_a (Optional[float]): Search range amplification factor
			c_r (Optional[float]): Search range reduction factor

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=mu, itype=ukwargs.pop('itype', IndividualES), **ukwargs)
		self.mu, self.k, self.c_a, self.c_r, self.epsilon = mu, k, c_a, c_r, epsilon

	def mutate(self, x, rho):
		r"""Mutate individual.

		Args:
			x (Individual): Current individual.
			rho (float): Current standard deviation.

		Returns:
			Individual: Mutated individual.
		"""
		return x + self.normal(0, rho, len(x))

	def updateRho(self, rho, k):
		r"""Update standard deviation.

		Args:
			rho (float): Current standard deviation.
			k (int): Number of succesfull mutations.

		Returns:
			float: New standard deviation.
		"""
		phi = k / self.k
		if phi < 0.2: return self.c_r * rho if rho > self.epsilon else 1
		elif phi > 0.2: return self.c_a * rho if rho > self.epsilon else 1
		else: return rho

	def init_population(self, task):
		r"""Initialize starting individual.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[Individual, float, Dict[str, Any]]:
				1, Initialized individual.
				2, Initialized individual fitness/function value.
				3. Additional arguments:
					* ki (int): Number of successful rho update.
		"""
		c, ki = IndividualES(task=task, rnd=self.Rand), 0
		return c, c.f, {'ki': ki}

	def run_iteration(self, task, c, fpop, xb, fxb, ki, **dparams):
		r"""Core function of EvolutionStrategy(1+1) algorithm.

		Args:
			task (Task): Optimization task.
			pop (Individual): Current position.
			fpop (float): Current position function/fitness value.
			xb (Individual): Global best position.
			fxb (float): Global best function/fitness value.
			ki (int): Number of successful updates before rho update.
			dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[Individual, float, Individual, float, Dict[str, Any]]:
				1, Initialized individual.
				2, Initialized individual fitness/function value.
				3. New global best solution.
				4. New global best soluitons fitness/objective value.
				5. Additional arguments:
					* ki (int): Number of successful rho update.
		"""
		if task.Iters % self.k == 0: c.rho, ki = self.updateRho(c.rho, ki), 0
		cn = objects2array([task.repair(self.mutate(c.x, c.rho), self.Rand) for _i in range(self.mu)])
		cn_f = np.asarray([task.eval(cn[i]) for i in range(len(cn))])
		ib = np.argmin(cn_f)
		if cn_f[ib] < c.f:
			c.x, c.f, ki = cn[ib], cn_f[ib], ki + 1
			if cn_f[ib] < fxb: xb, fxb = self.get_best(cn[ib], cn_f[ib], xb, fxb)
		return c, c.f, xb, fxb, {'ki': ki}


class EvolutionStrategyMp1(EvolutionStrategy1p1):
	r"""Implementation of (mu + 1) evolution strategy algorithm. Algorithm creates mu mutants but into new generation goes only one individual.

	Algorithm:
		(:math:`\mu + 1`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.basic.EvolutionStrategy1p1`
	"""
	Name = ['EvolutionStrategyMp1', 'EvolutionStrategy(mu+1)', 'ES(m+1)']

	def set_parameters(self, **kwargs):
		r"""Set core parameters of EvolutionStrategy(mu+1) algorithm.

		Args:
			kwargs (Dict[str, Any]):

		See Also:
			* :func:`NiaPy.algorithms.basic.EvolutionStrategy1p1.setParameters`
		"""
		mu = kwargs.pop('mu', 40)
		EvolutionStrategy1p1.set_parameters(self, mu=mu, **kwargs)


class EvolutionStrategyMpL(EvolutionStrategy1p1):
	r"""Implementation of (mu + lambda) evolution strategy algorithm. Mulation creates lambda individual. Lambda individual compete with mu individuals for survival, so only mu individual go to new generation.

	Algorithm:
		(:math:`\mu + \lambda`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		lam (int): TODO

	See Also:
		* :class:`NiaPy.algorithms.basic.EvolutionStrategy1p1`
	"""
	Name = ['EvolutionStrategyMpL', 'EvolutionStrategy(mu+lambda)', 'ES(m+l)']

	@staticmethod
	def type_parameters():
		r"""TODO.

		Returns:
			Dict[str, Any]:
				* lam (Callable[[int], bool]): TODO.

		See Also:
			* :func:`NiaPy.algorithms.basic.EvolutionStrategy1p1`
		"""
		d = EvolutionStrategy1p1.type_parameters()
		d['lam'] = lambda x: isinstance(x, int) and x > 0
		return d

	def set_parameters(self, lam=45, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			lam (int): Number of new individual generated by mutation.

		See Also:
			* :func:`NiaPy.algorithms.basic.es.EvolutionStrategy1p1.setParameters`
		"""
		EvolutionStrategy1p1.set_parameters(self, InitPopFunc=default_individual_init, **ukwargs)
		self.lam = lam

	def updateRho(self, pop, k):
		r"""Update standard deviation for population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			k (int): Number of successful mutations.
		"""
		phi = k / self.k
		if phi < 0.2:
			for i in pop: i.rho = self.c_r * i.rho
		elif phi > 0.2:
			for i in pop: i.rho = self.c_a * i.rho

	def changeCount(self, c, cn):
		r"""Update number of successful mutations for population.

		Args:
			c (numpy.ndarray[Individual]): Current population.
			cn (numpy.ndarray[Individual]): New population.

		Returns:
			int: Number of successful mutations.
		"""
		k = 0
		for e in cn:
			if e not in c: k += 1
		return k

	def mutateRand(self, pop, task):
		r"""Mutate random individual form population.

		Args:
			pop (numpy.ndarray): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Random individual from population that was mutated.
		"""
		i = self.randint(self.mu)
		return task.repair(self.mutate(pop[i].x, pop[i].rho), rnd=self.Rand)

	def init_population(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				1. Initialized populaiton.
				2. Initialized populations function/fitness values.
				3. Additional arguments:
					* ki (int): Number of successful mutations.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		c, fc, d = Algorithm.init_population(self, task)
		d.update({'ki': 0})
		return c, fc, d

	def run_iteration(self, task, c, fpop, xb, fxb, ki, **dparams):
		r"""Core function of EvolutionStrategyMpL algorithm.

		Args:
			task (Task): Optimization task.
			c (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals fitness/function value.
			ki (int): Number of successful mutations.
			dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations function/fitness values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments:
					* ki (int): Number of successful mutations.
		"""
		if task.Iters % self.k == 0: _, ki = self.updateRho(c, ki), 0
		cn = objects2array([IndividualES(x=self.mutateRand(c, task), task=task, rnd=self.Rand) for _ in range(self.lam)])
		cn = np.append(cn, c)
		cn = objects2array([cn[i] for i in np.argsort([i.f for i in cn])[:self.mu]])
		ki += self.changeCount(c, cn)
		fcn = np.asarray([x.f for x in cn])
		xb, fxb = self.get_best(cn, fcn, xb, fxb)
		return cn, fcn, xb, fxb, {'ki': ki}


class EvolutionStrategyML(EvolutionStrategyMpL):
	r"""Implementation of (mu, lambda) evolution strategy algorithm. Algorithm is good for dynamic environments. Mu individual create lambda chields. Only best mu chields go to new generation. Mu parents are discarded.

	Algorithm:
		(:math:`\mu + \lambda`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names

	See Also:
		* :class:`NiaPy.algorithm.basic.es.EvolutionStrategyMpL`
	"""
	Name = ['EvolutionStrategyML', 'EvolutionStrategy(mu,lambda)', 'ES(m,l)']

	def newPop(self, pop):
		r"""Return new population.

		Args:
			pop (numpy.ndarray): Current population.

		Returns:
			numpy.ndarray: New population.
		"""
		pop_s = np.argsort([i.f for i in pop])
		if self.mu < self.lam: return objects2array([pop[i] for i in pop_s[:self.mu]])
		npop = list()
		for i in range(int(ceil(float(self.mu) / self.lam))): npop.extend(pop[:self.lam if (self.mu - i * self.lam) >= self.lam else self.mu - i * self.lam])
		return objects2array(npop)

	def init_population(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.es.EvolutionStrategyMpL.initPopulation`
		"""
		c, fc, _ = EvolutionStrategyMpL.init_population(self, task)
		return c, fc, {}

	def run_iteration(self, task, c, fpop, xb, fxb, **dparams):
		r"""Core function of EvolutionStrategyML algorithm.

		Args:
			task (Task): Optimization task.
			c (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals fitness/function value.
			dparams Dict[str, Any]: Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
		"""
		cn = objects2array([IndividualES(x=self.mutateRand(c, task), task=task, rand=self.Rand) for _ in range(self.lam)])
		c = self.newPop(cn)
		fc = np.asarray([x.f for x in c])
		xb, fxb = self.get_best(c, fc, xb, fxb)
		return c, fc, xb, fxb, {}


def CovarianceMaatrixAdaptionEvolutionStrategyF(task, epsilon=1e-20, rnd=rand):
	lam, alpha_mu, hs, sigma0 = (4 + np.round(3 * np.log(task.D))) * 10, 2, 0, 0.3 * task.range()
	mu = int(np.round(lam / 2))
	w = np.log(mu + 0.5) - np.log(range(1, mu + 1))
	w = w / np.sum(w)
	mueff = 1 / np.sum(w ** 2)
	cs = (mueff + 2) / (task.D + mueff + 5)
	ds = 1 + cs + 2 * max(np.sqrt((mueff - 1) / (task.D + 1)) - 1, 0)
	ENN = np.sqrt(task.D) * (1 - 1 / (4 * task.D) + 1 / (21 * task.D ** 2))
	cc, c1 = (4 + mueff / task.D) / (4 + task.D + 2 * mueff / task.D), 2 / ((task.D + 1.3) ** 2 + mueff)
	cmu, hth = min(1 - c1, alpha_mu * (mueff - 2 + 1 / mueff) / ((task.D + 2) ** 2 + alpha_mu * mueff / 2)), (1.4 + 2 / (task.D + 1)) * ENN
	ps, pc, C, sigma, M = np.full(task.D, 0.0), np.full(task.D, 0.0), np.eye(task.D), sigma0, np.full(task.D, 0.0)
	x = rnd.uniform(task.lower(), task.upper())
	x_f = task.eval(x)
	while not task.stop_cond_i():
		pop_step = np.asarray([rnd.multivariate_normal(np.full(task.D, 0.0), C) for _ in range(int(lam))])
		pop = np.asarray([task.repair(x + sigma * ps, rnd) for ps in pop_step])
		pop_f = np.apply_along_axis(task.eval, 1, pop)
		isort = np.argsort(pop_f)
		pop, pop_f, pop_step = pop[isort[:mu]], pop_f[isort[:mu]], pop_step[isort[:mu]]
		if pop_f[0] < x_f: x, x_f = pop[0], pop_f[0]
		M = np.sum(w * pop_step.T, axis=1)
		ps = solve(chol(C).conj() + epsilon, ((1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * M + epsilon).T)[0].T
		sigma *= np.exp(cs / ds * (norm(ps) / ENN - 1)) ** 0.3
		ifix = np.where(sigma == np.inf)
		if np.any(ifix): sigma[ifix] = sigma0
		if norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (task.Iters + 1))) < hth: hs = 1
		else: hs = 0
		delta = (1 - hs) * cc * (2 - cc)
		pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * M
		C = (1 - c1 - cmu) * C + c1 * (np.tile(pc, [len(pc), 1]) * np.tile(pc.reshape([len(pc), 1]), [1, len(pc)]) + delta * C)
		for i in range(mu): C += cmu * w[i] * np.tile(pop_step[i], [len(pop_step[i]), 1]) * np.tile(pop_step[i].reshape([len(pop_step[i]), 1]), [1, len(pop_step[i])])
		E, V = eig(C)
		if np.any(E < epsilon):
			E = np.fmax(E, 0)
			C = lstsq(V.T, np.dot(V, np.diag(E)).T, rcond=None)[0].T
	return x, x_f


class CovarianceMatrixAdaptionEvolutionStrategy(Algorithm):
	r"""Implementation of (mu, lambda) evolution strategy algorithm. Algorithm is good for dynamic environments. Mu individual create lambda chields. Only best mu chields go to new generation. Mu parents are discarded.

	Algorithm:
		(:math:`\mu + \lambda`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://arxiv.org/abs/1604.00772

	Reference paper:
		Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016).

	Attributes:
		Name (List[str]): List of names representing algorithm names
		epsilon (float): TODO
	"""
	Name = ['CovarianceMatrixAdaptionEvolutionStrategy', 'CMA-ES', 'CMAES']
	epsilon = 1e-20

	@staticmethod
	def type_parameters(): return {
		'epsilon': lambda x: isinstance(x, (float, int)) and 0 < x < 1
	}

	def set_parameters(self, epsilon=1e-20, **ukwargs):
		r"""Set core parameters of CovarianceMatrixAdaptionEvolutionStrategy algorithm.

		Args:
			epsilon (float): Small number.
			ukwargs (Dict[str, Any]): Additional arguments.
		"""
		Algorithm.set_parameters(self, **ukwargs)
		self.epsilon = epsilon

	def run_task(self, task):
		r"""TODO.

		Args:
			task (Task): Optimization task.

		Returns:
			TODO.
		"""
		return CovarianceMaatrixAdaptionEvolutionStrategyF(task, self.epsilon, rnd=self.Rand)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
