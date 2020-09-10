# encoding=utf8

"""Glow warm swarm optimization algorithm module."""

import numpy as np
from scipy.spatial.distance import euclidean

from WeOptPy.algorithms.interfaces.algorithm import Algorithm

__all__ = [
	'GlowwormSwarmOptimization',
	'GlowwormSwarmOptimizationV1',
	'GlowwormSwarmOptimizationV2',
	'GlowwormSwarmOptimizationV3'
]


class GlowwormSwarmOptimization(Algorithm):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings represeinting algorithm name.
		l0 (float): Initial luciferin quantity for each glowworm.
		nt (float): --
		rs (float): Maximum sensing range.
		rho (float): Luciferin decay constant.
		gamma (float): Luciferin enhancement constant.
		beta (float): --
		s (float): --
		Distance (Callable[[numpy.ndarray, numpy.ndarray], float]]): Measure distance between two individuals.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['GlowwormSwarmOptimization', 'GSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* n (Callable[[int], bool])
				* l0 (Callable[[Union[float, int]], bool])
				* nt (Callable[[Union[float, int]], bool])
				* rho (Callable[[Union[float, int]], bool])
				* gamma (Callable[[float], bool])
				* beta (Callable[[float], bool])
				* s (Callable[[float], bool])
		"""
		return {
			'n': lambda x: isinstance(x, int) and x > 0,
			'l0': lambda x: isinstance(x, (float, int)) and x > 0,
			'nt': lambda x: isinstance(x, (float, int)) and x > 0,
			'rho': lambda x: isinstance(x, float) and 0 < x < 1,
			'gamma': lambda x: isinstance(x, float) and 0 < x < 1,
			'beta': lambda x: isinstance(x, float) and x > 0,
			's': lambda x: isinstance(x, float) and x > 0
		}

	def set_parameters(self, n=25, l0=5, nt=5, rho=0.4, gamma=0.6, beta=0.08, s=0.03, Distance=euclidean, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (Optional[int]): Number of glowworms in population.
			l0 (Optional[float]): Initial luciferin quantity for each glowworm.
			nt (Optional[float]): --
			rs (Optional]float]): Maximum sensing range.
			rho (Optional[float]): Luciferin decay constant.
			gamma (Optional[float]): Luciferin enhancement constant.
			beta (Optional[float]): --
			s (Optional[float]): --
			Distance (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]]): Measure distance between two individuals.
		"""
		ukwargs.pop('n', None)
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.l0, self.nt, self.rho, self.gamma, self.beta, self.s, self.Distance = l0, nt, rho, gamma, beta, s, Distance

	def get_parameters(self):
		r"""Get algorithms parameters values.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.get_parameters(self)
		d.pop('n', None)
		d.update({
			'n': self.NP,
			'l0': self.l0,
			'nt': self.nt,
			'rho': self.rho,
			'gamma': self.gamma,
			'beta': self.beta,
			's': self.s,
			'Distance': self.Distance
		})
		return d

	def get_neighbors(self, i, r, GS, L):
		r"""Get neighbours of glowworm.

		Args:
			i (int): Index of glowworm.
			r (float): Neighborhood distance.
			GS (numpy.ndarray):
			L (numpy.ndarray): Luciferin value of glowworm.

		Returns:
			numpy.ndarray: Indexes of neighborhood glowworms.
		"""
		N = np.full(self.NP, 0)
		for j, gw in enumerate(GS): N[j] = 1 if i != j and self.Distance(GS[i], gw) <= r and L[i] >= L[j] else 0
		return N

	def probabilities(self, i, N, L):
		r"""Calculate probabilities for glowworm to movement.

		Args:
			i (int): Index of glowworm to search for probable movement.
			N (numpy.ndarray):
			L (numpy.ndarray):

		Returns:
			numpy.ndarray: Probabilities for each glowworm in swarm.
		"""
		d, P = np.sum(L[np.where(N == 1)] - L[i]), np.full(self.NP, .0)
		for j in range(self.NP): P[i] = ((L[j] - L[i]) / d) if N[j] == 1 else 0
		return P

	def move_select(self, pb, i):
		r"""TODO.

		Args:
			pb:
			i:

		Returns:

		"""
		r, b_l, b_u = self.rand(), 0, 0
		for j in range(self.NP):
			b_l, b_u = b_u, b_u + pb[i]
			if b_l < r < b_u: return j
		return self.randint(self.NP)

	def calc_luciferin(self, L, GS_f):
		r"""TODO.

		Args:
			L:
			GS_f:

		Returns:

		"""
		return (1 - self.rho) * L + self.gamma * GS_f

	def range_update(self, R, N, rs):
		r"""TODO.

		Args:
			R:
			N:
			rs:

		Returns:

		"""
		return R + self.beta * (self.nt - np.sum(N))

	def init_population(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], list, dict]:
				1. Initialized population of glowwarms.
				2. Initialized populations function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* L (numpy.ndarray): TODO.
					* R (numpy.ndarray): TODO.
					* rs (numpy.ndarray): TODO.
		"""
		GS, GS_f, args, kwargs = Algorithm.init_population(self, task)
		rs = euclidean(np.full(task.D, 0), task.bRange)
		L, R = np.full(self.NP, self.l0), np.full(self.NP, rs)
		kwargs.update({'L': L, 'R': R, 'rs': rs})
		return GS, GS_f, args, kwargs

	def run_iteration(self, task, GS, GS_f, xb, fxb, L, R, rs, *args, **kwargs):
		r"""Core function of GlowwormSwarmOptimization algorithm.

		Args:
			task (Task): Optimization taks.
			GS (numpy.ndarray): Current population.
			GS_f (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function/fitness value.
			L (numpy.ndarray):
			R (numpy.ndarray):
			rs (numpy.ndarray):
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. Initialized population of glowwarms.
				2. Initialized populations function/fitness values.
				3. New global best solution
				4. New global best sloutions fitness/objective value.
				5. Additional arguments:
					* L (numpy.ndarray): TODO.
					* R (numpy.ndarray): TODO.
					* rs (numpy.ndarray): TODO.
		"""
		GSo, Ro = np.copy(GS), np.copy(R)
		L = self.calc_luciferin(L, GS_f)
		N = [self.get_neighbors(i, Ro[i], GSo, L) for i in range(self.NP)]
		P = [self.probabilities(i, N[i], L) for i in range(self.NP)]
		j = [self.move_select(P[i], i) for i in range(self.NP)]
		for i in range(self.NP): GS[i] = task.repair(GSo[i] + self.s * ((GSo[j[i]] - GSo[i]) / (self.Distance(GSo[j[i]], GSo[i]) + 1e-31)), rnd=self.Rand)
		for i in range(self.NP): R[i] = max(0, min(rs, self.range_update(Ro[i], N[i], rs)))
		GS_f = np.apply_along_axis(task.eval, 1, GS)
		xb, fxb = self.get_best(GS, GS_f, xb, fxb)
		return GS, GS_f, xb, fxb, args, {'L': L, 'R': R, 'rs': rs}


class GlowwormSwarmOptimizationV1(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		alpha (float): --

	See Also:
		* :class:`WeOptPy.algorithms.GlowwormSwarmOptimization`
	"""
	Name = ['GlowwormSwarmOptimizationV1', 'GSOv1']

	def set_parameters(self, **kwargs):
		r"""Set default parameters of the algorithm.

		Args:
			kwargs (dict): Additional arguments.
		"""
		GlowwormSwarmOptimization.set_parameters(self, **kwargs)

	def calc_luciferin(self, L, GS_f):
		r"""TODO.

		Args:
			L:
			GS_f:

		Returns:

		"""
		return np.fmax(0, (1 - self.rho) * L + self.gamma * GS_f)

	def range_update(self, R, N, rs):
		r"""TODO.

		Args:
			R:
			N:
			rs:

		Returns:

		"""
		return rs / (1 + self.beta * (np.sum(N) / (np.pi * rs ** 2)))


class GlowwormSwarmOptimizationV2(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		alpha (float): --

	See Also:
		* :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`
	"""
	Name = ['GlowwormSwarmOptimizationV2', 'GSOv2']

	def set_parameters(self, alpha=0.2, **kwargs):
		r"""Set core parameters for GlowwormSwarmOptimizationV2 algorithm.

		Args:
			alpha (Optional[float]): --
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.GlowwormSwarmOptimization.setParameters`
		"""
		GlowwormSwarmOptimization.set_parameters(self, **kwargs)
		self.alpha = alpha

	def range_update(self, P, N, rs):
		r"""TODO.

		Args:
			P:
			N:
			rs:

		Returns:
			float: TODO
		"""
		return self.alpha + (rs - self.alpha) / (1 + self.beta * np.sum(N))


class GlowwormSwarmOptimizationV3(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		beta1 (float): --

	See Also:
		* :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`
	"""
	Name = ['GlowwormSwarmOptimizationV3', 'GSOv3']

	def set_parameters(self, beta1=0.2, **kwargs):
		r"""Set core parameters for GlowwormSwarmOptimizationV3 algorithm.

		Args:
			beta1 (Optional[float]): --
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.GlowwormSwarmOptimization.setParameters`
		"""
		GlowwormSwarmOptimization.set_parameters(self, **kwargs)
		self.beta1 = beta1

	def range_update(self, R, N, rs):
		r"""TODO.

		Args:
			R:
			N:
			rs:

		Returns:

		"""
		return R + (self.beta * np.sum(N)) if np.sum(N) < self.nt else (-self.beta1 * np.sum(N))


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
