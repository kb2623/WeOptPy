# encoding=utf8

"""Krill heard algorithm module."""

import numpy as np
from scipy.spatial.distance import euclidean as ed

from WeOptPy.util.utility import full_array
from WeOptPy.algorithms.interfaces.algorithm import Algorithm

__all__ = [
	'KrillHerdV1',
	'KrillHerdV2',
	'KrillHerdV3',
	'KrillHerdV4',
	'KrillHerdV11'
]


class KrillHerd(Algorithm):
	r"""Implementation of krill herd algorithm.

	Algorithm:
		Krill Herd Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.sciencedirect.com/science/article/pii/S1007570412002171

	Reference paper:
		Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: a new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.

	Attributes:
		Name (List[str]): List of strings representing algorithm names..
		n (int): Number of krill herds in population.
		N_max (float): Maximum induced speed.
		V_f (float): Foraging speed.
		D_max (float): Maximum diffusion speed.
		C_t (float): Constant :math:`\in [0, 2]`
		W_n (Union[int, float, numpy.ndarray]): Interta weights of the motion induced from neighbors :math:`\in [0, 1]`.
		W_f (Union[int, float, numpy.ndarray]): Interta weights of the motion induced from foraging :math`\in [0, 1]`.
		d_s (float): Maximum euclidean distance for neighbors.
		nn (int): Maximum neighbors for neighbors effect.
		Cr (float): Crossover probability.
		Mu (float): Mutation probability.
		epsilon (float): Small numbers for division.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['KrillHerd', 'KH']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* N_max (Callable[[Union[int, float]], bool])
				* V_f (Callable[[Union[int, float]], bool])
				* D_max (Callable[[Union[int, float]], bool])
				* C_t (Callable[[Union[int, float]], bool])
				* W_n (Callable[[Union[int, float]], bool])
				* W_f (Callable[[Union[int, float]], bool])
				* d_s (Callable[[Union[int, float]], boool])
				* nn (Callable[[int], bool])
				* Cr (Callable[[float], bool])
				* Mu (Callable[[float], bool])
				* epsilon (Callable[[float], bool])

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'N_max': lambda x: isinstance(x, (int, float)) and x > 0,
			'V_f': lambda x: isinstance(x, (int, float)) and x > 0,
			'D_max': lambda x: isinstance(x, (int, float)) and x > 0,
			'C_t': lambda x: isinstance(x, (int, float)) and x > 0,
			'W_n': lambda x: isinstance(x, (int, float)) and x > 0,
			'W_f': lambda x: isinstance(x, (int, float)) and x > 0,
			'd_s': lambda x: isinstance(x, (int, float)) and x > 0,
			'nn': lambda x: isinstance(x, int) and x > 0,
			'Cr': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'Mu': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		})
		return d

	def set_parameters(self, n=50, N_max=0.01, V_f=0.02, D_max=0.002, C_t=0.93, W_n=0.42, W_f=0.38, d_s=2.63, nn=5, Cr=0.2, Mu=0.05, epsilon=1e-31, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (Optional[int]): Number of krill herds in population.
			N_max (Optional[float]): Maximum induced speed.
			V_f (Optional[float]): Foraging speed.
			D_max (Optional[float]): Maximum diffusion speed.
			C_t (Optional[float]): Constant $\in [0, 2]$.
			W_n (Optional[Union[int, float, numpy.ndarray]]): Intera weights of the motion induced from neighbors :math:`\in [0, 1]`.
			W_f (Optional[Union[int, float, numpy.ndarray]]): Intera weights of the motion induced from foraging :math:`\in [0, 1]`.
			d_s (Optional[float]): Maximum euclidean distance for neighbors.
			nn (Optional[int]): Maximum neighbors for neighbors effect.
			Cr (Optional[float]): Crossover probability.
			Mu (Optional[float]): Mutation probability.
			epsilon (Optional[float]): Small numbers for division.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.N_max, self.V_f, self.D_max, self.C_t, self.W_n, self.W_f, self.d_s, self.nn, self._Cr, self._Mu, self.epsilon = N_max, V_f, D_max, C_t, W_n, W_f, d_s, nn, Cr, Mu, epsilon

	def get_parameters(self):
		r"""Get parameter values for the algorithm.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'N_max': self.N_max,
			'V_f': self.V_f,
			'D_max': self.D_max,
			'C_t': self.C_t,
			'W_n': self.W_n,
			'W_f': self.W_f,
			'd_s': self.d_s,
			'nn': self.nn,
			'Cr': self.Cr,
			'Mu': self.Mu,
			'epsilon': self.epsilon
		})
		return d

	def init_weights(self, task):
		r"""Initialize weights.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. Weights for neighborhood.
				2. Weights for foraging.
		"""
		return full_array(self.W_n, task.D), full_array(self.W_f, task.D)

	def sens_range(self, ki, KH):
		r"""Calculate sense range for selected individual.

		Args:
			ki (int): Selected individual.
			KH (numpy.ndarray): Krill heard population.

		Returns:
			float: Sense range for krill.
		"""
		return np.sum([ed(KH[ki], KH[i]) for i in range(self.NP)]) / (self.nn * self.NP)

	def get_neighbours(self, i, ids, KH):
		r"""Get neighbours.

		Args:
			i (int): Individual looking for neighbours.
			ids (float): Maximal distance for being a neighbour.
			KH (numpy.ndarray): Current population.

		Returns:
			numpy.ndarray: Neighbours of krill heard.
		"""
		N = list()
		for j in range(self.NP):
			if j != i and ids > ed(KH[i], KH[j]): N.append(j)
		if not N: N.append(self.randint(self.NP))
		return np.asarray(N)

	def funX(self, x, y):
		r"""Get x values.

		Args:
			x (numpy.ndarray): First krill/individual.
			y (numpy.ndarray): Second krill/individual.

		Returns:
			numpy.ndarray: TODO.
		"""
		return ((y - x) + self.epsilon) / (ed(y, x) + self.epsilon)

	def funK(self, x, y, b, w):
		r"""Get k values.

		Args:
			x (numpy.ndarray): First krill/individual.
			y (numpy.ndarray): Second krill/individual.
			b (numpy.ndarray): Best krill/individual.
			w (numpy.ndarray): Worst krill/individual.

		Returns:
			numpy.ndarray: TODO
		"""
		return ((x - y) + self.epsilon) / ((w - b) + self.epsilon)

	def induce_neighbors_motion(self, i, n, W, KH, KH_f, ikh_b, ikh_w, task):
		r"""Induced neighbours motion operator.

		Args:
			i (int): Index of individual being applied with operator.
			n (): TODO
			W (numpy.ndarray): Wights for this operator.
			KH (numpy.ndarray): Current heard/population.
			KH_f (numpy.ndarray): Current populations/heard function/fitness values.
			ikh_b (int): Current best krill in heard/population.
			ikh_w (int): Current worst krill in heard/population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Moved krill.
		"""
		Ni = self.get_neighbours(i, self.sens_range(i, KH), KH)
		Nx, Nf, f_b, f_w = KH[Ni], KH_f[Ni], KH_f[ikh_b], KH_f[ikh_w]
		alpha_l = np.sum(np.asarray([self.funK(KH_f[i], j, f_b, f_w) for j in Nf]) * np.asarray([self.funX(KH[i], j) for j in Nx]).T)
		alpha_t = 2 * (1 + self.rand() * task.Iters / task.nGEN)
		return self.N_max * (alpha_l + alpha_t) + W * n

	def induceForagingMotion(self, i, x, x_f, f, W, KH, KH_f, ikh_b, ikh_w, task):
		r"""Induced foraging motion operator.

		Args:
			i (int): Index of current krill being operated.
			x (numpy.ndarray): Position of food.
			x_f (float): Fitness/function values of food.
			f (): TODO
			W (numpy.ndarray): Weights for this operator.
			KH (numpy.ndarray):  Current population/heard.
			KH_f (numpy.ndarray): Current heard/populations function/fitness values.
			ikh_b (int): Index of current best krill in heard.
			ikh_w (int): Index of current worst krill in heard.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Moved krill.
		"""
		beta_f = 2 * (1 - task.Iters / task.nGEN) * self.funK(KH_f[i], x_f, KH_f[ikh_b], KH_f[ikh_w]) * self.funX(KH[i], x) if KH_f[ikh_b] < KH_f[i] else 0
		beta_b = self.funK(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) * self.funX(KH[i], KH[ikh_b])
		return self.V_f * (beta_f + beta_b) + W * f

	def induce_physical_diffusion(self, task):
		r"""Induced physical diffusion operator.

		Args:
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: TODO.
		"""
		return self.D_max * (1 - task.Iters / task.nGEN) * self.uniform(-1, 1, task.D)

	def deltaT(self, task):
		r"""Get new delta for all dimensions.

		Args:
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: TODO.
		"""
		return self.C_t * np.sum(task.bRange)

	def crossover(self, x, xo, Cr):
		r"""Crossover operator.

		Args:
			x (numpy.ndarray): Krill/individual being applied with operator.
			xo (numpy.ndarray): Krill/individual being used in conjunction within operator.
			Cr (float): Crossover probability.

		Returns:
			numpy.ndarray: Crossoverd krill/individual.
		"""
		return [xo[i] if self.rand() < Cr else x[i] for i in range(len(x))]

	def mutate(self, x, x_b, Mu):
		r"""Mutate operator.

		Args:
			x (numpy.ndarray): Individual being mutated.
			x_b (numpy.ndarray): Global best individual.
			Mu (float): Probability of mutations.

		Returns:
			numpy.ndarray: Mutated krill.
		"""
		return [x[i] if self.rand() < Mu else (x_b[i] + self.rand()) for i in range(len(x))]

	def get_food_location(self, KH, KH_f, task):
		r"""Get food location for krill heard.

		Args:
			KH (numpy.ndarray): Current heard/population.
			KH_f (numpy.ndarray): Current heard/populations function/fitness values.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Location of food.
				2. Foods function/fitness value.
		"""
		x_food = task.repair(np.asarray([np.sum(KH[:, i] / KH_f) for i in range(task.D)]) / np.sum(1 / KH_f), rnd=self.Rand)
		x_food_f = task.eval(x_food)
		return x_food, x_food_f

	def Mu(self, xf, yf, xf_best, xf_worst):
		r"""Get mutation probability.

		Args:
			xf (float):
			yf (float):
			xf_best (float):
			xf_worst (float):

		Returns:
			float: New mutation probability.
		"""
		return self._Mu / (self.funK(xf, yf, xf_best, xf_worst) + 1e-31)

	def Cr(self, xf, yf, xf_best, xf_worst):
		r"""Get crossover probability.

		Args:
			xf (float):
			yf (float):
			xf_best (float):
			xf_worst (flaot):

		Returns:
			float: New crossover probability.
		"""
		return self._Cr * self.funK(xf, yf, xf_best, xf_worst)

	def init_population(self, task):
		r"""Initialize stating population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population.
				2. Initialized populations function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* W_n (numpy.ndarray): Weights neighborhood.
					* W_f (numpy.ndarray): Weights foraging.
					* N (numpy.ndarray): TODO
					* F (numpy.ndarray): TODO

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		KH, KH_f, args, kwargs = Algorithm.init_population(self, task)
		W_n, W_f = self.init_weights(task)
		N, F = np.full(self.NP, .0), np.full(self.NP, .0)
		kwargs.update({'W_n': W_n, 'W_f': W_f, 'N': N, 'F': F})
		return KH, KH_f, args, kwargs

	def run_iteration(self, task, KH, KH_f, xb, fxb, W_n, W_f, N, F, *args, **kwargs):
		r"""Core function of KrillHerd algorithm.

		Args:
			task (Task): Optimization task.
			KH (numpy.ndarray): Current heard/population.
			KH_f (numpy.ndarray[float]): Current heard/populations function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function fitness values.
			W_n (numpy.ndarray): TODO
			W_f (numpy.ndarray): TODO
			N (numpy.ndarray): TODO
			F (numpy.ndarray): TODO
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple [numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New herd/population
				2. New herd/populations function/fitness values.
				3. New global best solution.
				4. New global best solutoins fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* W_n (numpy.ndarray): TODO
					* W_f (numpy.ndarray): TODO
					* N (numpy.ndarray): TODO
					* F (numpy.ndarray): TODO
		"""
		ikh_b, ikh_w = np.argmin(KH_f), np.argmax(KH_f)
		x_food, x_food_f = self.get_food_location(KH, KH_f, task)
		if x_food_f < fxb: xb, fxb = x_food, x_food_f  # noqa: F841
		N = np.asarray([self.induce_neighbors_motion(i, N[i], W_n, KH, KH_f, ikh_b, ikh_w, task) for i in range(self.NP)])
		F = np.asarray([self.induceForagingMotion(i, x_food, x_food_f, F[i], W_f, KH, KH_f, ikh_b, ikh_w, task) for i in range(self.NP)])
		D = np.asarray([self.induce_physical_diffusion(task) for i in range(self.NP)])
		KH_n = KH + (self.deltaT(task) * (N + F + D))
		Cr = np.asarray([self.Cr(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(self.NP)])
		KH_n = np.asarray([self.crossover(KH_n[i], KH[i], Cr[i]) for i in range(self.NP)])
		Mu = np.asarray([self.Mu(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(self.NP)])
		KH_n = np.asarray([self.mutate(KH_n[i], KH[ikh_b], Mu[i]) for i in range(self.NP)])
		KH = np.apply_along_axis(task.repair, 1, KH_n, rnd=self.Rand)
		KH_f = np.apply_along_axis(task.eval, 1, KH)
		xb, fxb = self.get_best(KH, KH_f, xb, fxb)
		return KH, KH_f, xb, fxb, args, {'W_n': W_n, 'W_f': W_f, 'N': N, 'F': F}


class KrillHerdV4(KrillHerd):
	r"""Implementation of krill herd algorithm.

	Algorithm:
		Krill Herd Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.sciencedirect.com/science/article/pii/S1007570412002171

	Reference paper:
		Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: a new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
	"""
	Name = ['KrillHerdV4', 'KHv4']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Dictionary with testing functions for parameters.

		See Also:
			* :func:NiaPy.algorithms.basic.kh.KrillHerd.typeParameters`
		"""
		d = KrillHerd.type_parameters()
		d.pop('Cr', None)
		d.pop('Mu', None)
		d.pop('epsilon', None)
		return d

	def set_parameters(self, n=50, N_max=0.01, V_f=0.02, D_max=0.002, C_t=0.93, W_n=0.42, W_f=0.38, d_s=2.63, **ukwargs):
		r"""Set algorithm core parameters.

		Args:
			n (int): Number of kills in herd.
			N_max (Optional[float]): TODO
			V_f (Optional[float]): TODO
			D_max (Optional[float]): TODO
			C_t (Optional[float]): TODO
			W_n (Optional[Union[int, float, numpy.ndarray, list]]): Weights for neighborhood.
			W_f (Optional[Union[int, float, numpy.ndarray, list]]): Weights for foraging.
			d_s (Optional[float]): TODO
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:NiaPy.algorithms.basic.kh.KrillHerd.KrillHerd.setParameters`
		"""
		KrillHerd.set_parameters(self, n=n, N_max=N_max, V_f=V_f, D_max=D_max, C_t=C_t, W_n=W_n, W_f=W_f, d_s=d_s, nn=4, Cr=0.2, Mu=0.05, epsilon=1e-31, **ukwargs)


class KrillHerdV1(KrillHerd):
	r"""Implementation of krill herd algorithm.

	Algorithm:
		Krill Herd Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.sciencedirect.com/science/article/pii/S1007570412002171

	Reference paper:
		Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: a new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :func:NiaPy.algorithms.basic.kh.KrillHerd.KrillHerd`
	"""
	Name = ['KrillHerdV1', 'KHv1']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Dictionary with testing functions for parameters.

		See Also:
			* :func:NiaPy.algorithms.basic.kh.KrillHerd.typeParameters`
		"""
		return KrillHerd.type_parameters()

	def crossover(self, x, xo, Cr):
		r"""Preform a crossover operation on individual.

		Args:
			x (numpy.ndarray): Current individual.
			xo (numpy.ndarray): New individual.
			Cr (float): Crossover probability.

		Returns:
			numpy.ndarray: Crossover individual.
		"""
		return x

	def mutate(self, x, x_b, Mu):
		r"""Mutate individual.

		Args:
			x (numpy.ndarray): Current individual.
			x_b (numpy.ndarray): Global best individual.
			Mu (float): Mutation probability.

		Returns:
			numpy.ndarray: Mutated krill.
		"""
		return x


class KrillHerdV2(KrillHerd):
	r"""Implementation of krill herd algorithm.

	Algorithm:
		Krill Herd Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.sciencedirect.com/science/article/pii/S1007570412002171

	Reference paper:
		Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: a new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
	"""
	Name = ['KrillHerdV2', 'KHv2']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Dictionary with testing functions for algorithms parameters.

		See Also:
			* :func:NiaPy.algorithms.basic.kh.KrillHerd.typeParameters`
		"""
		d = KrillHerd.type_parameters()
		d.pop('Mu', None)
		return d

	def mutate(self, x, x_b, Mu):
		r"""Mutate individual.

		Args:
			x (numpy.ndarray): Individual to mutate.
			x_b (numpy.ndarray): Global best individual.
			Mu (float): Mutation probability.

		Returns:
			numpy.ndarray: Mutated individual.
		"""
		return x


class KrillHerdV3(KrillHerd):
	r"""Implementation of krill herd algorithm.

	Algorithm:
		Krill Herd Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.sciencedirect.com/science/article/pii/S1007570412002171

	Reference paper:
		Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: a new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.
	"""
	Name = ['KrillHerdV3', 'KHv3']

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Dictionary with testing functions for algorithms parameters.

		See Also:
			* :func:NiaPy.algorithms.basic.kh.KrillHerd.typeParameters`
		"""
		d = KrillHerd.type_parameters()
		d.pop('Cr', None)
		return d

	def crossover(self, x, xo, Cr):
		r"""Crossover operator.

		Args:
			x (numpy.ndarray): Krill/individual being applied with operator.
			xo (numpy.ndarray): Krill/individual being used in operator.
			Cr (float): Crossover probability.

		Returns:
			numpy.ndarray: Crossover krill/individual.
		"""
		return x


class KrillHerdV11(KrillHerd):
	r"""Implementation of krill herd algorithm.

	Algorithm:
		Krill Herd Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:
	"""
	Name = ['KrillHerdV11', 'KHv11']

	def elitist_selection(self, KH, KH_f, KHo, KHo_f):
		r"""Select krills/individuals that are better than odl krills.

		Args:
			KH (numpy.ndarray): Current herd/population.
			KH_f (numpy.ndarray): Current herd/populations function/fitness values
			KHo (numpy.ndarray): New herd/population.
			KHo_f (numpy.ndarray): New herd/populations function/fitness vales.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New herd/population.
				2. New herd/populations function/fitness values.
		"""
		ipb = np.where(KHo_f >= KH_f)
		KHo[ipb], KHo_f[ipb] = KH[ipb], KH_f[ipb]
		return KHo, KHo_f

	def neighbors(self, i, KH, KH_f, iw, ib, N, W_n, task):
		r"""Neighbors operator.

		Args:
			i (int): Index of krill being applied with operator.
			KH (numpy.ndarray): Current herd/population.
			KH_f (numpy.ndarray): Current herd/populations function/fitness values.
			iw (int): Index of worst krill/individual.
			ib (int): Index of best krill/individual.
			N (numpy.ndarray): TODO
			W_n (numpy.ndarray): Weights for neighbors operator.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: TODO
		"""
		Rgb, RR, Kw_Kgb = KH[ib] - KH[i], KH - KH[i], KH_f[iw] - KH_f[ib]
		R = np.sqrt(np.sum(RR * RR))
		alpha_b = -2 * (1 + self.rand() * task.Iters / task.nGEN) * (KH_f[ib]) / Kw_Kgb / np.sqrt(np.sum(Rgb * Rgb)) * Rgb if KH_f[ib] < KH_f[i] else 0
		alpah_n, nn, ds = 0.0, 0, np.mean(R) / 5
		for n in range(self.NP):
			if R < ds and n != i:
				nn += 1
				if nn <= 4 and KH_f[i] != KH[n]: alpah_n -= (KH(n) - KH[i]) / Kw_Kgb / R[n] * RR[n]
		return W_n * N * self.N_max * (alpha_b + alpah_n)

	def foraging(self, KH, KH_f, KHo, KHo_f, W_f, F, KH_wf, KH_bf, x_food, x_food_f, task):
		r"""Foraging operator.

		Args:
			KH (numpy.ndarray): Current heard/population.
			KH_f (numpy.ndarray): Current herd/populations function/fitness values.
			KHo (numpy.ndarray): New heard/population.
			KHo_f (numpy.ndarray): New heard/population function/fitness values.
			W_f (numpy.ndarray): Weights for foraging.
			F (numpy.ndarray): TODO
			KH_wf (numpy.ndarray): Worst krill in herd/population.
			KH_bf (numpy.ndarray): Best krill in herd/population.
			x_food (numpy.ndarray): Foods position.
			x_food_f (float): Foods function/fitness value.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: TODO.
		"""
		Rf, Kw_Kgb = x_food - KH, KH_wf - KH_bf
		beta_f = -2 * (1 - task.Iters / task.nGEN) * (x_food_f - KH_f) / Kw_Kgb / np.sqrt(np.sum(Rf * Rf)) * Rf if x_food_f < KH_f else 0
		Rib = KHo - KH
		beta_b = -(KHo_f - KH_f) / Kw_Kgb / np.sqrt(np.sum(Rib * Rib)) * Rib if KHo_f < KH_f else 0
		return W_f * F + self.V_f * (beta_b + beta_f)

	def Cr(self, KH_f, KHb_f, KHw_f):
		r"""Calculate crossover probability.

		Args:
			KH_f (float): Krill/individuals function/fitness value.
			KHb_f (float): Best krill/individual function/fitness value.
			KHw_f (float): Worst krill/individual function/fitness value.

		Returns:
			float: Crossover probability.
		"""
		return 0.8 + 0.2 * (KH_f - KHb_f) / (KHw_f - KHb_f)

	def init_population(self, task):
		r"""Initialize the first herd/population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized herd/population.
				2. Initialized herd/populations function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* KHo (numpy.ndarray): TODO
					* KHo_f (float): TODO
					* N (numpy.ndarray): TODO
					* F (numpy.ndarray): TODO
					* Dt (numpy.ndarray): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		KH, KH_f, args, kwargs = Algorithm.init_population(self, task)
		KHo, KHo_f = np.full([self.NP, task.D], task.optType.value * np.inf), np.full(self.NP, task.optType.value * np.inf)
		N, F, Dt = np.full(self.NP, .0), np.full(self.NP, .0), np.mean(task.range()) / 2
		kwargs.update({'KHo': KHo, 'KHo_f': KHo_f, 'N': N, 'F': F, 'Dt': Dt})
		return KH, KH_f, args, kwargs

	def run_iteration(self, task, KH, KH_f, xb, fxb, KHo, KHo_f, N, F, Dt, *args, **kwargs):
		r"""Core function of KrillHerdV11 algorithm.

		Args:
			task (Task): Optimization task.
			KH (numpy.ndarray): Current herd/population.
			KH_f (numpy.ndarray): Current herd/populations function/fitness values.
			xb (numpy.ndarray): Global best krill.
			fxb (float): Global best krill function/fitness value.
			KHo (numpy.ndarray): TODO
			KHo_f (float): TODO
			N (numpy.ndarray): TODO
			F (numpy.ndarray): TODO
			Dt (numpy.ndarray): TODO
			args (list): Additional arguments.
			kwargs (dict): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New herd/population.
				2. New herd/populations function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments.
		"""
		w = np.full(task.D, 0.1 + 0.8 * (1 - task.Iters / task.nGEN))
		ib, iw = np.argmin(KH_f), np.argmax(KH_f)
		x_food, x_food_f = self.get_food_location(KH, KH_f, task)
		xb, fxb = self.get_best(x_food, x_food_f, xb, fxb)
		N = np.asarray([self.neighbors(i, KH, KH_f, iw, ib, N[i], w, task) for i in range(self.NP)])
		F = np.asarray([self.foraging(KH[i], KH_f[i], KHo[i], KHo_f[i], w, F[i], KH_f[iw], KH_f[ib], x_food, x_food_f, task) for i in range(self.NP)])
		Cr = np.asarray([self.Cr(KH_f[i], KH_f[ib], KH_f[iw]) for i in range(self.NP)])
		KH_n = np.asarray([self.crossover(KH[self.randint(self.NP)], KH[i], Cr[i]) for i in range(self.NP)])
		KH_n = KH_n + Dt * (F + N)
		KH = np.apply_along_axis(task.repair, 1, KH_n, self.Rand)
		KH_f = np.apply_along_axis(task.eval, 1, KH)
		KHo, KHo_f = self.elitist_selection(KH, KH_f, KHo, KHo_f)
		xb, fxb = self.get_best(KH, KH_f, xb, fxb)
		return KH, KH_f, xb, fxb, args, {'KHo': KHo, 'KHo_f': KHo_f, 'N': N, 'F': F, 'Dt': Dt}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
