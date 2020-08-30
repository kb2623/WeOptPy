# encoding=utf8

import numpy as np

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.util import full_array

__all__ = [
	'FireworksAlgorithm',
	'EnhancedFireworksAlgorithm',
	'DynamicFireworksAlgorithm',
	'DynamicFireworksAlgorithmGauss',
	'BareBonesFireworksAlgorithm'
]


class BareBonesFireworksAlgorithm(Algorithm):
	r"""Implementation of bare bone fireworks algorithm.

	Algorithm:
		Bare Bones Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S1568494617306609

	Reference paper:
		Junzhi Li, Ying Tan, The bare bones fireworks algorithm: a minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.

	Attributes:
		Name (lsit of str): List of strings representing algorithm names
		n (int): Number of spraks
		C_a (float): amplification coefficient
		C_r (float): reduction coefficient
	"""
	Name = ['BareBonesFireworksAlgorithm', 'BBFWA']

	@staticmethod
	def algorithm_info():
		r"""Get default information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Junzhi Li, Ying Tan, The bare bones fireworks algorithm: a minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046."""

	@staticmethod
	def type_parameters(): return {
		'n': lambda x: isinstance(x, int) and x > 0,
		'C_a': lambda x: isinstance(x, (float, int)) and x > 1,
		'C_r': lambda x: isinstance(x, (float, int)) and 0 < x < 1
	}

	def set_parameters(self, n=10, C_a=1.5, C_r=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (int): Number of sparks :math:`\in [1, \infty)`.
			C_a (float): Amplification coefficient :math:`\in [1, \infty)`.
			C_r (float): Reduction coefficient :math:`\in (0, 1)`.
		"""
		ukwargs.pop('n', None)
		Algorithm.set_parameters(self, n=1, **ukwargs)
		self.n, self.C_a, self.C_r = n, C_a, C_r

	def init_population(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, list, dict]:
				1. Initial solution.
				2. Initial solution function/fitness value.
				3. Additional arguments.
				4. Additional keyword arguments:
					* a (numpy.ndarray): Starting aplitude or search range.
		"""
		x, x_fit, args, d = Algorithm.init_population(self, task)
		d.update({'a': task.bRange})
		return x, x_fit, args, d

	def run_iteration(self, task, x, x_fit, xb, fxb, a, *args, **dparams):
		r"""Core function of Bare Bones Fireworks Algorithm.

		Args:
			task (Task): Optimization task.
			x (numpy.ndarray): Current solution.
			x_fit (float): Current solution fitness/function value.
			xb (numpy.ndarray): Current best solution.
			fxb (float): Current best solution fitness/function value.
			a (numpy.ndarray): Serach range.
			args (list): Additional parameters.
			dparams (dict): Additional keyword parameters.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float, list, dict]:
				1. New solution.
				2. New solution fitness/function value.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* a (numpy.ndarray): Search range.
		"""
		S = np.apply_along_axis(task.repair, 1, self.uniform(x - a, x + a, [self.n, task.D]), self.Rand)
		S_fit = np.apply_along_axis(task.eval, 1, S)
		iS = np.argmin(S_fit)
		if S_fit[iS] < x_fit:x, x_fit, a = S[iS], S_fit[iS], self.C_a * a
		else: a = self.C_r * a
		return x, x_fit, x.copy(), x_fit, args, {'a': a}


class FireworksAlgorithm(Algorithm):
	r"""Implementation of fireworks algorithm.

	Algorithm:
		Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783662463529

	Reference paper:
		Tan, Ying. "Firework Algorithm: a Novel Swarm Intelligence Optimization Method." (2015).

	Attributes:
		Name (List[str]): List of stirngs representing algorithm names.
	"""
	Name = ['FireworksAlgorithm', 'FWA']

	@staticmethod
	def algorithm_info():
		r"""Get default information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Tan, Ying. "Firework Algorithm: a Novel Swarm Intelligence Optimization Method." (2015)."""

	@staticmethod
	def type_parameters(): return {
		'N': lambda x: isinstance(x, int) and x > 0,
		'm': lambda x: isinstance(x, int) and x > 0,
		'a': lambda x: isinstance(x, (int, float)) and x > 0,
		'b': lambda x: isinstance(x, (int, float)) and x > 0,
		'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
	}

	def set_parameters(self, N=40, m=40, a=1, b=2, A=40, epsilon=1e-31, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			N (int): Number of Fireworks
			m (int): Number of sparks
			a (int): Limitation of sparks
			b (int): Limitation of sparks
			A (float): TODO.
			epsilon (float): Small number for non 0 devision
		"""
		ukwargs.pop('n', None)
		Algorithm.set_parameters(self, n=N, **ukwargs)
		self.m, self.a, self.b, self.A, self.epsilon = m, a, b, A, epsilon

	def init_amplitude(self, task):
		r"""Initialize amplitudes for dimensions.

		Args:
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Starting amplitudes.
		"""
		return full_array(self.A, task.D)

	def sparks_no(self, x_f, xw_f, Ss):
		r"""Calculate number of sparks based on function value of individual.

		Args:
			x_f (float): Individuals function/fitness value.
			xw_f (float): Worst individual function/fitness value.
			Ss (): TODO

		Returns:
			int: Number of sparks that individual will create.
		"""
		s = self.m * (xw_f - x_f + self.epsilon) / (Ss + self.epsilon)
		return np.round(self.b * self.m) if s > self.b * self.m and self.a < self.b < 1 else np.round(self.a * self.m)

	def explosion_amplitude(self, x_f, xb_f, A, As):
		r"""Calculate explosion amplitude.

		Args:
			x_f (float): Individuals function/fitness value.
			xb_f (float): Best individuals function/fitness value.
			A (numpy.ndarray): Amplitudes.
			As ():

		Returns:
			numpy.ndarray: TODO.
		"""
		return A * (x_f - xb_f - self.epsilon) / (As + self.epsilon)

	def explode_spark(self, x, A, task):
		r"""Explode a spark.

		Args:
			x (numpy.ndarray): Individuals creating spark.
			A (numpy.ndarray): Amplitude of spark.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Sparks exploded in with specified amplitude.
		"""
		return self.mapping(x + self.rand(task.D) * self.uniform(-A, A, task.D), task)

	def gaussian_spark(self, x, task):
		r"""Create gaussian spark.

		Args:
			x (numpy.ndarray): Individual creating a spark.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Spark exploded based on gaussian amplitude.
		"""
		return self.mapping(x + self.rand(task.D) * self.normal(1, 1, task.D), task)

	def mapping(self, x, task):
		r"""Fix value to bounds..

		Args:
			x (numpy.ndarray): Individual to fix.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Individual in search range.
		"""
		ir = np.where(x > task.upper)
		x[ir] = task.lower[ir] + x[ir] % task.bRange[ir]
		ir = np.where(x < task.lower)
		x[ir] = task.lower[ir] + x[ir] % task.bRange[ir]
		return x

	def R(self, x, FW):
		r"""Calculate ranges.

		Args:
			x (numpy.ndarray): Individual in population.
			FW (numpy.ndarray): Current population.

		Returns:
			numpy,ndarray: Ranges values.
		"""
		return np.sqrt(np.sum(np.fabs(x - FW)))

	def p(self, r, Rs):
		r"""Calculate p.

		Args:
			r (float): Range of individual.
			Rs (float): Sum of ranges.

		Returns:
			float: p value.
		"""
		return r / Rs

	def next_generation(self, FW, FW_f, FWn, task):
		r"""Generate new generation of individuals.

		Args:
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray[float]): Currents population fitness/function values.
			FWn (numpy.ndarray): New population.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float]]:
				1. New population.
				2. New populations fitness/function values.
		"""
		FWn_f = np.apply_along_axis(task.eval, 1, FWn)
		ib = np.argmin(FWn_f)
		if FWn_f[ib] < FW_f[0]: FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		R = np.asarray([self.R(FWn[i], FWn) for i in range(len(FWn))])
		Rs = np.sum(R)
		P = np.asarray([self.p(R[i], Rs) for i in range(len(FWn))])
		isort = np.argsort(P)[-(self.NP - 1):]
		FW[1:], FW_f[1:] = np.asarray(FWn)[isort], FWn_f[isort]
		return FW, FW_f

	def init_population(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations function/fitness values.
				3. Additional arguments:
					* Ah (numpy.ndarray): Initialized amplitudes.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		FW, FW_f, d = Algorithm.init_population(self, task)
		Ah = self.init_amplitude(task)
		d.update({'Ah': Ah})
		return FW, FW_f, d

	def run_iteration(self, task, FW, FW_f, xb, fxb, Ah, *args, **dparams):
		r"""Core function of Fireworks algorithm.

		Args:
			task (Task): Optimization task.
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray[float]): Current populations function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals fitness/function value.
			Ah (numpy.ndarray): Current amplitudes.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. Initialized population.
				2. Initialized populations function/fitness values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* Ah (numpy.ndarray): Initialized amplitudes.

		See Also:
			* :func:`FireworksAlgorithm.SparsksNo`.
			* :func:`FireworksAlgorithm.ExplosionAmplitude`
			* :func:`FireworksAlgorithm.ExplodeSpark`
			* :func:`FireworksAlgorithm.GaussianSpark`
			* :func:`FireworksAlgorithm.NextGeneration`
		"""
		iw, ib = np.argmax(FW_f), 0
		Ss, As = np.sum(FW_f[iw] - FW_f), np.sum(FW_f - FW_f[ib])
		S = [self.sparks_no(FW_f[i], FW_f[iw], Ss) for i in range(self.NP)]
		A = [self.explosion_amplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(self.NP)]
		FWn = [self.explode_spark(FW[i], A[i], task) for i in range(self.NP) for _ in range(S[i])]
		for i in range(self.m): FWn.append(self.gaussian_spark(self.randint(self.NP), task))
		FW, FW_f = self.next_generation(FW, FW_f, FWn, task)
		xb, fxb = self.get_best(FW, FW_f, xb, fxb)
		return FW, FW_f, xb, fxb, args, {'Ah': Ah}


class EnhancedFireworksAlgorithm(FireworksAlgorithm):
	r"""Implementation of enganced fireworks algorithm.

	Algorithm:
		Enhanced Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/6557813/

	Reference paper:
		S. Zheng, a. Janecek and Y. Tan, "Enhanced Fireworks Algorithm," 2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2069-2077. doi: 10.1109/CEC.2013.6557813

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		Ainit (float): Initial amplitude of sparks.
		Afinal (float): Maximal amplitude of sparks.
	"""
	Name = ['EnhancedFireworksAlgorithm', 'EFWA']

	@staticmethod
	def algorithm_info():
		r"""Get default information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""S. Zheng, a. Janecek and Y. Tan, "Enhanced Fireworks Algorithm," 2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2069-2077. doi: 10.1109/CEC.2013.6557813"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* Ainit (Callable[[Union[int, float]], bool]): TODO
				* Afinal (Callable[[Union[int, float]], bool]): TODO

		See Also:
			* :func:`FireworksAlgorithm.typeParameters`
		"""
		d = FireworksAlgorithm.type_parameters()
		d['Ainit'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['Afinal'] = lambda x: isinstance(x, (float, int)) and x > 0
		return d

	def set_parameters(self, Ainit=20, Afinal=5, **ukwargs):
		r"""Set EnhancedFireworksAlgorithm algorithms core parameters.

		Args:
			Ainit (float): TODO
			Afinal (float): TODO
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`FireworksAlgorithm.setParameters`
		"""
		FireworksAlgorithm.set_parameters(self, **ukwargs)
		self.Ainit, self.Afinal = Ainit, Afinal

	def init_ranges(self, task):
		r"""Initialize ranges.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
				1. Initial amplitude values over dimensions.
				2. Final amplitude values over dimensions.
				3. uAmin.
		"""
		Ainit, Afinal = full_array(self.Ainit, task.D), full_array(self.Afinal, task.D)
		return Ainit, Afinal, self.uAmin(Ainit, Afinal, task)

	def uAmin(self, Ainit, Afinal, task):
		r"""Calculate the value of `uAmin`.

		Args:
			Ainit (numpy.ndarray[float]): Initial amplitude values over dimensions.
			Afinal (numpy.ndarray[float]): Final amplitude values over dimensions.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: uAmin.
		"""
		return Ainit - np.sqrt(task.Evals * (2 * task.nFES - task.Evals)) * (Ainit - Afinal) / task.nFES

	def explosion_amplitude(self, x_f, xb_f, Ah, As, A_min=None):
		r"""Calculate explosion amplitude.

		Args:
			x_f (float): Individuals function/fitness value.
			xb_f (float): Best individual function/fitness value.
			Ah (numpy.ndarray):
			As (): TODO.
			A_min (Optional[numpy.ndarray]): Minimal amplitude values.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New amplitude.
		"""
		A = FireworksAlgorithm.explosion_amplitude(self, x_f, xb_f, Ah, As)
		ifix = np.where(A < A_min)
		A[ifix] = A_min[ifix]
		return A

	def gaussian_spark(self, x, xb, task):
		r"""Create new individual.

		Args:
			x (numpy.ndarray):
			xb (numpy.ndarray):
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New individual generated by gaussian noise.
		"""
		return self.mapping(x + self.rand(task.D) * (xb - x) * self.normal(1, 1, task.D), task)

	def next_generation(self, FW, FW_f, FWn, task):
		r"""Generate new population.

		Args:
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray): Current populations fitness/function values.
			FWn (numpy.ndarray): New population.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New population.
				2. New populations fitness/function values.
		"""
		FWn_f = np.apply_along_axis(task.eval, 1, FWn)
		ib = np.argmin(FWn_f)
		if FWn_f[ib] < FW_f[0]: FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		for i in range(1, self.NP):
			r = self.randint(len(FWn))
			if FWn_f[r] < FW_f[i]: FW[i], FW_f[i] = FWn[r], FWn_f[r]
		return FW, FW_f

	def init_population(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initial population.
				2. Initial populations fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* Ainit (numpy.ndarray): Initial amplitude values.
					* Afinal (numpy.ndarray): Final amplitude values.
					* A_min (numpy.ndarray): Minimal amplitude values.

		See Also:
			* :func:`FireworksAlgorithm.initPopulation`
		"""
		FW, FW_f, args, d = FireworksAlgorithm.init_population(self, task)
		Ainit, Afinal, A_min = self.init_ranges(task)
		d.update({'Ainit': Ainit, 'Afinal': Afinal, 'A_min': A_min})
		return FW, FW_f, args, d

	def run_iteration(self, task, FW, FW_f, xb, fxb, Ah, Ainit, Afinal, A_min, *args, **dparams):
		r"""Core function of EnhancedFireworksAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray[float]): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function/fitness value.
			Ah (numpy.ndarray[float]): Current amplitude.
			Ainit (numpy.ndarray[float]): Initial amplitude.
			Afinal (numpy.ndarray[float]): Final amplitude values.
			A_min (numpy.ndarray[float]): Minial amplitude values.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. Initial population.
				2. Initial populations fitness/function values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* Ainit (numpy.ndarray): Initial amplitude values.
					* Afinal (numpy.ndarray): Final amplitude values.
					* A_min (numpy.ndarray): Minimal amplitude values.
		"""
		iw, ib = np.argmax(FW_f), 0
		Ss, As = np.sum(FW_f[iw] - FW_f), np.sum(FW_f - FW_f[ib])
		S = [self.sparks_no(FW_f[i], FW_f[iw], Ss) for i in range(self.NP)]
		A = [self.explosion_amplitude(FW_f[i], FW_f[ib], Ah, As, A_min) for i in range(self.NP)]
		A_min = self.uAmin(Ainit, Afinal, task)
		FWn = [self.explode_spark(FW[i], A[i], task) for i in range(self.NP) for _ in range(S[i])]
		for i in range(self.m): FWn.append(self.gaussian_spark(self.randint(self.NP), FW[ib], task))
		FW, FW_f = self.next_generation(FW, FW_f, FWn, task)
		xb, fxb = self.get_best(FW, FW_f, xb, fxb)
		return FW, FW_f, xb, fxb, args, {'Ah': Ah, 'Ainit': Ainit, 'Afinal': Afinal, 'A_min': A_min}


class DynamicFireworksAlgorithmGauss(EnhancedFireworksAlgorithm):
	r"""Implementation of dynamic fireworks algorithm.

	Algorithm:
		Dynamic Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223

	Reference paper:
		S. Zheng, a. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		A_cf (Union[float, int]): TODO
		C_a (Union[float, int]): Amplification factor.
		C_r (Union[float, int]): Reduction factor.
		epsilon (Union[float, int]): Small value.
	"""
	Name = ['DynamicFireworksAlgorithmGauss', 'dynFWAG']

	@staticmethod
	def algorithm_info():
		r"""Get default information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""S. Zheng, a. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* A_cr (Callable[[Union[float, int], bool]): TODo

		See Also:
			* :func:`FireworksAlgorithm.typeParameters`
		"""
		d = FireworksAlgorithm.type_parameters()
		d['A_cf'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['C_a'] = lambda x: isinstance(x, (float, int)) and x > 1
		d['C_r'] = lambda x: isinstance(x, (float, int)) and 0 < x < 1
		d['epsilon'] = lambda x: isinstance(x, (float, int)) and 0 < x < 1
		return d

	def set_parameters(self, A_cf=20, C_a=1.2, C_r=0.9, epsilon=1e-8, **ukwargs):
		r"""Set core arguments of DynamicFireworksAlgorithmGauss.

		Args:
			A_cf (Union[int, float]):
			C_a (Union[int, float]):
			C_r (Union[int, float]):
			epsilon (Union[int, float]):
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`FireworksAlgorithm.setParameters`
		"""
		FireworksAlgorithm.set_parameters(self, **ukwargs)
		self.A_cf, self.C_a, self.C_r, self.epsilon = A_cf, C_a, C_r, epsilon

	def init_amplitude(self, task):
		r"""Initialize amplitude.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. Initial amplitudes.
				2. Amplitude for best spark.
		"""
		return FireworksAlgorithm.init_amplitude(self, task), task.bRange

	def mapping(self, x, task):
		r"""Fix out of bound solution/individual.

		Args:
			x (numpy.ndarray): Individual.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Fixed individual.
		"""
		ir = np.where(x > task.upper)
		x[ir] = self.uniform(task.lower[ir], task.upper[ir])
		ir = np.where(x < task.lower)
		x[ir] = self.uniform(task.lower[ir], task.upper[ir])
		return x

	def repair(self, x, d, epsilon):
		r"""Repair solution.

		Args:
			x (numpy.ndarray): Individual.
			d (numpy.ndarray): Default value.
			epsilon (float): Limiting value.

		Returns:
			numpy.ndarray: Fixed solution.
		"""
		ir = np.where(x <= epsilon)
		x[ir] = d[ir]
		return x

	def next_generation(self, FW, FW_f, FWn, task):
		r"""TODO.

		Args:
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray[float]): Current populations function/fitness values.
			FWn (numpy.ndarray): New population.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New population.
				2. New populations function/fitness values.
		"""
		FWn_f = np.apply_along_axis(task.eval, 1, FWn)
		ib = np.argmin(FWn_f)
		for i, f in enumerate(FW_f):
			r = self.randint(len(FWn))
			if FWn_f[r] < f: FW[i], FW_f[i] = FWn[r], FWn_f[r]
		FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		return FW, FW_f

	def uCF(self, xnb, xcb, xcb_f, xb, xb_f, Acf, task):
		r"""TODO.

		Args:
			xnb (numpy.ndarray): TODO.
			xcb (numpy.ndarray): TODO.
			xcb_f (float): TODO.
			xb (numpy.ndarray): TODO.
			xb_f (float): TODO.
			Acf (): TODO.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray]:
				1. TODO
		"""
		xnb_f = np.apply_along_axis(task.eval, 1, xnb)
		ib_f = np.argmin(xnb_f)
		if xnb_f[ib_f] <= xb_f: xb, xb_f = xnb[ib_f], xnb_f[ib_f]
		Acf = self.repair(Acf, task.bRange, self.epsilon)
		if xb_f >= xcb_f: xb, xb_f, Acf = xcb, xcb_f, Acf * self.C_a
		else: Acf = Acf * self.C_r
		return xb, xb_f, Acf

	def explosion_amplitude(self, x_f, xb_f, Ah, As, A_min=None):
		return FireworksAlgorithm.explosion_amplitude(self, x_f, xb_f, Ah, As)

	def init_population(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population.
				2. Initialized population function/fitness values.
				3. Additional arguments:
				4. Additional keyword arguments:
					* Ah (): TODO
					* Ab (): TODO
		"""
		FW, FW_f, args, _ = Algorithm.init_population(self, task)
		Ah, Ab = self.init_amplitude(task)
		return FW, FW_f, args, {'Ah': Ah, 'Ab': Ab}

	def run_iteration(self, task, FW, FW_f, xb, fxb, Ah, Ab, *args, **dparams):
		r"""Core function of DynamicFireworksAlgorithmGauss algorithm.

		Args:
			task (Task): Optimization task.
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best fitness/function value.
			Ah (Union[numpy.ndarray, float]): TODO
			Ab (Union[numpy.ndarray, float]): TODO
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
				5. Additional keyword arguments:
					* Ah (Union[numpy.ndarray, float]): TODO
					* Ab (Union[numpy.ndarray, float]): TODO
		"""
		iw, ib = np.argmax(FW_f), np.argmin(FW_f)
		Ss, As = np.sum(FW_f[iw] - FW_f), np.sum(FW_f - FW_f[ib])
		S, sb = [self.sparks_no(FW_f[i], FW_f[iw], Ss) for i in range(len(FW))], self.sparks_no(fxb, FW_f[iw], Ss)
		A = [self.explosion_amplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(len(FW))]
		FWn, xnb = [self.explode_spark(FW[i], A[i], task) for i in range(self.NP) for _ in range(S[i])], [self.explode_spark(xb, Ab, task) for _ in range(sb)]
		for i in range(self.m): FWn.append(self.gaussian_spark(self.randint(self.NP), FW[ib], task))
		FW, FW_f = self.next_generation(FW, FW_f, FWn, task)
		iw, ib = np.argmax(FW_f), 0
		xb, fxb, Ab = self.uCF(xnb, FW[ib], FW_f[ib], xb, fxb, Ab, task)
		return FW, FW_f, xb, fxb, args, {'Ah': Ah, 'Ab': Ab}


class DynamicFireworksAlgorithm(DynamicFireworksAlgorithmGauss):
	r"""Implementation of dynamic fireworks algorithm.

	Algorithm:
		Dynamic Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223

	Reference paper:
		S. Zheng, a. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.basic.DynamicFireworksAlgorithmGauss`
	"""
	Name = ['DynamicFireworksAlgorithm', 'dynFWA']

	@staticmethod
	def algorithm_info():
		r"""Get default information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""S. Zheng, a. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485"""

	def run_iteration(self, task, FW, FW_f, xb, fxb, Ah, Ab, *args, **dparams):
		r"""Core function of Dynamic Fireworks Algorithm.

		Args:
			task (Task): Optimization task
			FW (numpy.ndarray): Current population
			FW_f (numpy.ndarray[float]): Current population fitness/function values
			xb (numpy.ndarray): Current best solution
			fxb (float): Current best solution's fitness/function value
			Ah (): TODO
			Ab (): TODO
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], list, dict]:
				1. New population.
				2. New population function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* Ah (): TODO
					* Ab (): TODO
		"""
		iw, ib = np.argmax(FW_f), np.argmin(FW_f)
		Ss, As = np.sum(FW_f[iw] - FW_f), np.sum(FW_f - FW_f[ib])
		S, sb = [self.sparks_no(FW_f[i], FW_f[iw], Ss) for i in range(len(FW))], self.sparks_no(fxb, FW_f[iw], Ss)
		A = [self.explosion_amplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(len(FW))]
		FWn, xnb = [self.explode_spark(FW[i], A[i], task) for i in range(self.NP) for _ in range(S[i])], [self.explode_spark(xb, Ab, task) for _ in range(sb)]
		FW, FW_f = self.next_generation(FW, FW_f, FWn, task)
		iw, ib = np.argmax(FW_f), 0
		xb, fxb, Ab = self.uCF(xnb, FW[ib], FW_f[ib], xb, fxb, Ab, task)
		return FW, FW_f, xb, fxb, args, {'Ah': Ah, 'Ab': Ab}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
