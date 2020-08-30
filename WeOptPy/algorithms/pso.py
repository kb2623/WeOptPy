# encoding=utf8

import numpy as np

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.util import full_array
from WeOptPy.util.utility import reflect_repair

__all__ = [
	'ParticleSwarmAlgorithm',
	'ParticleSwarmOptimization',
	'CenterParticleSwarmOptimization',
	'MutatedParticleSwarmOptimization',
	'MutatedCenterParticleSwarmOptimization',
	'ComprehensiveLearningParticleSwarmOptimizer',
	'MutatedCenterUnifiedParticleSwarmOptimization',
	'OppositionVelocityClampingParticleSwarmOptimization'
]


class ParticleSwarmAlgorithm(Algorithm):
	r"""Implementation of Particle Swarm Optimization algorithm.

	Algorithm:
		Particle Swarm Optimization algorithm

	Date:
		2018

	Authors:
		Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		TODO: Find the right paper

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		C1 (float): Cognitive component.
		C2 (float): Social component.
		w (Union[float, numpy.ndarray[float]]): Inertial weight.
		vMin (Union[float, numpy.ndarray[float]]): Minimal velocity.
		vMax (Union[float, numpy.ndarray[float]]): Maximal velocity.
		Repair (Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, mtrnd.RandomState], numpy.ndarray]): Repair method for velocity.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['WeightedVelocityClampingParticleSwarmAlgorithm', 'WVCPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""TODO find one"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable[[Union[int, float]], bool]]:
				* n (Callable[[int], bool])
				* C1 (Callable[[Union[int, float]], bool])
				* C2 (Callable[[Union[int, float]], bool])
				* w (Callable[[float], bool])
				* vMin (Callable[[Union[int, float]], bool])
				* vMax (Callable[[Union[int, float], bool])
		"""
		d = Algorithm.type_parameters()
		d.update({
			'C1': lambda x: isinstance(x, (int, float)) and x >= 0,
			'C2': lambda x: isinstance(x, (int, float)) and x >= 0,
			'w': lambda x: isinstance(x, float) and x >= 0,
			'min_velocity': lambda x: isinstance(x, (int, float)),
			'max_velocity': lambda x: isinstance(x, (int, float))
		})
		return d

	def set_parameters(self, n=25, c1=2.0, c2=2.0, w=0.7, min_velocity=-1.5, max_velocity=1.5, repair=reflect_repair, **ukwargs):
		r"""Set Particle Swarm Algorithm main parameters.

		Args:
			n (int): Population size
			c1 (float): Cognitive component.
			c2 (float): Social component.
			w (Union[float, numpy.ndarray]): Inertial weight.
			min_velocity (Union[float, numpy.ndarray]): Minimal velocity.
			max_velocity (Union[float, numpy.ndarray]): Maximal velocity.
			repair (Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray]): Repair method for velocity.
			**ukwargs: Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.C1, self.C2, self.w, self.vMin, self.vMax, self.Repair = c1, c2, w, min_velocity, max_velocity, repair

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.getParameters`
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'c1': self.C1,
			'c2': self.C2,
			'w': self.w,
			'min_velocity': self.vMin,
			'max_velocity': self.vMax
		})
		return d

	def init(self, task):
		r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Dict[str, Union[float, numpy.ndarray]]:
				* w (numpy.ndarray): Inertial weight.
				* vMin (numpy.ndarray): Minimal velocity.
				* vMax (numpy.ndarray): Maximal velocity.
				* V (numpy.ndarray): Initial velocity of particle.
		"""
		return {
			'w': full_array(self.w, task.D),
			'min_velocity': full_array(self.vMin, task.D),
			'max_velocity': full_array(self.vMax, task.D),
			'v': np.full([self.NP, task.D], 0.0)
		}

	def init_population(self, task):
		r"""Initialize population and dynamic arguments of the Particle Swarm Optimization algorithm.

		Args:
			task: Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initial population.
				2. Initial population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* popb (numpy.ndarray): particles best population.
					* fpopb (numpy.ndarray[float]): particles best positions function/fitness value.
					* w (numpy.ndarray): Inertial weight.
					* min_velocity (numpy.ndarray): Minimal velocity.
					* max_velocity (numpy.ndarray): Maximal velocity.
					* V (numpy.ndarray): Initial velocity of particle.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		pop, fpop, args, d = Algorithm.init_population(self, task)
		d.update(self.init(task))
		d.update({'popb': pop.copy(), 'fpopb': fpop.copy()})
		return pop, fpop, args, d

	def update_velocity(self, v, p, pb, gb, w, min_velocity, max_velocity, task, **kwargs):
		r"""Update particle velocity.

		Args:
			v (numpy.ndarray): Current velocity of particle.
			p (numpy.ndarray): Current position of particle.
			pb (numpy.ndarray): Personal best position of particle.
			gb (numpy.ndarray): Global best position of particle.
			w (numpy.ndarray): Weights for velocity adjustment.
			min_velocity (numpy.ndarray): Minimal velocity allowed.
			max_velocity (numpy.ndarray): Maximal velocity allowed.
			task (Task): Optimization task.
			kwargs: Additional arguments.

		Returns:
			numpy.ndarray: Updated velocity of particle.
		"""
		return self.Repair(w * v + self.C1 * self.rand(task.D) * (pb - p) + self.C2 * self.rand(task.D) * (gb - p), min_velocity, max_velocity)

	def run_iteration(self, task, pop, fpop, xb, fxb, popb, fpopb, w, min_velocity, max_velocity, v, *args, **dparams):
		r"""Core function of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current populations.
			fpop (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Current best particle.
			fxb (float): Current best particle fitness/function value.
			popb (numpy.ndarray): Particles best position.
			fpopb (numpy.ndarray): Particles best positions fitness/function values.
			w (numpy.ndarray): Inertial weights.
			min_velocity (numpy.ndarray): Minimal velocity.
			max_velocity (numpy.ndarray): Maximal velocity.
			v (numpy.ndarray): Velocity of particles.
			args (list): Additional function arguments.
			dparams (dict): Additional function keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
				1. New population.
				2. New population fitness/function values.
				3. New global best position.
				4. New global best positions function/fitness value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* popb (numpy.ndarray): Particles best population.
					* fpopb (numpy.ndarray[float]): Particles best positions function/fitness value.
					* w (numpy.ndarray): Inertial weight.
					* vMin (numpy.ndarray): Minimal velocity.
					* vMax (numpy.ndarray): Maximal velocity.
					* V (numpy.ndarray): Initial velocity of particle.

		See Also:
			* :class:`NiaPy.algorithms.algorithm.runIteration`
		"""
		for i in range(len(pop)):
			v[i] = self.update_velocity(v[i], pop[i], popb[i], xb, w, min_velocity, max_velocity, task)
			pop[i] = task.repair(pop[i] + v[i], rnd=self.Rand)
			fpop[i] = task.eval(pop[i])
			if fpop[i] < fpopb[i]: popb[i], fpopb[i] = pop[i].copy(), fpop[i]
			if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
		return pop, fpop, xb, fxb, args, {'popb': popb, 'fpopb': fpopb, 'w': w, 'min_velocity': min_velocity, 'max_velocity': max_velocity, 'v': v}


class ParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Particle Swarm Optimization algorithm.

	Algorithm:
		Particle Swarm Optimization algorithm

	Date:
		2018

	Authors:
		Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		C1 (float): Cognitive component.
		C2 (float): Social component.
		Repair (Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, mtrnd.RandomState], numpy.ndarray]): Repair method for velocity.

	See Also:
		* :class:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`
	"""
	Name = ['ParticleSwarmAlgorithm', 'PSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable[[Union[int, float]], bool]]:
			* n: Population size.
			* C1: Cognitive component.
			* C2: Social component.
		"""
		d = ParticleSwarmAlgorithm.type_parameters()
		d.pop('w', None), d.pop('vMin', None), d.pop('vMax', None)
		return d

	def set_parameters(self, **ukwargs):
		r"""Set core parameters of algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional parameters.

		See Also:
			* :func:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm.setParameters`
		"""
		ukwargs.pop('w', None), ukwargs.pop('vMin', None), ukwargs.pop('vMax', None)
		ParticleSwarmAlgorithm.set_parameters(self, w=1, min_velocity=-np.inf, max_velocity=np.inf, **ukwargs)

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.get_parameters(self)
		d.pop('w', None), d.pop('vMin', None), d.pop('vMax', None)
		return d


class OppositionVelocityClampingParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Opposition-Based Particle Swarm Optimization with Velocity Clamping.

	Algorithm:
		Opposition-Based Particle Swarm Optimization with Velocity Clamping

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Shahzad, Farrukh, et al. "Opposition-based particle swarm optimization with velocity clamping (OVCPSO)." Advances in Computational Intelligence. Springer, Berlin, Heidelberg, 2009. 339-348

	Attributes:
		p0: Probability of opposite learning phase.
		w_min: Minimum inertial weight.
		w_max: Maximum inertial weight.
		sigma: Velocity scaling factor.

	See Also:
		* :class:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm`
	"""
	Name = ['OppositionVelocityClampingParticleSwarmOptimization', 'OVCPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Shahzad, Farrukh, et al. "Opposition-based particle swarm optimization with velocity clamping (OVCPSO)." Advances in Computational Intelligence. Springer, Berlin, Heidelberg, 2009. 339-348"""

	def set_parameters(self, p0=.3, w_min=.4, w_max=.9, sigma=.1, c1=1.49612, c2=1.49612, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			p0 (float): Probability of running Opposite learning.
			w_min (numpy.ndarray): Minimal value of weights.
			w_max (numpy.ndarray): Maximum value of weights.
			sigma (numpy.ndarray): Velocity range factor.
			c1 (float): Cognitive component.
			c2 (float): Social component.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.ParticleSwarmAlgorithm.setParameters`
		"""
		kwargs.pop('w', None)
		ParticleSwarmAlgorithm.set_parameters(self, w=w_max, c1=c1, c2=c2, **kwargs)
		self.p0, self.w_min, self.w_max, self.sigma = p0, w_min, w_max, sigma

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.get_parameters(self)
		d.pop('vMin', None), d.pop('vMax', None)
		d.update({
			'p0': self.p0, 'w_min': self.w_min, 'w_max': self.w_max, 'sigma': self.sigma
		})
		return d

	def opposite_learning(self, s_l, s_h, pop, fpop, task):
		r"""Run opposite learning phase.

		Args:
			s_l (numpy.ndarray): lower limit of opposite particles.
			s_h (numpy.ndarray): upper limit of opposite particles.
			pop (numpy.ndarray): Current populations positions.
			fpop (numpy.ndarray): Current populations functions/fitness values.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
				1. New particles position
				2. New particles function/fitness values
				3. New best position of opposite learning phase
				4. new best function/fitness value of opposite learning phase
		"""
		s_r = s_l + s_h
		s = np.asarray([s_r - e for e in pop])
		s_f = np.asarray([task.eval(e) for e in s])
		s, s_f = np.concatenate([pop, s]), np.concatenate([fpop, s_f])
		sinds = np.argsort(s_f)
		return s[sinds[:len(pop)]], s_f[sinds[:len(pop)]], s[sinds[0]], s_f[sinds[0]]

	def init_population(self, task):
		r"""Init starting population and dynamic parameters.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population.
				2. Initialized populations function/fitness values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* popb (numpy.ndarray): particles best population.
					* fpopb (numpy.ndarray[float]): particles best positions function/fitness value.
					* vMin (numpy.ndarray): Minimal velocity.
					* vMax (numpy.ndarray): Maximal velocity.
					* V (numpy.ndarray): Initial velocity of particle.
					* S_u (numpy.ndarray): upper bound for opposite learning.
					* S_l (numpy.ndarray): lower bound for opposite learning.
		"""
		pop, fpop, args, d = ParticleSwarmAlgorithm.init_population(self, task)
		s_l, s_h = task.Lower, task.Upper
		pop, fpop, _, _ = self.opposite_learning(s_l, s_h, pop, fpop, task)
		pb_inds = np.where(fpop < d['fpopb'])
		d['popb'][pb_inds], d['fpopb'][pb_inds] = pop[pb_inds], fpop[pb_inds]
		d['min_velocity'], d['max_velocity'] = self.sigma * (task.Upper - task.Lower), self.sigma * (task.Lower - task.Upper)
		d.update({'s_l': s_l, 's_h': s_h})
		return pop, fpop, args, d

	def run_iteration(self, task, pop, fpop, xb, fxb, popb, fpopb, min_velocity, max_velocity, v, s_l, s_h, *args, **dparams):
		r"""Core function of Opposite-based Particle Swarm Optimization with velocity clamping algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray): Current global best position.
			fxb (float): Current global best positions function/fitness value.
			popb (numpy.ndarray): Personal best position.
			fpopb (numpy.ndarray): Personal best positions function/fitness values.
			min_velocity (numpy.ndarray): Minimal allowed velocity.
			max_velocity (numpy.ndarray): Maximal allowed velocity.
			v (numpy.ndarray): Populations velocity.
			s_l (numpy.ndarray): lower bound of opposite learning.
			s_h (numpy.ndarray): upper bound of opposite learning.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New populations function/fitness values.
				3. New global best position.
				4. New global best positions function/fitness value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* popb: particles best population.
					* fpopb: particles best positions function/fitness value.
					* min_velocity: Minimal velocity.
					* max_velocity: Maximal velocity.
					* V: Initial velocity of particle.
					* s_u: upper bound for opposite learning.
					* s_l: lower bound for opposite learning.
		"""
		if self.rand() < self.p0:
			pop, fpop, nb, fnb = self.opposite_learning(s_l, s_h, pop, fpop, task)
			pb_inds = np.where(fpop < fpopb)
			popb[pb_inds], fpopb[pb_inds] = pop[pb_inds], fpop[pb_inds]
			if fnb < fxb: xb, fxb = nb.copy(), fnb
		else:
			w = self.w_max - ((self.w_max - self.w_min) / task.nGEN) * task.Iters
			for i in range(len(pop)):
				v[i] = self.update_velocity(v[i], pop[i], popb[i], xb, w, min_velocity, max_velocity, task)
				pop[i] = task.repair(pop[i] + v[i], rnd=self.Rand)
				fpop[i] = task.eval(pop[i])
				if fpop[i] < fpopb[i]:
					popb[i], fpopb[i] = pop[i].copy(), fpop[i]
					if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
			min_velocity, max_velocity = self.sigma * np.min(pop, axis=0), self.sigma * np.max(pop, axis=0)
		return pop, fpop, xb, fxb, args, {'popb': popb, 'fpopb': fpopb, 'min_velocity': min_velocity, 'max_velocity': max_velocity, 'v': v, 's_l': s_l, 's_h': s_h}


class CenterParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Center Particle Swarm Optimization.

	Algorithm:
		Center Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		H.-C. Tsai, Predicting strengths of concrete-type specimens using hybrid multilayer perceptrons with center-Unified particle swarm optimization, Adv. Eng. Softw. 37 (2010) 1104–1112.

	See Also:
		* :class:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`
	"""
	Name = ['CenterParticleSwarmOptimization', 'CPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""H.-C. Tsai, Predicting strengths of concrete-type specimens using hybrid multilayer perceptrons with center-Unified particle swarm optimization, Adv. Eng. Softw. 37 (2010) 1104–1112."""

	def set_parameters(self, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			**kwargs: Additional arguments.

		See Also:
			:func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.setParameters`
		"""
		kwargs.pop('vMin', None), kwargs.pop('vMax', None)
		ParticleSwarmAlgorithm.set_parameters(self, min_velocity=-np.inf, max_velocity=np.inf, **kwargs)

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.get_parameters(self)
		d.pop('vMin', None), d.pop('vMax', None)
		return d

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population of particles.
			fpop (numpy.ndarray): Current particles function/fitness values.
			xb (numpy.ndarray): Current global best particle.
			fxb (numpy.ndarray): Current global best particles function/fitness value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
				1. New population of particles.
				2. New populations function/fitness values.
				3. New global best particle.
				4. New global best particle function/fitness value.
				5. Additional arguments.
				6. Additional keyword arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.runIteration`
		"""
		pop, fpop, xb, fxb, args, d = ParticleSwarmAlgorithm.run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams)
		c = np.sum(pop, axis=0) / len(pop)
		fc = task.eval(c)
		if fc <= fxb: xb, fxb = c, fc
		return pop, fpop, xb, fxb, args, d


class MutatedParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Mutated Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		H. Wang, C. Li, Y. Liu, S. Zeng, a hybrid particle swarm algorithm with cauchy mutation, Proceedings of the 2007 IEEE Swarm Intelligence Symposium (2007) 356–360.

	Attributes:
		nmutt (int): Number of mutations of global best particle.

	See Also:
		* :class:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`
	"""
	Name = ['MutatedParticleSwarmOptimization', 'MPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""H. Wang, C. Li, Y. Liu, S. Zeng, a hybrid particle swarm algorithm with cauchy mutation, Proceedings of the 2007 IEEE Swarm Intelligence Symposium (2007) 356–360."""

	def set_parameters(self, nmutt=10, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			nmutt (int): Number of mutations of global best particle.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.setParameters`
		"""
		kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
		ParticleSwarmAlgorithm.set_parameters(self, min_velocity=-np.inf, max_velocity=np.inf, **kwargs)
		self.nmutt = nmutt

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.get_parameters(self)
		d.pop('min_velocity', None), d.pop('max_velocity', None)
		d.update({'nmutt': self.nmutt})
		return d

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population of particles.
			fpop (numpy.ndarray): Current particles function/fitness values.
			xb (numpy.ndarray): Current global best particle.
			fxb (float): Current global best particles function/fitness value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population of particles.
				2. New populations function/fitness values.
				3. New global best particle.
				4. New global best particle function/fitness value.
				5. Additional arguments.
				6. Additional keyword arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.runIteration`
		"""
		pop, fpop, xb, fxb, args, d = ParticleSwarmAlgorithm.run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams)
		v = d['V']
		v_a = (np.sum(v, axis=0) / len(v))
		v_a = v_a / np.max(np.abs(v_a))
		for _ in range(self.nmutt):
			g = task.repair(xb + v_a * self.uniform(task.Lower, task.Upper), self.Rand)
			fg = task.eval(g)
			if fg <= fxb: xb, fxb = g, fg
		return pop, fpop, xb, fxb, args, d


class MutatedCenterParticleSwarmOptimization(CenterParticleSwarmOptimization):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Mutated Center Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		TODO find one

	Attributes:
		nmutt (int): Number of mutations of global best particle.

	See Also:
		* :class:`NiaPy.algorithms.basic.CenterParticleSwarmOptimization`
	"""
	Name = ['MutatedCenterParticleSwarmOptimization', 'MCPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""TODO find one"""

	def set_parameters(self, nmutt=10, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			nmutt (int): Number of mutations of global best particle.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.CenterParticleSwarmOptimization.setParameters`
		"""
		kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
		ParticleSwarmAlgorithm.set_parameters(self, min_velocity=-np.inf, max_velocity=np.inf, **kwargs)
		self.nmutt = nmutt

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.CenterParticleSwarmOptimization.getParameters`
		"""
		d = CenterParticleSwarmOptimization.get_parameters(self)
		d.update({'nmutt': self.nmutt})
		return d

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population of particles.
			fpop (numpy.ndarray): Current particles function/fitness values.
			xb (numpy.ndarray): Current global best particle.
			fxb (float: Current global best particles function/fitness value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population of particles.
				2. New populations function/fitness values.
				3. New global best particle.
				4. New global best particle function/fitness value.
				5. Additional arguments.
				6. Additional keyword arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.runIteration`
		"""
		pop, fpop, xb, fxb, args, d = CenterParticleSwarmOptimization.run_iteration(self, task, pop, fpop, xb, fxb, **dparams)
		v = d['v']
		v_a = (np.sum(v, axis=0) / len(v))
		v_a = v_a / np.max(np.abs(v_a))
		for _ in range(self.nmutt):
			g = task.repair(xb + v_a * self.uniform(task.Lower, task.Upper), self.Rand)
			fg = task.eval(g)
			if fg <= fxb: xb, fxb = g, fg
		return pop, fpop, xb, fxb, args, d


class MutatedCenterUnifiedParticleSwarmOptimization(MutatedCenterParticleSwarmOptimization):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Mutated Center Unified Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Tsai, Hsing-Chih. "Unified particle swarm delivers high efficiency to particle swarm optimization." Applied Soft Computing 55 (2017): 371-383.

	Attributes:
		nmutt (int): Number of mutations of global best particle.

	See Also:
		* :class:`NiaPy.algorithms.basic.CenterParticleSwarmOptimization`
	"""
	Name = ['MutatedCenterUnifiedParticleSwarmOptimization', 'MCUPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Tsai, Hsing-Chih. "Unified particle swarm delivers high efficiency to particle swarm optimization." Applied Soft Computing 55 (2017): 371-383."""

	def set_parameters(self, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.MutatedCenterParticleSwarmOptimization.setParameters`
		"""
		kwargs.pop('vMin', None), kwargs.pop('vMax', None)
		MutatedCenterParticleSwarmOptimization.set_parameters(self, vMin=-np.inf, vMax=np.inf, **kwargs)

	def update_velocity(self, v, p, pb, gb, w, min_velocity, max_velocity, task, **kwargs):
		r"""Update particle velocity.

		Args:
			v (numpy.ndarray): Current velocity of particle.
			p (numpy.ndarray): Current position of particle.
			pb (numpy.ndarray): Personal best position of particle.
			gb (numpy.ndarray): Global best position of particle.
			w (numpy.ndarray): Weights for velocity adjustment.
			min_velocity (numpy.ndarray): Minimal velocity allowed.
			max_velocity (numpy.ndarray): Maxmimal velocity allowed.
			task (Task): Optimization task.
			kwargs (dict): Additional arguments.

		Returns:
			numpy.ndarray: Updated velocity of particle.
		"""
		r3 = self.rand(task.D)
		return self.Repair(w * v + self.C1 * self.rand(task.D) * (pb - p) * r3 + self.C2 * self.rand(task.D) * (gb - p) * (1 - r3), min_velocity, max_velocity)


class ComprehensiveLearningParticleSwarmOptimizer(ParticleSwarmAlgorithm):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Comprehensive Learning Particle Swarm Optimizer

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		J. J. Liang, a. K. Qin, P. N. Suganthan and S. Baskar, "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, June 2006. doi: 10.1109/TEVC.2005.857610

	Reference URL:
		http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1637688&isnumber=34326

	Attributes:
		w0 (float): Inertia weight.
		w1 (float): Inertia weight.
		C (float): Velocity constant.
		m (int): Refresh rate.

	See Also:
		* :class:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm`
	"""
	Name = ['ComprehensiveLearningParticleSwarmOptimizer', 'CLPSO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""J. J. Liang, a. K. Qin, P. N. Suganthan and S. Baskar, "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, June 2006. doi: 10.1109/TEVC.2005.857610	"""

	def set_parameters(self, m=10, w0=.9, w1=.4, c=1.49445, **ukwargs):
		r"""Set Particle Swarm Algorithm main parameters.

		Args:
			w0 (int): Inertia weight.
			w1 (float): Inertia weight.
			c (float): Velocity constant.
			m (float): Refresh rate.
			ukwargs (dict): Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.setParameters`
		"""
		ParticleSwarmAlgorithm.set_parameters(self, **ukwargs)
		self.m, self.w0, self.w1, self.C = m, w0, w1, c

	def get_parameters(self):
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.get_parameters(self)
		d.update({
			'm': self.m,
			'w0': self.w0,
			'w1': self.w1,
			'c': self.C
		})
		return d

	def init(self, task):
		r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Dict[str, numpy.ndarray]:
				* vMin: Mininal velocity.
				* vMax: Maximal velocity.
				* V: Initial velocity of particle.
				* flag: Refresh gap counter.
		"""
		return {'min_velocity': full_array(self.vMin, task.D), 'max_velocity': full_array(self.vMax, task.D), 'v': np.full([self.NP, task.D], 0.0), 'flag': np.full(self.NP, 0), 'pc': np.asarray([.05 + .45 * (np.exp(10 * (i - 1) / (self.NP - 1)) - 1) / (np.exp(10) - 1) for i in range(self.NP)])}

	def generate_personal_best_cl(self, i, pc, pbs, fpbs):
		r"""Generate new personal best position for learning.

		Args:
			i (int): Current particle.
			pc (float): Learning probability.
			pbs (numpy.ndarray): Personal best positions for population.
			fpbs (numpy.ndarray): Personal best positions function/fitness values for persolan best position.

		Returns:
			numpy.ndarray: Personal best for learning.
		"""
		pbest = []
		for j in range(len(pbs[i])):
			if self.rand() > pc: pbest.append(pbs[i, j])
			else:
				r1, r2 = int(self.rand() * len(pbs)), int(self.rand() * len(pbs))
				if fpbs[r1] < fpbs[r2]: pbest.append(pbs[r1, j])
				else: pbest.append(pbs[r2, j])
		return np.asarray(pbest)

	def update_velocity_cl(self, v, p, pb, w, min_velocity, max_velocity, task, **kwargs):
		r"""Update particle velocity.

		Args:
			v (numpy.ndarray): Current velocity of particle.
			p (numpy.ndarray): Current position of particle.
			pb (numpy.ndarray): Personal best position of particle.
			w (numpy.ndarray): Weights for velocity adjustment.
			min_velocity (numpy.ndarray): Minimal velocity allowed.
			max_velocity (numpy.ndarray): Maxmimal velocity allowed.
			task (Task): Optimization task.
			kwargs (dict): Additional arguments.

		Returns:
			numpy.ndarray: Updated velocity of particle.
		"""
		return self.Repair(w * v + self.C * self.rand(task.D) * (pb - p), min_velocity, max_velocity)

	def run_iteration(self, task, pop, fpop, xb, fxb, popb, fpopb, min_velocity, max_velocity, v, flag, pc, *args, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current populations.
			fpop (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Current best particle.
			fxb (float): Current best particle fitness/function value.
			popb (numpy.ndarray): Particles best position.
			fpopb (numpy.ndarray): Particles best positions fitness/function values.
			min_velocity (numpy.ndarray): Minimal velocity.
			max_velocity (numpy.ndarray): Maximal velocity.
			v (numpy.ndarray): Velocity of particles.
			flag (numpy.ndarray): Refresh rate counter.
			pc (numpy.ndarray): Learning rate.
			args (list): Additional function arguments.
			dparams (dict): Additional function keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. New global best position.
				4. New global best positions function/fitness value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* popb: Particles best population.
					* fpopb: Particles best positions function/fitness value.
					* min_velocity: Minimal velocity.
					* max_velocity: Maximal velocity.
					* V: Initial velocity of particle.
					* flag: Refresh gap counter.
					* pc: Learning rate.

		See Also:
			* :class:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.runIteration`
		"""
		w = self.w0 * (self.w0 - self.w1) * task.Iters / task.nGEN
		for i in range(len(pop)):
			if flag[i] >= self.m:
				v[i] = self.update_velocity(v[i], pop[i], popb[i], xb, 1, min_velocity, max_velocity, task)
				pop[i] = task.repair(pop[i] + v[i], rnd=self.Rand)
				fpop[i] = task.eval(pop[i])
				if fpop[i] < fpopb[i]:
					popb[i], fpopb[i] = pop[i].copy(), fpop[i]
					if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
				flag[i] = 0
			pbest = self.generate_personal_best_cl(i, pc[i], popb, fpopb)
			v[i] = self.update_velocity_cl(v[i], pop[i], pbest, w, min_velocity, max_velocity, task)
			pop[i] = pop[i] + v[i]
			if not ((pop[i] < task.Lower).any() or (pop[i] > task.Upper).any()):
				fpop[i] = task.eval(pop[i])
				if fpop[i] < fpopb[i]:
					popb[i], fpopb[i] = pop[i].copy(), fpop[i]
					if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
		return pop, fpop, xb, fxb, args, {'popb': popb, 'fpopb': fpopb, 'min_velocity': min_velocity, 'max_velocity': max_velocity, 'v': v, 'flag': flag, 'pc': pc}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
