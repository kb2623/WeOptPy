# encoding=utf8

from WeOptPy.algorithms import (
	ParticleSwarmOptimization,
	ParticleSwarmAlgorithm,
	OppositionVelocityClampingParticleSwarmOptimization,
	CenterParticleSwarmOptimization,
	MutatedParticleSwarmOptimization,
	MutatedCenterParticleSwarmOptimization,
	ComprehensiveLearningParticleSwarmOptimizer,
	MutatedCenterUnifiedParticleSwarmOptimization
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	MyBenchmark
)


class PSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))

	def test_custom_works_fine(self):
		pso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, pso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		pso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		pso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, pso_custom, pso_customc, MyBenchmark())


class PSATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ParticleSwarmAlgorithm

	def test_algorithm_info(self):
		al = ParticleSwarmAlgorithm.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = ParticleSwarmAlgorithm.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		wvcpso_custom = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wvcpso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		wvcpso_custom = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_customc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, wvcpso_custom, wvcpso_customc, MyBenchmark())


class OVCPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = OppositionVelocityClampingParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		wvcpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wvcpso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		wvcpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, wvcpso_custom, wvcpso_customc, MyBenchmark())


class CPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CenterParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		cpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cpso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		cpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		cpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, cpso_custom, cpso_customc, MyBenchmark())


class MPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MutatedParticleSwarmOptimization

	def test_algorithm_info(self):
		al = MutatedParticleSwarmOptimization.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = MutatedParticleSwarmOptimization.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mpso_custom = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mpso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		mpso_custom = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mpso_customc = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mpso_custom, mpso_customc, MyBenchmark())


class MCPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MutatedCenterParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mcpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mcpso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		mcpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mcpso_custom, mcpso_customc, MyBenchmark())


class MCUPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MutatedCenterUnifiedParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mcupso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mcupso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		mcupso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcupso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, mcupso_custom, mcupso_customc, MyBenchmark())


class CLPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ComprehensiveLearningParticleSwarmOptimizer

	def test_algorithm_info(self):
		al = self.algo.algorithm_info()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.type_parameters()
		self.assertTrue(d['c1'](10))
		self.assertTrue(d['c2'](10))
		self.assertTrue(d['c1'](0))
		self.assertTrue(d['c2'](0))
		self.assertFalse(d['c1'](-10))
		self.assertFalse(d['c2'](-10))
		self.assertTrue(d['max_velocity'](10))
		self.assertTrue(d['min_velocity'](10))
		self.assertTrue(d['n'](10))
		self.assertFalse(d['n'](-10))
		self.assertFalse(d['n'](0))

	def test_custom_works_fine(self):
		clpso_custom = self.algo(n=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, clpso_custom, MyBenchmark())

	def test_custom_works_fine_parallel(self):
		clpso_custom = self.algo(n=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		clpso_customc = self.algo(n=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run_parallel(self, clpso_custom, clpso_customc, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
