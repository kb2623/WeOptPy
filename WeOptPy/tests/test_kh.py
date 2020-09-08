# encoding=utf8

"""Krill herd algorithm test case module."""

from WeOptPy.algorithms import (
	KrillHerdV1,
	KrillHerdV2,
	KrillHerdV3,
	KrillHerdV4,
	KrillHerdV11
)
from WeOptPy.tests.test_algorithm import (
	AlgorithmTestCase,
	Sphere
)


class KHV1TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV1 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = KrillHerdV1

	def test_type_parametes(self):
		d = self.algo.type_parameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNotNone(d.get('Cr', None))
		self.assertIsNotNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, Sphere())


class KHV2TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV2 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = KrillHerdV2

	def test_type_parametes(self):
		d = self.algo.type_parameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNotNone(d.get('Cr', None))
		self.assertIsNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, Sphere())


class KHV3TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV3 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = KrillHerdV3

	def test_type_parametes(self):
		d = self.algo.type_parameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNone(d.get('Cr', None))
		self.assertIsNotNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, Sphere())


class KHV4TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV4 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = KrillHerdV4

	def test_type_parametes(self):
		d = self.algo.type_parameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNone(d.get('Cr', None))
		self.assertIsNone(d.get('Mu', None))
		self.assertIsNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, Sphere())


class KHV11TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV11 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = KrillHerdV11

	def test_type_parametes(self):
		d = self.algo.type_parameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNotNone(d.get('Cr', None))
		self.assertIsNotNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, Sphere())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
