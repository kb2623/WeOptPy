# encoding=utf8

from WeOptPy.algorithms import ForestOptimizationAlgorithm
from WeOptPy.tests.test_algorithm import (
    AlgorithmTestCase,
    MyBenchmark
)


class FOATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ForestOptimizationAlgorithm

    def test_type_parameters(self):
        tp = self.algo.type_parameters()
        self.assertTrue(tp['n'](1))
        self.assertFalse(tp['n'](0))
        self.assertFalse(tp['n'](-1))
        self.assertFalse(tp['n'](1.0))
        self.assertTrue(tp['lt'](1))
        self.assertFalse(tp['lt'](0))
        self.assertFalse(tp['lt'](-1))
        self.assertFalse(tp['lt'](1.0))
        self.assertTrue(tp['al'](1))
        self.assertFalse(tp['al'](0))
        self.assertFalse(tp['al'](-1))
        self.assertFalse(tp['al'](1.0))
        self.assertTrue(tp['lsc'](1))
        self.assertFalse(tp['lsc'](0))
        self.assertFalse(tp['lsc'](-1))
        self.assertFalse(tp['lsc'](1.0))
        self.assertTrue(tp['gsc'](1))
        self.assertFalse(tp['gsc'](0))
        self.assertFalse(tp['gsc'](-1))
        self.assertFalse(tp['gsc'](1.0))
        self.assertTrue(tp['tr'](1.0))
        self.assertTrue(tp['tr'](0.5))
        self.assertTrue(tp['tr'](0.0))
        self.assertFalse(tp['tr'](-1))
        self.assertFalse(tp['tr'](1.1))

    def test_works_fine(self):
        foa = self.algo(NP=20, lt=5, lsc=1, gsc=1, al=20, tr=0.35, seed=self.seed)
        foac = self.algo(NP=20, lt=5, lsc=1, gsc=1, al=20, tr=0.35, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, foa, foac, MyBenchmark())


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
