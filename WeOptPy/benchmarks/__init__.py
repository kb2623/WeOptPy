# encoding=utf8

"""Module with implementations of benchmark functions."""

from WeOptPy.benchmarks import interfaces
from WeOptPy.benchmarks.ackley import Ackley
from WeOptPy.benchmarks.alpine import (
	Alpine1,
	Alpine2
)
from WeOptPy.benchmarks.autocorelation import (
	AutoCorrelation,
	AutoCorrelationEnergy
)
from WeOptPy.benchmarks.bentcigar import BentCigar
from WeOptPy.benchmarks.bohachevsky import Bohachevsky
from WeOptPy.benchmarks.chungReynolds import ChungReynolds
from WeOptPy.benchmarks.clustering import (
	Clustering,
	ClusteringMin,
	ClusteringMinPenalty,
	ClusteringClassification
)
from WeOptPy.benchmarks.cosinemixture import CosineMixture
from WeOptPy.benchmarks.csendes import Csendes
from WeOptPy.benchmarks.deb import (
	Deb01,
	Deb02
)
from WeOptPy.benchmarks.deflectedcorrugatespring import DeflectedCorrugatedSpring
from WeOptPy.benchmarks.discus import Discus
from WeOptPy.benchmarks.dixonprice import DixonPrice
from WeOptPy.benchmarks.easom import Easom
from WeOptPy.benchmarks.elliptic import Elliptic
from WeOptPy.benchmarks.exponential import Exponential
from WeOptPy.benchmarks.griewank import (
	Griewank,
	ExpandedGriewankPlusRosenbrock
)
from WeOptPy.benchmarks.happyCat import HappyCat
from WeOptPy.benchmarks.hgbat import HGBat
from WeOptPy.benchmarks.hilbert import Hilbert
from WeOptPy.benchmarks.infinity import Infinity
from WeOptPy.benchmarks.katsuura import Katsuura
from WeOptPy.benchmarks.lennardjones import LennardJones
from WeOptPy.benchmarks.levy import Levy
from WeOptPy.benchmarks.michalewichz import Michalewichz
from WeOptPy.benchmarks.needleeye import NeedleEye
from WeOptPy.benchmarks.perm import Perm
from WeOptPy.benchmarks.pinter import Pinter
from WeOptPy.benchmarks.powell import Powell
from WeOptPy.benchmarks.qing import Qing
from WeOptPy.benchmarks.quintic import Quintic
from WeOptPy.benchmarks.rastrigin import Rastrigin
from WeOptPy.benchmarks.ridge import Ridge
from WeOptPy.benchmarks.rosenbrock import Rosenbrock
from WeOptPy.benchmarks.salomon import Salomon
from WeOptPy.benchmarks.schaffer import (
	SchafferN2,
	SchafferN4,
	ExpandedSchafferF6
)
from WeOptPy.benchmarks.schumerSteiglitz import SchumerSteiglitz
from WeOptPy.benchmarks.schwefel import (
	Schwefel,
	Schwefel221,
	Schwefel222,
	ModifiedSchwefel
)
from WeOptPy.benchmarks.sphere import (
	Sphere,
	Sphere2,
	Sphere3
)
from WeOptPy.benchmarks.step import (
	Step,
	Step2,
	Step3
)
from WeOptPy.benchmarks.styblinskiTang import StyblinskiTang
from WeOptPy.benchmarks.tchebyshev import Tchebychev
from WeOptPy.benchmarks.trid import Trid
from WeOptPy.benchmarks.weierstrass import Weierstrass
from WeOptPy.benchmarks.whitley import Whitley
from WeOptPy.benchmarks.xinsheyang import (
	XinSheYang01,
	XinSheYang02,
	XinSheYang03,
	XinSheYang04
)
from WeOptPy.benchmarks.yaoliu import YaoLiu09
from WeOptPy.benchmarks.zakharov import Zakharov

__all__ = [
	'interfaces',
	'Ackley',
	'Alpine1',
	'Alpine2',
	'AutoCorrelation',
	'AutoCorrelationEnergy',
	'BentCigar',
	'Bohachevsky',
	'ChungReynolds',
	'Clustering',
	'ClusteringMin',
	'ClusteringMinPenalty',
	'ClusteringClassification',
	'CosineMixture',
	'Csendes',
	'Deb01',
	'Deb02',
	'DeflectedCorrugatedSpring',
	'Discus',
	'DixonPrice',
	'Easom',
	'Elliptic',
	'ExpandedGriewankPlusRosenbrock',
	'ExpandedSchafferF6',
	'Exponential',
	'Griewank',
	'HappyCat',
	'HGBat',
	'Hilbert',
	'Infinity',
	'Katsuura',
	'LennardJones',
	'Levy',
	'Michalewichz',
	'ModifiedSchwefel',
	'NeedleEye',
	'Perm',
	'Pinter',
	'Powell',
	'Qing',
	'Quintic',
	'Rastrigin',
	'Ridge',
	'Rosenbrock',
	'Salomon',
	'SchafferN2',
	'SchafferN4',
	'SchumerSteiglitz',
	'Schwefel',
	'Schwefel221',
	'Schwefel222',
	'Sphere',
	'Sphere2',
	'Sphere3',
	'Step',
	'Step2',
	'Step3',
	'StyblinskiTang',
	'Tchebychev',
	'Trid',
	'Weierstrass',
	'Whitley',
	'XinSheYang01',
	'XinSheYang02',
	'XinSheYang03',
	'XinSheYang04',
	'YaoLiu09',
	'Zakharov'
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
