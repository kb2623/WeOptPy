# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from WeOptPy.algorithms import interfaces
from WeOptPy.algorithms.abc import ArtificialBeeColonyAlgorithm
from WeOptPy.algorithms.aso import AnarchicSocietyOptimization
from WeOptPy.algorithms.ba import BatAlgorithm
from WeOptPy.algorithms.bea import BeesAlgorithm
from WeOptPy.algorithms.ca import CamelAlgorithm
from WeOptPy.algorithms.cro import CoralReefsOptimization
from WeOptPy.algorithms.cs import CuckooSearch
from WeOptPy.algorithms.cso import CatSwarmOptimization
from WeOptPy.algorithms.de import (
	DifferentialEvolution,
	AgingNpDifferentialEvolution,
	AgingNpMultiMutationDifferentialEvolution,
	DynNpDifferentialEvolution,
	DynNpMultiStrategyDifferentialEvolution,
	MultiStrategyDifferentialEvolution,
	CrowdingDifferentialEvolution
)

__all__ = [
	'interfaces',
	'ArtificialBeeColonyAlgorithm',
	'AnarchicSocietyOptimization',
	''
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
