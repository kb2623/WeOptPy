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
from WeOptPy.algorithms.es import (
	EvolutionStrategy1p1,
	EvolutionStrategyMp1,
	EvolutionStrategyML,
	EvolutionStrategyMpL,
	CovarianceMatrixAdaptionEvolutionStrategy
)
from WeOptPy.algorithms.fa import FireflyAlgorithm
from WeOptPy.algorithms.foa import ForestOptimizationAlgorithm
from WeOptPy.algorithms.fpa import FlowerPollinationAlgorithm
from WeOptPy.algorithms.fss import FishSchoolSearch
from WeOptPy.algorithms.fwa import (
	BareBonesFireworksAlgorithm,
	DynamicFireworksAlgorithm,
	DynamicFireworksAlgorithmGauss,
	EnhancedFireworksAlgorithm,
	FireworksAlgorithm
)
from WeOptPy.algorithms.ga import GeneticAlgorithm
from WeOptPy.algorithms.gso import (
	GlowwormSwarmOptimization,
	GlowwormSwarmOptimizationV1,
	GlowwormSwarmOptimizationV2,
	GlowwormSwarmOptimizationV3
)
from WeOptPy.algorithms.gsa import GravitationalSearchAlgorithm
from WeOptPy.algorithms.gwo import GreyWolfOptimizer
from WeOptPy.algorithms.hba import HybridBatAlgorithm
from WeOptPy.algorithms.hc import HillClimbAlgorithm
from WeOptPy.algorithms.hde import (
	DifferentialEvolutionMTS,
	DifferentialEvolutionMTSv1,
	DynNpDifferentialEvolutionMTS,
	DynNpDifferentialEvolutionMTSv1,
	MultiStrategyDifferentialEvolutionMTS,
	MultiStrategyDifferentialEvolutionMTSv1,
	DynNpMultiStrategyDifferentialEvolutionMTS,
	DynNpMultiStrategyDifferentialEvolutionMTSv1
)
from WeOptPy.algorithms.hs import (
	HarmonySearch,
	HarmonySearchV1
)
from WeOptPy.algorithms.hsaba import HybridSelfAdaptiveBatAlgorithm
from WeOptPy.algorithms.jade import AdaptiveArchiveDifferentialEvolution
from WeOptPy.algorithms.jde import (
	SelfAdaptiveDifferentialEvolution,
	MultiStrategySelfAdaptiveDifferentialEvolution,
	DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
	DynNpMultiStrategySelfAdaptiveDifferentialEvolution
)
from WeOptPy.algorithms.kh import (
	KrillHerdV1,
	KrillHerdV2,
	KrillHerdV3,
	KrillHerdV4,
	KrillHerdV11
)
from WeOptPy.algorithms.mbo import MonarchButterflyOptimization
from WeOptPy.algorithms.mfo import MothFlameOptimizer
from WeOptPy.algorithms.mke import (
	MonkeyKingEvolutionV1,
	MonkeyKingEvolutionV2,
	MonkeyKingEvolutionV3
)
from WeOptPy.algorithms.mts import (
	MultipleTrajectorySearch,
	MultipleTrajectorySearchV1
)
from WeOptPy.algorithms.nmm import NelderMeadMethod
from WeOptPy.algorithms.pso import (
	ParticleSwarmAlgorithm,
	ParticleSwarmOptimization,
	CenterParticleSwarmOptimization,
	ComprehensiveLearningParticleSwarmOptimizer,
	MutatedCenterParticleSwarmOptimization,
	MutatedCenterUnifiedParticleSwarmOptimization,
	MutatedParticleSwarmOptimization,
	OppositionVelocityClampingParticleSwarmOptimization
)
from WeOptPy.algorithms.sa import SimulatedAnnealing
from WeOptPy.algorithms.saba import (
	AdaptiveBatAlgorithm,
	SelfAdaptiveBatAlgorithm
)
from WeOptPy.algorithms.sade import (
	StrategyAdaptationDifferentialEvolution,
	StrategyAdaptationDifferentialEvolutionV1
)
from WeOptPy.algorithms.sca import SineCosineAlgorithm
from WeOptPy.algorithms.ts import TabuSearch

__all__ = [
	'interfaces',
	'ArtificialBeeColonyAlgorithm',
	'AnarchicSocietyOptimization',
	'BatAlgorithm',
	'BeesAlgorithm',
	'CamelAlgorithm',
	'CoralReefsOptimization',
	'CuckooSearch',
	'CatSwarmOptimization',
	'DifferentialEvolution',
	'AgingNpDifferentialEvolution',
	'AgingNpMultiMutationDifferentialEvolution',
	'DynNpDifferentialEvolution',
	'DynNpMultiStrategyDifferentialEvolution',
	'MultiStrategyDifferentialEvolution',
	'CrowdingDifferentialEvolution',
	'EvolutionStrategy1p1',
	'EvolutionStrategyMp1',
	'EvolutionStrategyML',
	'EvolutionStrategyMpL',
	'CovarianceMatrixAdaptionEvolutionStrategy',
	'FireflyAlgorithm',
	'ForestOptimizationAlgorithm',
	'FlowerPollinationAlgorithm',
	'FishSchoolSearch',
	'BareBonesFireworksAlgorithm',
	'DynamicFireworksAlgorithm',
	'DynamicFireworksAlgorithmGauss',
	'EnhancedFireworksAlgorithm',
	'FireworksAlgorithm',
	'GeneticAlgorithm',
	'GlowwormSwarmOptimization',
	'GlowwormSwarmOptimizationV1',
	'GlowwormSwarmOptimizationV2',
	'GlowwormSwarmOptimizationV3',
	'GravitationalSearchAlgorithm',
	'GreyWolfOptimizer',
	'HybridBatAlgorithm',
	'HillClimbAlgorithm',
	'HarmonySearch',
	'HarmonySearchV1',
	'HybridSelfAdaptiveBatAlgorithm',
	'AdaptiveArchiveDifferentialEvolution',
	'SelfAdaptiveDifferentialEvolution',
	'MultiStrategySelfAdaptiveDifferentialEvolution',
	'DynNpSelfAdaptiveDifferentialEvolutionAlgorithm',
	'DynNpMultiStrategySelfAdaptiveDifferentialEvolution',
	'KrillHerdV1',
	'KrillHerdV2',
	'KrillHerdV3',
	'KrillHerdV4',
	'KrillHerdV11',
	'MonarchButterflyOptimization',
	'MothFlameOptimizer',
	'MonkeyKingEvolutionV1',
	'MonkeyKingEvolutionV2',
	'MonkeyKingEvolutionV3',
	'MultipleTrajectorySearch',
	'MultipleTrajectorySearchV1',
	'NelderMeadMethod',
	'ParticleSwarmAlgorithm',
	'ParticleSwarmOptimization',
	'CenterParticleSwarmOptimization',
	'ComprehensiveLearningParticleSwarmOptimizer',
	'MutatedCenterParticleSwarmOptimization',
	'MutatedCenterUnifiedParticleSwarmOptimization',
	'MutatedParticleSwarmOptimization',
	'OppositionVelocityClampingParticleSwarmOptimization',
	'SimulatedAnnealing',
	'AdaptiveBatAlgorithm',
	'SelfAdaptiveBatAlgorithm',
	'StrategyAdaptationDifferentialEvolution',
	'StrategyAdaptationDifferentialEvolutionV1',
	'SineCosineAlgorithm',
	'TabuSearch',
	'DifferentialEvolutionMTS',
	'DifferentialEvolutionMTSv1',
	'DynNpDifferentialEvolutionMTS',
	'DynNpDifferentialEvolutionMTSv1',
	'MultiStrategyDifferentialEvolutionMTS',
	'MultiStrategyDifferentialEvolutionMTSv1',
	'DynNpMultiStrategyDifferentialEvolutionMTS',
	'DynNpMultiStrategyDifferentialEvolutionMTSv1'
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
