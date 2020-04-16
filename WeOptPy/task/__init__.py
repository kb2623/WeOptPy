"""Module with implementations of tasks."""

from WeOptPy.task import interfaces
from WeOptPy.task.countingtask import CountingTask
from WeOptPy.task.stoppingtask import StoppingTask
from WeOptPy.task.throwingtask import ThrowingTask
from WeOptPy.task.optimizationtype import OptimizationType

__all__ = [
	'interfaces',
	'CountingTask',
	'StoppingTask',
	'ThrowingTask',
	'OptimizationType'
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
