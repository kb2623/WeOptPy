# encoding=utf8

"""The implementation of tasks."""

from WeOptPy.task.stoppingtask import StoppingTask
from WeOptPy.util.exception import (
	FesException,
	GenException,
	RefException
)


class ThrowingTask(StoppingTask):
	r"""Task that throw exceptions when stopping condition is meet.

	See Also:
		* :class:`WeOptPy.util.StoppingTask`
	"""

	def __init__(self, **kwargs):
		r"""Initialize optimization task.

		Args:
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`WeOptPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)

	def stop_cond_e(self):
		r"""Throw exception for the given stopping condition.

		Raises:
			* FesException: Thrown when the number of function/fitness evaluations is reached.
			* GenException: Thrown when the number of algorithms generations/iterations is reached.
			* RefException: Thrown when the reference values is reached.
			* TimeException: Thrown when algorithm exceeds time run limit.
		"""
		# dtime = datetime.now() - self.startTime
		if self.Evals >= self.nFES:
			raise FesException()
		if self.Iters >= self.nGEN:
			raise GenException()
		# if self.runTime is not None and self.runTime >= dtime: raise TimeException()
		if self.refValue >= self.x_f:
			raise RefException()

	def eval(self, a):
		r"""Evaluate solution.

		Args:
			a (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Function/fitness values of solution.

		See Also:
			* :func:`WeOptPy.util.ThrowingTask.stopCondE`
			* :func:`WeOptPy.util.StoppingTask.eval`
		"""
		self.stop_cond_e()
		return StoppingTask.eval(self, a)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
