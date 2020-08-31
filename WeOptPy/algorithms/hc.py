# encoding=utf8

from numpy import random as rand

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = ['HillClimbAlgorithm']


def neighborhood(x, delta, task, rnd=rand):
    r"""Get neighbours of point.

    Args:
        x numpy.ndarray: Point.
        delta (float): Standard deviation.
        task (Task): Optimization task.
        rnd (Optional[mtrand.RandomState]): Random generator.

    Returns:
        Tuple[numpy.ndarray, float]:
            1. New solution.
            2. New solutions function/fitness value.
    """
    X = x + rnd.normal(0, delta, task.D)
    X = task.repair(X, rnd)
    Xfit = task.eval(X)
    return X, Xfit


class HillClimbAlgorithm(Algorithm):
    r"""Implementation of iterative hill climbing algorithm.

    Algorithm:
        Hill Climbing Algorithm

    Date:
        2018

    Authors:
        Jan Popič

    License:
        MIT

    Reference URL:

    Reference paper:

    See Also:
        * :class:`WeOptPy.algorithms.interfaces.Algorithm`

    Attributes:
        delta (float): Change for searching in neighborhood.
        Neighborhood (Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]): Function for getting neighbours.
    """
    Name = ['HillClimbAlgorithm', 'BBFA']

    @staticmethod
    def algorithmInfo():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information.
        """
        return r"""TODO"""

    @staticmethod
    def type_parameters():
        r"""TODO.

        Returns:
            Dict[str, Callable]:
                * delta (Callable[[Union[int, float]], bool]): TODO
        """
        return {'delta': lambda x: isinstance(x, (int, float)) and x > 0}

    def set_parameters(self, delta=0.5, Neighborhood=neighborhood, **ukwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            * delta (Optional[float]): Change for searching in neighborhood.
            * Neighborhood (Optional[Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]]): Function for getting neighbours.
        """
        Algorithm.set_parameters(self, n=1, **ukwargs)
        self.delta, self.Neighborhood = delta, Neighborhood

    def get_parameters(self):
        d = Algorithm.get_parameters(self)
        d.update({
            'delta': self.delta,
            'Neighborhood': self.Neighborhood
        })
        return d

    def init_population(self, task):
        r"""Initialize stating point.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, list, dict]:
                1. New individual.
                2. New individual function/fitness value.
                3. Additional arguments.
                4. Additional keyword arguments.
        """
        x = task.Lower + self.rand(task.D) * task.bRange
        return x, task.eval(x), [], {}

    def run_iteration(self, task, x, fx, xb, fxb, *args, **dparams):
        r"""Core function of HillClimbAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            x (numpy.ndarray): Current solution.
            fx (float): Current solutions fitness/function value.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best solutions function/fitness value.
            args (list): Additional arguments.
            dparams (dict): Additional keyword arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
                1. New solution.
                2. New solutions function/fitness value.
                3. Global best solution.
                4. Global best solution fitness.
                5. Additional arguments.
                6. Additional keyword arguments.
        """
        lo, xn = False, task.Lower + task.bRange * self.rand(task.D)
        xn_f = task.eval(xn)
        while not lo:
            yn, yn_f = self.Neighborhood(x, self.delta, task, rnd=self.Rand)
            if yn_f < xn_f: xn, xn_f = yn, yn_f
            else: lo = True or task.stop_cond()
        xb, fxb = self.get_best(xn, xn_f, xb, fxb)
        return xn, xn_f, xb, fxb, args, {}
