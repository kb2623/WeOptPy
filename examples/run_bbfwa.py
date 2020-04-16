# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm
from NiaPy.task.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run Fireworks Algorithm for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=1000, benchmark=Sphere())
	algo = BareBonesFireworksAlgorithm()
	best = algo.run(task)
	print('%s -> %s' % (best[0], best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
