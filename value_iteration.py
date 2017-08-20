import numpy as np

from grid import grid
from agent import agent
g = grid([3,4])

g.set_unavailable([1,1])

g.set_terminate([2,3], 100)
g.set_terminate([1,3], -100)
g.init()

g.show()


a = agent(g, [0,1], 10, 1, tprob = [0.25, 0.25, 0.5, 0.])

a.random_walk(epoch = 100, show = True)

a.show_policy()




