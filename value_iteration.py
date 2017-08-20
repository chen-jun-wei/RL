import numpy as np

from grid import grid
from agent import agent
g = grid([3,4])

g.set_unavailable([1,1])

g.set_terminate([2,3], 1)
g.set_terminate([1,3], -1)
g.init()

g.show()


a = agent(g, [0,5], 10, 1, tprob = [0.7,0.1,0.1,0.1])

a.random_walk(epoch = 1, show = True)

a.show_policy()




