import numpy as np

from grid import grid
from agent import agent
g = grid([10,10])

g.set_unavailable([1,1])

g.set_terminate([9,9], 1)
g.set_terminate([9,0], -1)
g.set_terminate([0,9], 1)
g.init()

g.show()


a = agent(g, [0,5], 10, 1, tprob = [0.7,0.1,0.1,0.1])

a.random_walk(epoch = 100, show = True)

a.show_policy()




