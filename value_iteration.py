import numpy as np
from grid import grid
from agent import agent

g = grid([5,5])

g.set_unavailable([1,1]) # set 

g.set_terminate([4,4], 1)

g.set_terminate([0,4], -1)

g.init()

#g.show()

"""

tprob = 0.1 agent got a probability of 0.1 tranistion to non selected state

tprob = 1 - 0.1 * 3  = 0.7, agent got a probability of 0.7 transition to selected state (0.1 for other three action)

"""
a = agent(g, [0,0], discount = 0.9, reward = -0.1, tprob = 0.1)

a.random_walk(epoch = 10, show = True)

a.show_policy()




