import numpy as np

from grid import grid
from agent import agent
g = grid([3,4])

g.set_hole([1,1])

g.set_goal([2,3])

g.init()

g.show()


a = agent(g, [0,1])






