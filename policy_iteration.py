import numpy as np
from agent import agent
from grid import grid

class policy_iteration_agent(agent):

    def __init__(self, env, start, discount, reward, tprob):

        agent.__init__(self, env, start, discount)
    
    def policy_iteration(self, epoch):
        
        self.show()

        for i in xrange(epoch):
            
            self.state = self.random_start()

            self.step = 0

            if self.state in self.env.terminate or self.state in self.env.unavailable:
                continue

            while not self.isBreak:

                self.act(self.policy[self.state[0], self.state[1]])
                
                self.step += 1
                    
                self.update_policy()

            self.isBreak = False


g = grid([5,5])

g.set_unavailable([1,1]) # set 

g.set_terminate([4,4], 1)

g.set_terminate([0,4], -1)

g.init()

a = policy_iteration_agent(g, [0,0], discount = 0.9, reward = -1 tprob = 0.1)


a.policy_iteration(100)

a.show_policy()


