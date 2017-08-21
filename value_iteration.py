import numpy as np
from grid import grid
from agent import agent

g = grid([5,5])

g.set_unavailable([1,1]) # set 

g.set_terminate([4,4], 1)

g.set_terminate([0,4], -1)

g.init()

class value_iteration_agent(agent):

    def __init__(self, env, start, discount, reward, tprob):

        agent.__init__(self, env, start, discount)

    def value_iteration(self, epoch):

        for i in xrange(epoch):

            self.state = self.random_start()

            self.step = 0

            if self.state in self.env.terminate or self.state in self.env.unavailable:

                continue

            while not self.isBreak:
                
                for j in self.env.size[0]:

                    for i in self.env.size[1]:

                        for a in self.env.dir:
                            
                            maximum = None

                            sum_over_next_state = 0

                            for pns, pnd in zip(xrange(len(self.env.dir)):

                                sum_over_next_state += self.discount * self.tprob[j, i, a, pns_index] * self.value_function([j,i]) 
                                    
                                else:



                #self.act(self.policy[self.state[0], self.state[1])

                self.step += 1




"""

tprob = 0.1 agent got a probability of 0.1 tranistion to non selected state

tprob = 1 - 0.1 * 3  = 0.7, agent got a probability of 0.7 transition to selected state (0.1 for other three action)

"""
a = agent(g, [0,0], discount = 0.9, reward = 0, tprob = 0.1)

a.value_iteration(epoch = 50, show = True)

a.show_policy()




