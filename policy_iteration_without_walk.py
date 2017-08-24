import numpy as np
from agent import agent
from grid import grid

class policy_iteration_agent(agent):

    def __init__(self, env, start, discount, reward, tprob):

        agent.__init__(self, env, start, discount)
    
    def policy_iteration(self, epoch):
        
        self.show()

        for e in xrange(epoch):

            for j in xrange(self.env.size[0]):

                for i in xrange(self.env.size[1]):

                    if [j,i] in self.env.terminate or [j,i] in self.env.unavailable:
                        continue
                    
                    self.state = [j,i]

                    maximum = None

                    for a_index in xrange(len(self.env.dir)):

                        q = self.reward

                        for pns, pnd in zip(self.available()[0], self.available()[1]):
                            
                            if pns in self.env.unavailable:

                                continue

                            q += self.discount * self.tprob[j, i, a_index, self.action_index(pnd)] * self.value_function(pns)

                        self.q_value[j, i, a_index] = q

                        maximum = q if maximum == None else max(maximum, q)                         

                self.value_k1[j, i] = maximum
                    
                self.value = self.value_k1

                self.update_policy()


g = grid([5,5])

g.set_unavailable([1,1]) # set 

g.set_terminate([4,4], 1)

g.set_terminate([0,4], -1)

g.init()

a = policy_iteration_agent(g, [0,0], discount = 0.9, reward = -1, tprob = 0.1)


a.policy_iteration(100)

a.show_policy()


