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

        for e in xrange(epoch):
            
            for j in (xrange(self.env.size[0])):

                for i in (xrange(self.env.size[1])):
                    
                    if [j,i] in self.env.terminate or [j,i] in self.env.unavailable:

                        continue
            
                    self.state = [j,i]
                    
                    maximum = None
                    
                    for a_index in xrange(len(self.env.dir)):    

                        q = self.reward
                        print '-------------'
                        print 'Take {}'.format(self.env.dir[a_index])
                        print self.show()

                        for pns, pnd in zip(self.available()[0], self.available()[1]):    
                            if pns in self.env.unavailable:

                                continue
                                
                            q += self.discount * \
                                    self.tprob[j, i, a_index, self.action_index(pnd)] * \
                                    self.value_function(pns) 
                        
                        self.q_value[j, i, a_index] = q
                        
                        maximum = q if maximum == None else max(maximum, q)
                        

                    self.value_k1[j,i] = maximum
                    #if i == 2:

                     #   print self.q_value[j,i]

                      #  raise
            self.value = self.value_k1
        self.update_policy()            


"""

tprob = 0.1 agent got a probability of 0.1 tranistion to non selected state

tprob = 1 - 0.1 * 3  = 0.7, agent got a probability of 0.7 transition to selected state (0.1 for other three action)

"""
a = value_iteration_agent(g, [0,0], discount = 0.9, reward = -0.1, tprob = 0.1)

print a.q_value[1,1]
a.value_iteration(epoch = 40)

a.show_policy()
a.show()
print a.q_value[1,1]

print a.q_value[0,1]
print a.value_function([0,1], True)

print a.value_function([1,0])
print a.value_function([1,0], index = True)

print a.q_value[1,0]
