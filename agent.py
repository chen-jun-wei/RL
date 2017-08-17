from __future__ import print_function
import numpy as np
from utils import softmax, cdf
import copy

class agent:

    def __init__(self, env, start, epoch, discount):

        self.state = start
        self.discount = discount         
        self.env = env
        self.epoch = epoch
        self.step = 0
        self.isBreak = False
        # transition probability
        #
        # h x w         x a            x s'
        # 
        # [state] [choosed action] [reached state]
        #
        # in this example, choosed action is same as next state
        
        self.tprob = np.ones([env.size[0], env.size[1], len(env.dir), len(env.dir)]) / len(env.dir)
        
        
        # randomize the policy

        self.policy = np.random.randint(0, len(self.env.dir), (env.size[0], env.size[1]))
        
        self.q_value = np.zeros([env.size[0], env.size[1], len(env.dir)])
        
        self.value = np.zeros([env.size[0], env.size[1]])
        if not int(np.prod(np.array(env.size) > np.array(start))):        
        
            print ('[!] Start Position is not in Environment')


        self.initialize()
    
    def initialize(self):

        for un in self.env.unavailable:

            self.q_value[un[0],un[1]] = self.env.grid[un[0], un[1]]
            self.value[un[0], un[1]] = self.env.grid[un[0], un[1]]
        for tm in self.env.terminate:

            self.q_value[tm[0], tm[1]] = self.env.grid[tm[0], tm[1]]
            self.value[tm[0], tm[1]] = self.env.grid[tm[0], tm[1]]

    def available(self):
        
        available_state = []
        
        available_move = []
        
        for d in self.env.dir:

            temp = self.move(self.state, d)
            
            # pass if 'move' is attempt to move across the border
            
            if (temp[0] >= 0 and temp[0] < self.env.size[0]) and \
                    (temp[1] >= 0 and temp[1] < self.env.size[1]):
                
                _ = self.env.grid[temp[0], temp[1]]

                #if not np.isnan(_):
                
                available_state.append(temp)
                    
                available_move.append(d)

        return available_state, available_move # availabel state, not available move


    def move(self, state, direction):
        
        next_state = copy.copy(state)
        
        if direction == 'W':

            next_state[1] += -1

        elif direction == 'E':

            next_state[1] += 1

        elif direction == 'N':

            next_state[0] += 1

        elif direction == 'S':

            next_state[0] += -1

        else:

            print ('No Direction')

            raise
        
        return next_state

    
    def action_index(self, action):

        """
        
        action : a string specify which direction to go

        return : the index of direction

        """

        return self.env.dir.index(action)
    
    def random_walk(self, epoch, show = False):

        """
        
        Learning Process

        """

        
        
        for i in xrange(epoch):

            self.state = self.random_start()
            
            self.step = 0

            print (self.state)
                
            #print self.isBreak
            
            #print i

            if self.state in self.env.terminate or self.state in self.env.unavailable:

                continue
            
            while not self.isBreak:
                
                print ('-------------------------------------------------------')
                
                print ('Epoch : {}, Step : {}, State : {}'.format(i+1, self.step, self.state))
                
                print ('Choosed : ', self.env.dir[self.policy[self.state[0], self.state[1]]])
                r = self.act(self.policy[self.state[0], self.state[1]])

                self.step += 1
                
                if show:

                    self.show()

            self.isBreak = False

    def random_start(self):

        """
        
        randomly select a start position for each epoch

        
        """
        
        while True:

            coord = [np.random.choice(self.env.size[0]), \
                    np.random.choice(self.env.size[1])]
                    
            if coord in self.env.unavailable or coord in self.env.terminate:

                continue

            break

        return coord # set start position

    def isReachable(self, next_state):

        return False if np.isnan(self.env.grid[next_state[0], next_state[1]]) else True
                
    def act(self, action):

        """

        take the action and reach next state,
        but the transition is stochastic,
        which might not go the state one expected
        
        action : int -> action
        """
        
        # random select action
        # action = self.available() 

        # h x w x a x s'
    
        # stochastic transition probability
        
        tprob_available = []

        for ava in self.available()[1]:
        
            tprob_available.append(self.tprob[self.state[0], self.state[1], \
                    action, self.action_index(ava)])

        tprob_available = np.array(tprob_available)

        tprob_available = softmax(tprob_available, axis = 0)
        
        # available state [index]
        
        # reach next state sucessful
    
        #print 'available : ', self.available()[1]
        
        next_state = self.move(self.state, self.available()[1][self.transition(tprob_available)])
        
        print ('Next State : ', self.env.grid[next_state[0], next_state[1]])

        self.state = next_state if self.isReachable(next_state) else self.state
         
        print ('Reachable : ', self.isReachable(next_state))

        # update q value

        # pns : probable next state
        
        if self.state in self.env.terminate:

            # reset reward

            self.reward = 0
                
            self.isBreak = True

        elif self.isReachable(next_state) :      
            
            for a in xrange(len(self.env.dir)):
                #print ('possible next state value : ', self.value[pns[0],pns[1]])
                
                maximum = 0     

                for pns, pnd in zip(self.available()[0], self.available()[1]):
                    
                    if pns in self.env.unavailable:

                        continue
                    
                    #self.q_value[self.state[0], self.state[1], a] = self.tprob[self.state[0], self.state[1], a, self.action_index(pnd)] * \
                     #       (self.reward_function() + self.discount * self.value[pns[0], pns[1]])
                    maximum = max(maximum, self.tprob[self.state[0], self.state[1], a, self.action_index(pnd)] * \
                            (self.reward_function() + self.discount * self.value[pns[0], pns[1]]))
               
                # update value
                self.q_value[self.state[0], self.state[1], a] = maximum

                self.value[self.state[0], self.state[1]] = self.value_function()

                if self.value[pns[0],pns[1]] > 50:
                    
                    self.show() 
                    #raise
    # transition probability range, random probability

    def transition(self, tprob):

        """
        
        transition function
        
            randomly sample a number from uniform distribution


        """

        tprob_range = cdf(tprob)

        rprob = np.random.uniform()

        for i in xrange(len(tprob)):
            
            if tprob_range[i] < rprob and rprob < tprob_range[i+1]:
                
                print ('Reach ', self.env.dir[i])
                return i # return which action to run

    def reward_function(self):
        
        return -0.1
        #return self.discount ** self.step * self.value_function()

    def value_function(self, state = None, index = False):
        
        maximum = 0
        
        if state == None:

            state = self.state

        if index:
        
            return np.max(self.q_value[state[0], state[1]]), np.argmax(self.q_value[state[0], state[1]])
        
        else:
            
            return np.max(self.q_value[state[0], state[1]])

    def q_function(self, state, action):

        return q_value[state[0], state[1], action]

    def show(self):
        
        from utils import print2

        for h in reversed(xrange(self.env.size[0])):

            for w in xrange(self.env.size[1]):

                #print2(self.value_function([h,w]))
                
                print('%3.3f' % (self.value[h,w] ), end = '')
                print2(' | ')
            
            print (' ')



        for h in reversed(xrange(self.env.size[0])):

            for w in xrange(self.env.size[1]):

                #print2(self.value_function([h,w]))
                
                if [h,w] == self.state:

                    print2(' o ')

                else:
                    
                    print2('   ')

                print2(' | ')
            
            print (' ')
