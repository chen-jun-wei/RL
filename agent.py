from __future__ import print_function
import numpy as np
from utils import softmax, cdf
import copy
import time
class agent:

    def __init__(self, env, start, discount, reward = -0.1, tprob = 0.1):

        self.state = start
        self.discount = discount         
        self.env = env
        self.step = 0
        self.isBreak = False
        self.start_point = start
        self.reward = reward
        """
        Transition Probability
        
        h x w x a x s'  => [state] [choosed action] [reached state]
    
        in this example, number of choosed action is same as number of next state
        
        """

       # self.tprob = np.ones([env.size[0], env.size[1], len(env.dir), len(env.dir)]) / len(env.dir)

        #assert len(tprob) == len(env.dir)
       
       # fixed transition probability , E -> 0.7, W -> 0.1, S -> 0.1, N -> 0.1
       # self.tprob = np.tile(tprob, env.size[0] * env.size[1] * len(env.dir)).reshape([env.size[0], env.size[1], len(env.dir), len(env.dir)])
        
        # 0.7 for selected action, 0.1 for other 3 action

        self.tprob = self.get_tprob(tprob)

        # randomize the policy

        self.policy = np.random.randint(0, len(self.env.dir), (env.size[0], env.size[1]))
        
        self.q_value = np.zeros([env.size[0], env.size[1], len(env.dir)])
        
        self.value = np.zeros([env.size[0], env.size[1]])

        if not int(np.prod(np.array(env.size) > np.array(start))):        
            #continue
            print ('[!] Start Position is not in Environment')
            raise

        self.initialize()
    
    def initialize(self):

        for un in self.env.unavailable:

            self.q_value[un[0],un[1]] = self.env.grid[un[0], un[1]]
            self.value[un[0], un[1]] = self.env.grid[un[0], un[1]]
        
        for tm in self.env.terminate:

            self.q_value[tm[0], tm[1]] = self.env.grid[tm[0], tm[1]]
            self.value[tm[0], tm[1]] = self.env.grid[tm[0], tm[1]]
    
    def value_function(self, state, index = False):
        
        """

        Ignore the unset value
            

        """
        
        maximum = None

        maximum_index = None
        
        #print('State : {}'.format(state))
        # Check if next state is available

        for idx in xrange(len(self.env.dir)):

            nexts = self.move(state, self.env.dir[idx])
            #print ('Next : {}'.format(nexts))    
            
            
            if nexts in self.env.unavailable:

                continue

            if (nexts[0] >= 0 and nexts[0] < self.env.size[0]) and \
                    (nexts[1] >= 0 and nexts[1] < self.env.size[1]):
                
                if maximum == None:
                    maximum = self.q_value[state[0], state[1], idx]
                    maximum_index = idx
                else:
                    maximum_index = maximum_index if self.q_value[state[0], state[1], idx] > maximum else idx 
                    
                    maximum = max(maximum, self.q_value[state[0], state[1], idx])
                 
                #if np.isnan(maximum):

                 #   raise

        if index:
        
            return maximum,  np.argmax(self.q_value[state[0], state[1]])
        
        else:
            
            return maximum

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
    
    def get_tprob(self, tprob):
        
        # selected action -> reached state
        
        
        a_a = np.ones([len(self.env.dir), len(self.env.dir)]) * tprob

        np.fill_diagonal(a_a, 1 - tprob * (len(self.env.dir) - 1))

        return np.tile(a_a.flatten(), [self.env.size[0], self.env.size[1]]).reshape([self.env.size[0], self.env.size[1], len(self.env.dir), len(self.env.dir)])

    
    def random_walk(self, epoch, show = False):

        """
        
        Learning Process

        """

        self.show()

        print("\n\nBefore Value Iteration ... ")
        
        time.sleep(5)

        for i in xrange(epoch):
            print ('\n\n-----------------------------')
            print ('--         Epoch {}         --'.format(i+1))
            print ('-----------------------------\n\n')
            self.state = self.random_start()
            
            self.step = 0

            if self.state in self.env.terminate or self.state in self.env.unavailable:
                continue
            while not self.isBreak:
                print ('------------------------------')
                print ('Epoch : {}, Step : {}, State : {}'.format(i+1, self.step, self.state))
                #self.show()
                #print ('Choosed : ', self.env.dir[self.policy[self.state[0], self.state[1]]])
                r = self.act(self.policy[self.state[0], self.state[1]])
                self.step += 1
                #if show:
                    #self.show()
                    
            self.isBreak = False
        self.update_policy()

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
        print ('Start Point: ', coord)
        return coord # set start position

    def isReachable(self, next_state):
        
        try:
            self.env.grid[next_state[0], next_state[1]]
        except:
            return False
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
        
        #print ('Next State : ', self.env.grid[next_state[0], next_state[1]])

        self.state = next_state if self.isReachable(next_state) else self.state
         
        #print ('Reachable : ', self.isReachable(next_state))

        # update q value

        # pns : probable next state
        
        if self.state in self.env.terminate:

            # reset reward

            self.reward = 0
                
            self.isBreak = True

        #elif self.isReachable(next_state) :      
        else:

            maximum = -1e3

            for a in xrange(len(self.env.dir)):

                """

                pns : possible next state
            
                pnd : possible next direction

                """
                
                sum_over_next_state = 0

                for pns, pnd in zip(self.available()[0], self.available()[1]):    
                    if pns in self.env.unavailable:
                        continue
                    #print ('{}, {} : {}'.format(a, pnd, self.value_function(pns)))
                    sum_over_next_state += self.tprob[self.state[0], self.state[1], a, self.action_index(pnd)] * (self.reward + self.discount * self.value_function(pns))
                # update value
                    #print ('T x (reward + r * vk[s] : {} x ({} + {} * {})) = {}'.format(self.tprob[self.state[0], self.state[1], a, self.action_index(pnd)], self.reward_function(), self.discount, \
                     #       self.value_function(pns), self.tprob[self.state[0], self.state[1], a, self.action_index(pnd)] * \
                            #(self.reward_function() + self.discount * self.value_function(pns))
                
                
                #print ('Updating', self.q_value[self.state[0], self.state[1]])

                self.q_value[self.state[0], self.state[1], a] = sum_over_next_state#maximum
            
            self.value[self.state[0], self.state[1]] = self.value_function(self.state)

            #print ('Update : ', self.q_value[self.state[0], self.state[1]])
            
            #self.show()

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
                
                return i # return which action to run
    """
    def reward_function(self):
        
        return -0.1
        #return self.discount ** self.step * self.value_function()
    """
    """
    def value_function(self, state = None, index = False):
        
        #maximum = 0
        
        if state == None:

            state = self.state

        if index:
        
            return np.max(self.q_value[state[0], state[1]]), np.argmax(self.q_value[state[0], state[1]])
        
        else:
            
            return np.max(self.q_value[state[0], state[1]])
    """
    def q_function(self, state, action):

        return q_value[state[0], state[1], action]

    def show(self):
        
        
        print ('\nGrid \n')

        for h in reversed(xrange(self.env.size[0])):

            for w in xrange(self.env.size[1]):

                if [h,w] in self.env.unavailable:

                    print('  X  ', end = '')

                else:

                    print('%3.3f' % (self.value_function([h,w])), end = '')
                
                print(' | ', end = '')
            
            print (' ')

        print ('\n')
        print ('Current State')
        print ('\n')
        for h in reversed(xrange(self.env.size[0])):

            for w in xrange(self.env.size[1]):

                #print2(self.value_function([h,w]))
                
                if [h,w] == self.state:

                    print(' o ', end = '')

                else:
                    
                    print('   ', end = '')

                print(' | ', end = '')
            
            print (' ')

    def update_policy(self):
        
        print ("\n Updating Policy ...")

        for j in xrange(self.env.size[0]):

            for i in xrange(self.env.size[1]):

                _, self.policy[j, i] = self.value_function([j, i], index = True)
                
    
    def show_policy(self):
        
        print ('\n Policy \n ')
        for h in reversed(xrange(self.env.size[0])):

            for w in xrange(self.env.size[1]):
                
                #print (self.policy[h,w], end = '')
                p = self.env.dir[self.policy[h,w]]
                
                if [h,w] in self.env.unavailable:

                    print ('  X  ', end = '')
                    print ('|', end = '')
                    continue
                elif [h,w] in self.env.terminate:

                    print (' {} '.format(self.value_function([h,w])), end = '')
                    print ('|', end = '')
                    continue
               # print ('{}'.format(p))

                if p == 'E':

                    print ('  >  ', end = '')

                elif p == 'W':

                    print ('  <  ', end = '')

                elif p == 'S':

                    print ('  v  ', end = '')

                elif p == 'N':

                    print ('  ^  ', end = '')
                
                print ('|', end = '')
            
            print (' ')
