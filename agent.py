import numpy as np



class agent:

    def __init__(self, env, start, discount):

        self.state = start
        self.discount = discount         
        self.env = env
        self.step = 0
       
        # transition probability
        
        # h x w         x a            x s'
        #[state] [choosed action] [reached state]
        #
        # in this example, choosed action is same as next state
        
        self.tprob = np.ones(env.grid[0], env.grid[1], len(env.dir), len(env.dir)) / len(env.dir)
        
        self.policy = np.random.randint(0, len(self.env.dir), (env.grid[0], env.grid[1]))
        
        self.value = np.random.
        if not np.prod(env.size > np.array(start)):        
        
            print '[!] Start Position is not in Environment'
    

    def _probability(self):

        return np.random.uniform()

    def _available(self):
        
        available = []
        
        for d in self.dir:

            temp = self._move(self.state, d)

            try:
                _ = self.env[temp[0], temp[1]]

                if not np.isnan(_):
                    available.append(temp)
            except:
                pass

        return available # availabel state, not available move


    def _move(self, state, direction):
        
        next_state = state

        if direction == 'W':

            next_state += [0,-1]

        elif direction == 'E':

            next_state += [0, 1]

        elif direction == 'N':

            next_state += [1, 0]

        elif direction == 'S':

            next_state += [-1,0]

        else:

            print 'No Direction'

            raise

        return next_state

    

        pass

    def get_value(self, state):
        
            
        max_action = 0

        for a in self.available(): # iterate along available action
            
            value = 0

            for next_s in self._available(): # each action change reach each state

                value +=  transition(state, next_s) * (Reward + Value(-1)))
            
                                 

    

        self.grid.set(state, max(max_val, d) for d in self.dir)
    
    def random_walk(self, total_epoch):
        
        for i in xrange(total_epoch):

            self.random_start()

            if self.state in self.grid.terminate:

                continue

            while True:

                r = self.act()

    def random_start(self):
        
        while True:

            coord = [np.random.choice(self.size[0]), \
                    np.random.choice(self.size[1])]
            
            if coord in self.grid.unavailable:

                continue

            break

        return coord # set start


    def act(self, action):
        
        # random select action
        #action = self.available() 

        # h x w x a x s'
    
        # stochastic transition probability
        
        tprob = softmax(self.policy[self.state[0],\
                self.state[1], a])
        
        # available state [index]

        self.state = self.avaibale()[self.transition(tprob, np.random.uniform())]
         
    # transition probability range, random probability

    def transition(self, tprob, rprob):

        tprob = cdf(tprob)

        for i in xrange(len(tprob)):
            
            if tprob[i] < rprob and rprob < tprob[i+1]:

                return i # return which action to run

    def reward_function(self, t, discount):

        return self.reward += discount ** t * self.env.grid[self.state[0], self.state[1]

    def value_function(self, state):
        
        temp = 0

        for i in len(self.env.dir):
        
            self.[state[0], state[1], i]
         
        return

    def q_function(self, state, action):


