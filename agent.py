import numpy as np



class agent:

    def __init__(self, env, start, discount):

        self.state = start
        self.discount = discount         
        self.env = env
        
        # transition probability
        
        # h x w         x a            x s'
        #[state] [choosed action] [reached state]
        self.policy = np.ones(env.grid[0], env.grid[1], len(env.dir), len(env.dir)) / len(env.dir)


        
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

        return available


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

    
    def transition(self, state, action):

        pass

    def get_value(self, state):
        
            
        max_action = 0

        for a in self.available(): # iterate along available action
            
            for n in self._available(): # each action change reach each state

                max(max_action, transition(state, n) * (Reward + Value(-1)))
            
                 



        self.grid.set(state, max(max_val, d) for d in self.dir)
    

    def reward(self):

        pass    

    
   
