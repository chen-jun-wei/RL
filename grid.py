import numpy as np
import sys


def print2(to):

    sys.stdout.write(str(to))


class grid:

    def __init__(self, size):

        # size = [h, w]
        self.size = size
        self.grid = np.zeros(size)
        self.ogrid = None
        self.dir = ['E', 'W', 'N', 'S'] 
        self.terminate = []
        self.unavailable = []
    def init(self):

        self.ogrid = np.copy(self.grid)

    def set(self, coord, value):
        
        self.grid[coord[0], coord[1]] = value
        
    def set_terminate(self, coord, value):

        self.grid[coord[0], coord[1]] = value
    
        self.terminate.append(coord)

    
    def set_unavailable(self, coord):

        self.grid[coord[0], coord[1]] = np.nan
        
        self.unavailable.append(coord)

    def show(self, p = 'c'):

        for h in xrange(self.size[0]):
            for w in xrange(self.size[1]):
                if w == 0:
                    print2('|')
                if p == 'c':
                    print2(self.grid[h,w])
                elif p == 'o':
                    print2(self.ogrid[h,w])
                print2('|')
            print ' ' 
            
        #self.grid.set




