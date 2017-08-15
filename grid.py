import numpy as np
import sys


def print2(to):

    sys.stdout.write(str(to))


class grid:

    def __init__(self, size):

        # size = [h, w]
        self.size = size
        self.grid = np.zeros(size)
        self.ogrid = np.zeros(size)

    def set(self, coord, value):
        
        self.grid[coord[0], coord[1]] = value
        self.ogrid[coord[0], coord[1]] = value
    
    def set_goal(self, coord):

        self.grid[coord[0], coord[1]] = 100
        self.ogrid[coord[0], coord[1]] = 100
    
    def set_hole(self, coord):

        self.grid[coord[0], coord[1]] = np.nan
        self.ogrid[coord[0], coord[1]] = np.nan

    def show(self):

        for h in xrange(self.size[0]):
        
            for w in xrange(self.size[1]):

                if w == 0:
                    print2('|')
                
                print2(self.grid[h,w])
                print2('|')

            print ' ' 
            





