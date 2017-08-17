import numpy as np
import sys

def softmax(neurons, axis):
    
    exp = np.exp(neurons)
    
    total = np.sum(exp, axis = axis, keepdims = True)
    
    return exp / total


def cdf(probability):
    
#    print 'original : ', probability

    for i in xrange(len(probability) + 1):

        if i == 0:
            
            probability = np.insert(probability, 0, 0)

            continue
        
        probability[i] += probability[i-1]
        
     #   print '{} : {}'.format(i, probability)
    return probability


def print2(to):

    sys.stdout.write(str(to))
