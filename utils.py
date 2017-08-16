import numpy as np



def softmax(neurons, axis):
    
    exp = np.exp(neurons)
    
    total = np.sum(exp, axis = axis, keepdims = True)
    
    return exp / total


def cdf(probability):

    for i in xrange(len(probability)):

        if i == 0:
            
            np.insert(probability, 0, 0)

            continue

        probability[i] += probability[i-1]

    return probability


