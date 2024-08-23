import numpy as np

def focused_CE(p, alpha, gamma):
    ...
    
    
    
    FL = np.dot(np.dot(-alpha, np.power((1-p), gamma)), np.log(p))
    
    