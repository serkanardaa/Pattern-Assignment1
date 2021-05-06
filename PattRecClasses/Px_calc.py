import numpy as np
import numpy.matlib

def Px_calc(pD, x_series ):
    # INPUT 
    # Output distributions of each state (data type : list)
    # Series of X (data type: ndarray)
    
    num_rows = len(pD)
    num_cols = x_series.size 
    P = np.zeros((num_rows,num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            p =  pD[i].prob( x_series[j] )
            P[i,j] = p
    max_dists = np.max(P, axis = 0).reshape(1,-1) # Vector (1 x num_cols) that contains maximum probability of each observed x among all distributions 
    scale_factor = 1 / max_dists #scale factor for using it in forward algorithm to get the expected test results of c values
    

    return P, scale_factor