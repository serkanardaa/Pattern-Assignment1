import numpy as np
import numpy.matlib

def Px_calc(x_series, pD):
    # INPUT 
    # Series of X (data type: ndarray)
    # Output distributions of each state (data type : list)
    num_rows = x_series.size
    num_cols = len(pD)
    P = np.zeros((num_rows,num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            p =  pD[j].prob( x_series[i] )
            P[i,j] = p
    max_dists = np.max(P, axis = 1).reshape(-1,1) # Vector (num_rows x 1) that contains maximum probability of each observed x among all distributions 
    scale_factor = 1 / max_dists
    

    return P, scale_factor