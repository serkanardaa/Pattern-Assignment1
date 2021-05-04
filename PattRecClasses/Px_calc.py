import numpy as np
import numpy.matlib

def Px_calc(x_series, state_dists):
    # INPUT 
    # Series of X (data type: ndarray)
    # Output distributions of each state (data type : list)
    num_rows = x_series.size
    num_cols = len(state_dists)
    Px = np.zeros((num_rows,num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            p =  state_dists[j].prob( x_series[i] )
            Px[i,j] = p
    max_dists = np.max(Px, axis = 1).reshape(-1,1) # Vector (num_rows x 1) that contains maximum probability of each observed x among all distributions 
    scaled_Px =Px / max_dists
            

    return scaled_Px