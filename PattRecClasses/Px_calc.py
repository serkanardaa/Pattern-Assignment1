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
            
    sum_states = Px.sum(axis = 1).reshape(Px.shape[0],1)
    norm_Px = Px/sum_states
    return norm_Px