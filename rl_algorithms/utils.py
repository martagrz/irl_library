import numpy as np

def get_index(query,array): 
    bool_array = array == query
    #print(bool_array.shape)
    if bool_array.ndim==2:
        index = np.where(np.logical_and(bool_array[:,0],bool_array[:,1]))[0]
    elif bool_array.ndim==1: 
        index = np.where(bool_array)
    return index
