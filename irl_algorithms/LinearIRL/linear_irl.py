import numpy as np
import scipy
from scipy.optimize import linprog

class IRL:
    def __init__(self,n_cols,n_rows,dims=1,gamma=0.01):
        self.gamma = gamma
        self.dims = dims
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_states = n_rows * n_cols

    def _get_row_col(self,state):
        row = state // self.n_cols
        col = state % self.n_cols
        return row, col

    def _basis_transforms(self,x):
        row, col = self._get_row_col(x)
        basis = np.zeros(self.n_states)
        basis[x] = 1
        basis_row = np.zeros(self.n_rows)
        basis_row[row] = 1
        basis_col = np.zeros(self.n_cols)
        basis_col[col] = 1
        return [basis, basis_row, basis_col]
                
            
    def _get_traj_value(self,trajectory):
        n_states = len(trajectory)
        traj_value_vec = [0,0,0]
        for i in range(n_states):
            state = trajectory[i]
            state_basis = self._basis_transforms(state)
            gamma = self.gamma**i
            for dim in range(len(state_basis)): 
                basis_discount = state_basis[dim]*gamma
                traj_value_vec[dim] = traj_value_vec[dim] + basis_discount     
        traj_value_vec = np.array(traj_value_vec)
        return traj_value_vec
             
    def _get_new_alphas(self,optimal_traj_value, policy_traj_value):
        obj = np.zeros(len(optimal_traj_value))
        for i in range(len(obj)):
            dim_obj = 0
            diff = optimal_traj_value[i] - policy_traj_value[i]
            if diff < 0:
                dim_obj += diff*2
            else:
                dim_obj += diff
            obj[i] = dim_obj
        #print('obj',-obj)
        ####Linear programming solver goes here
        alphas = linprog(-obj,bounds=(-1,1))
        alphas = alphas['x']
        return alphas

    def train(self,optimal_policy,optimal_traj,policy_set,policy_traj):
        n_states = len(optimal_policy)
        n_policies = policy_set.shape[0]
        optimal_traj_value =  self._get_traj_value(optimal_traj)
        optimal_traj_value = optimal_traj_value*n_policies #returns vector of opt of length dims
        policy_traj_value = np.zeros_like(optimal_traj_value)
        #print('otv',otv.shape)
        for policy in range(n_policies):
            value = self._get_traj_value(policy_traj[policy])
            print
            policy_traj_value += value
        #reshape and stack over each dim 
        otv = optimal_traj_value[0]
        ptv = policy_traj_value[0]
        for dim in range(self.dims-1): 
            otv = np.concatenate((otv,optimal_traj_value[dim+1]))
            ptv = np.concatenate((ptv,policy_traj_value[dim+1]))
        #print('ptv', ptv)
        #print('otv',otv)
        new_alphas = self._get_new_alphas(otv,ptv)	
        print('alphas', new_alphas)

        def reward_func(state,new_alphas):
            basis_state = self._basis_transforms(state)
            #print('basis', basis_state)
            basis = basis_state[0]
            for i in range(self.dims-1):
                basis = np.concatenate((basis,basis_state[i+1]))
                
            reward_func = np.dot(new_alphas,basis)
            reward_func = np.sum(reward_func)
            #print(reward_func)
            return reward_func

        reward = np.zeros(n_states)
        for state in range(n_states): 
            reward[state] = reward_func(state,new_alphas)
        return reward








   
