# Customer models including reservation prices and arrival rates
# state: [time, stock]
# action: [price, stock_reorder, sell={0,1}]
import numpy as np


def exp_demand(params, stochastic):
    def function(price, time):
        arrival_rate = params[0]-params[1]*time
        if stochastic:
            demand = np.random.poisson(arrival_rate) * np.exp(-1.1*price/100) + 2*time/10
        else:
            demand = arrival_rate * np.exp(-1.3*price/100) + 2*time/10
        return demand
    return function


def linear_demand(params, stochastic):
    def function(price, time):
        arrival_rate = params[0]-params[1]*time
        if stochastic:
            demand = np.random.poisson(arrival_rate) + params[2] - params[3]*price
        else:
            demand = arrival_rate + params[2] - params[3]*price
        return demand
    return function

