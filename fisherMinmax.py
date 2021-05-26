# Library Imports
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import consumerUtility as cu
np.set_printoptions(precision=3)



############# Linear ###############

#### Max Oracle ####

def max_oracle_gd_linear(valuations, budgets, prices_0, learning_rate,  num_iters, decay = True):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    for iter in range(1, num_iters):
        if (not iter % 50):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        prices_hist.append(prices)

        demands = np.zeros(valuations.shape)
        for buyer in range(budgets.shape[0]):
            demands[buyer,:] = cu.get_linear_marshallian_demand(prices, budgets[buyer], valuations[buyer])
       
        demands_hist.append(demands)
        demand = np.sum(demands, axis = 0)
        demand = demand.clip(min=0.01)
        excess_demand = demand - 1
        
        if (decay) :
            step_size = iter**(-1/2)*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)

        prices = prices.clip(min=0.01)
        
    return (demands, prices, demands_hist, prices_hist)

#### Multi-step #####

def ms_gd_linear(valuations, budgets, prices_0, learning_rate , num_iters, decay = True):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    for iter_outer in range(1, num_iters):
        if (not iter_outer % 50):
            print(f" ----- Iteration {iter_outer}/{num_iters} ----- ")
        
        
        prices_hist.append(prices)
        demands = np.zeros(valuations.shape)  
                
        X = cp.Variable(valuations.shape)
        obj = cp.Maximize(np.sum(prices) + budgets.T @ cp.log(cp.sum(cp.multiply(valuations, X), axis = 1)))
        constr = [X>=0, X @ prices <= budgets]
        prob = cp.Problem(obj, constr)
        try:
            prob.solve(solver="ECOS")
        except cp.SolverError:
            prob.solve(solver="SCS")
            
            
        demands = X.value
        
        demands = demands.clip(min = 0)
        demands_hist.append(demands)
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay) :
            step_size = (iter_outer**(-1/2))*excess_demand
        
            prices += step_size*(prices > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*(prices > 0)
        
        prices = prices.clip(min=0.01)
        

    return (demands, prices, demands_hist, prices_hist)

############### Cobb-Douglas ###############
    
#### Max Oracle ####

def max_oracle_gd_cd(valuations, budgets, prices_0, learning_rate, num_iters, decay = False):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []

    for iter in range(1,num_iters):
        if (not iter % 50):
            print(f"Outer Iteration {iter}/{num_iters}")
        
        demands = np.zeros(valuations.shape)
        prices_hist.append(prices)

        for buyer in range(budgets.shape[0]):
            demands[buyer,:] = cu.get_CD_marshallian_demand(prices, budgets[buyer], valuations[buyer])
            demands[buyer,:] = demands[buyer,:].clip(min = 0.01)
        
        demands_hist.append(demands)
        demand = np.sum(demands, axis = 0)
        demand = demand.clip(min=0.001)
        excess_demand = demand - 1

        if (decay) :
            step_size = learning_rate*iter**(-1/2)*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)

        prices = prices.clip(min=0.01)
        

    return (demands, prices, demands_hist, prices_hist)

#### Multi-step #####

def ms_gd_cd(valuations, budgets, prices_0, learning_rate, num_iters, decay = False):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []

    for iter_outer in range(1, num_iters):
        if (not iter_outer % 50):
            print(f" ----- Iteration {iter_outer}/{num_iters} ----- ")
        
        prices_hist.append(prices)
        demands = np.zeros(valuations.shape)  

        X = cp.Variable(valuations.shape)
        obj = cp.Maximize(np.sum(prices) + budgets.T @ cp.sum(cp.multiply(valuations, cp.log(X)), axis = 1))
        constr = [X>=0, X @ prices <= budgets]
        prob = cp.Problem(obj, constr)
        try:
            prob.solve(solver="SCS")
        except cp.SolverError:
            prob.solve(solver="ECOS")
            
        demands = X.value
        demands = demands.clip(min = 0)
        
        demands_hist.append(demands)
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay) :
            step_size = learning_rate*(iter_outer**(-1/2))*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)

        prices = prices.clip(min=0.01)
        

    return (demands, prices, demands_hist, prices_hist)

############# Leontief ###############

def max_oracle_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters, decay = True):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    
    for iter in range(1, num_iters):
        if (not iter % 50):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        demands = np.zeros(valuations.shape)
        prices_hist.append(prices)

        for buyer in range(budgets.shape[0]):
            demands[buyer,:] = cu.get_leontief_marshallian_demand(prices, budgets[buyer], valuations[buyer])
            demands[buyer,:] = demands[buyer,:]
        
        demands = demands.clip(min=0)
        demands_hist.append(demands)
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1

        if (decay) :
            step_size = learning_rate*(iter**(-1/2))*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)
        
        prices = prices.clip(min = 0.00001)
        
    return (demands, prices, demands_hist, prices_hist)

def ms_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters, decay = True):
    prices = prices_0
    prices_hist = []
    demands_hist = []
    for iter_outer in range(1, num_iters):
        if (not iter_outer % 50):
            print(f" ----- Iteration {iter_outer}/{num_iters} ----- ")
        
        prices_hist.append(prices)
        demands = np.zeros(valuations.shape)
        
        X = cp.Variable(valuations.shape)
        obj = cp.Maximize(np.sum(prices) + budgets.T @ cp.log(cp.min(cp.multiply(X, (1/valuations)), axis = 1)))
        constr = [X>=0, X @ prices <= budgets]
        prob = cp.Problem(obj, constr)
        
        try:
            prob.solve(solver="ECOS")
        except cp.SolverError:
            prob.solve(solver="SCS")
            
            
        demands = X.value
        
     
        demands = demands.clip(min = 0)
        demands_hist.append(demands)
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        
        if (decay) :
            step_size = learning_rate*iter_outer**(-1/2)*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)

        prices = prices.clip(min=0.00001)
        
        
    return (demands, prices, demands_hist, prices_hist)
