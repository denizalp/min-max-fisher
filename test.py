
import fisherMinmax as fm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import consumerUtility as cu
from datetime import date

# Objective Functions for linear, Cobb-Douglas, and Leontief

def get_obj_linear(prices, demands, budgets, valuations):
    utils = np.sum(valuations*demands, axis = 1)
    return np.sum(prices) + budgets.T @ np.log(utils)

def get_obj_cd(prices, demands, budgets, valuations):
    utils = np.prod(np.power(demands, valuations), axis= 1)

    return np.sum(prices) + budgets.T @ np.log(utils)

def get_obj_leontief(prices, demands, budgets, valuations):
    utils = np.min(demands/valuations, axis = 1)
    return np.sum(prices) + budgets.T @ np.log(utils)

# Function that run Max-Oracle Gradient Descent and Nested Gradient Descent Ascent Tests and returns data
def run_test(num_buyers, num_goods, learning_rate, num_experiments, num_iters_linear, num_iters_cd, num_iters_leontief):
    
    prices_hist_mogd_linear_all_low = []
    demands_hist_mogd_linear_all_low = []
    obj_hist_mogd_linear_all_low = []
    prices_hist_msgd_linear_all_low = []
    demands_hist_msgd_linear_all_low = []
    obj_hist_msgd_linear_all_low = []
    prices_hist_mogd_cd_all_low = []
    demands_hist_mogd_cd_all_low = []
    obj_hist_mogd_cd_all_low = []
    prices_hist_msgd_cd_all_low = []
    demands_hist_msgd_cd_all_low = []
    obj_hist_msgd_cd_all_low = []
    prices_hist_mogd_leontief_all_low = []
    demands_hist_mogd_leontief_all_low = []
    obj_hist_mogd_leontief_all_low = []
    prices_hist_msgd_leontief_all_low = []
    demands_hist_msgd_leontief_all_low = []
    obj_hist_msgd_leontief_all_low = []
    prices_hist_mogd_linear_all_high = []
    demands_hist_mogd_linear_all_high = []
    obj_hist_mogd_linear_all_high = []
    prices_hist_msgd_linear_all_high = []
    demands_hist_msgd_linear_all_high = []
    obj_hist_msgd_linear_all_high = []
    prices_hist_mogd_cd_all_high = []
    demands_hist_mogd_cd_all_high = []
    obj_hist_mogd_cd_all_high = []
    prices_hist_msgd_cd_all_high = []
    demands_hist_msgd_cd_all_high = []
    obj_hist_msgd_cd_all_high = []
    prices_hist_mogd_leontief_all_high = []
    demands_hist_mogd_leontief_all_high = []
    obj_hist_mogd_leontief_all_high = []
    prices_hist_msgd_leontief_all_high = []
    demands_hist_msgd_leontief_all_high = []
    obj_hist_msgd_leontief_all_high = []

    for experiment_num in range(num_experiments):

        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")

        # Initialize random market parameters
        valuations = np.random.rand(num_buyers, num_goods)*10 + 5
        budgets = np.random.rand(num_buyers)*10 + 100
        
        
        print(f"****** Market Parameters ******\nval = {valuations}\n budgets = {budgets}\n")
        print(f"*******************************")
        
        print(f"------------ Low Initial Prices ------------")

        print(f"------ Linear Fisher Market ------\n")
        prices_0  = np.random.rand(num_goods)*10 + 5

        print(f"***** Max-Oracle Gradient Descent *****")
        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_linear(valuations, budgets, prices_0, learning_rate, num_iters_linear)
        
        prices_hist_mogd_linear_all_low.append(prices_mogd)
        demands_hist_mogd_linear_all_low.append(demands_mogd)
        objective_values = []
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_mogd_linear_all_low.append(objective_values)
        
        print(f"***** Nested Gradient Descent-Ascent *****")
        
        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_linear(valuations, budgets, prices_0, learning_rate, num_iters_linear)
        prices_hist_msgd_linear_all_low.append(prices_msgd)
        demands_hist_msgd_linear_all_low.append(demands_msgd)
        objective_values = []
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        
        obj_hist_msgd_linear_all_low.append(objective_values)
        
        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
        prices_0  = np.random.rand(num_goods) + 5
        print(f"------ Cobb-Douglas Fisher Market ------")
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T # Normalize valuations for Cobb-Douglas
        
        
        print(f"***** Max-Oracle Gradient Descent *****")

        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_cd)
        prices_hist_mogd_cd_all_low.append(prices_mogd)
        demands_hist_mogd_cd_all_low.append(demands_mogd)
        objective_values = []
        
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_mogd_cd_all_low.append(objective_values)
        
        print(f"***** Nested Gradient Descent Ascent *****")
        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_cd)
        prices_hist_msgd_cd_all_low.append(prices_msgd)
        demands_hist_msgd_cd_all_low.append(demands_msgd)
        objective_values = []
        
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_msgd_cd_all_low.append(objective_values)

        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
    
        print(f"------ Leontief Fisher Market ------")
        prices_0  = np.random.rand(num_goods) +5
        print(f"***** Max-Oracle Gradient Descent *****")
        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters_leontief)
        prices_hist_mogd_leontief_all_low.append(prices_mogd)
        demands_hist_mogd_leontief_all_low.append(demands_mogd)
        objective_values = []
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_mogd_leontief_all_low.append(objective_values)        
        
        print(f"***** Nested Gradient Descent Ascent *****")

        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters_leontief)
        prices_hist_msgd_leontief_all_low.append(prices_msgd)
        demands_hist_msgd_leontief_all_low.append(demands_msgd)
        objective_values = []
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_msgd_leontief_all_low.append(objective_values)

        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
        
        print(f"------------ High Initial Prices ------------")

        

        print(f"------ Linear Fisher Market ------\n")
        
        prices_0  = np.random.rand(num_goods)*5  + 50
        print(f"***** Max-Oracle Gradient Descent *****")
        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_linear(valuations, budgets, prices_0, learning_rate, num_iters_linear)
        
        prices_hist_mogd_linear_all_high.append(prices_mogd)
        demands_hist_mogd_linear_all_high.append(demands_mogd)
        objective_values = []
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_mogd_linear_all_high.append(objective_values)
        
        print(f"***** Nested Gradient Descent-Ascent *****")
        
        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_linear(valuations, budgets, prices_0, learning_rate, num_iters_linear)
        prices_hist_msgd_linear_all_high.append(prices_msgd)
        demands_hist_msgd_linear_all_high.append(demands_msgd)
        objective_values = []
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        
        obj_hist_msgd_linear_all_high.append(objective_values)
        
        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")

        print(f"------ Cobb-Douglas Fisher Market ------")
    
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T # Normalize valuations for Cobb-Douglas
        
        
        print(f"***** Max-Oracle Gradient Descent *****")

        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_cd)
        prices_hist_mogd_cd_all_high.append(prices_mogd)
        demands_hist_mogd_cd_all_high.append(demands_mogd)
        objective_values = []
        
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_mogd_cd_all_high.append(objective_values)
        
        print(f"***** Nested Gradient Descent Ascent *****")
        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_cd)
        prices_hist_msgd_cd_all_high.append(prices_msgd)
        demands_hist_msgd_cd_all_high.append(demands_msgd)
        objective_values = []
        
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_msgd_cd_all_high.append(objective_values)

        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
    
        print(f"------ Leontief Fisher Market ------")

        
        
        print(f"***** Max-Oracle Gradient Descent *****")
        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters_leontief)
        prices_hist_mogd_leontief_all_high.append(prices_mogd)
        demands_hist_mogd_leontief_all_high.append(demands_mogd)
        objective_values = []
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_mogd_leontief_all_high.append(objective_values)        
        
        print(f"***** Nested Gradient Descent Ascent *****")

        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters_leontief)
        prices_hist_msgd_leontief_all_high.append(prices_msgd)
        demands_hist_msgd_leontief_all_high.append(demands_msgd)
        objective_values = []
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_msgd_leontief_all_high.append(objective_values)

        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
        
    return (prices_hist_mogd_linear_all_low,
            demands_hist_mogd_linear_all_low,
            obj_hist_mogd_linear_all_low,
            prices_hist_msgd_linear_all_low,
            demands_hist_msgd_linear_all_low,
            obj_hist_msgd_linear_all_low,
            prices_hist_mogd_cd_all_low,
            demands_hist_mogd_cd_all_low,
            obj_hist_mogd_cd_all_low,
            prices_hist_msgd_cd_all_low,
            demands_hist_msgd_cd_all_low,
            obj_hist_msgd_cd_all_low,
            prices_hist_mogd_leontief_all_low,
            demands_hist_mogd_leontief_all_low,
            obj_hist_mogd_leontief_all_low,
            prices_hist_msgd_leontief_all_low,
            demands_hist_msgd_leontief_all_low,
            obj_hist_msgd_leontief_all_low,
            prices_hist_mogd_linear_all_high,
            demands_hist_mogd_linear_all_high,
            obj_hist_mogd_linear_all_high,
            prices_hist_msgd_linear_all_high,
            demands_hist_msgd_linear_all_high,
            obj_hist_msgd_linear_all_high,
            prices_hist_mogd_cd_all_high,
            demands_hist_mogd_cd_all_high,
            obj_hist_mogd_cd_all_high,
            prices_hist_msgd_cd_all_high,
            demands_hist_msgd_cd_all_high,
            obj_hist_msgd_cd_all_high,
            prices_hist_mogd_leontief_all_high,
            demands_hist_mogd_leontief_all_high,
            obj_hist_mogd_leontief_all_high,
            prices_hist_msgd_leontief_all_high,
            demands_hist_msgd_leontief_all_high,
            obj_hist_msgd_leontief_all_high)


if __name__ == '__main__':

    num_experiments = 500
    num_buyers =  5
    num_goods = 8
    learning_rate =  5
    num_iters_linear = 500
    num_iters_cd = 300
    num_iters_leontief = 700
    # results = 
    (prices_hist_mogd_linear_all_low,
                demands_hist_mogd_linear_all_low,
                obj_hist_mogd_linear_all_low,
                prices_hist_msgd_linear_all_low,
                demands_hist_msgd_linear_all_low,
                obj_hist_msgd_linear_all_low,
                prices_hist_mogd_cd_all_low,
                demands_hist_mogd_cd_all_low,
                obj_hist_mogd_cd_all_low,
                prices_hist_msgd_cd_all_low,
                demands_hist_msgd_cd_all_low,
                obj_hist_msgd_cd_all_low,
                prices_hist_mogd_leontief_all_low,
                demands_hist_mogd_leontief_all_low,
                obj_hist_mogd_leontief_all_low,
                prices_hist_msgd_leontief_all_low,
                demands_hist_msgd_leontief_all_low,
                obj_hist_msgd_leontief_all_low,
                prices_hist_mogd_linear_all_high,
                demands_hist_mogd_linear_all_high,
                obj_hist_mogd_linear_all_high,
                prices_hist_msgd_linear_all_high,
                demands_hist_msgd_linear_all_high,
                obj_hist_msgd_linear_all_high,
                prices_hist_mogd_cd_all_high,
                demands_hist_mogd_cd_all_high,
                obj_hist_mogd_cd_all_high,
                prices_hist_msgd_cd_all_high,
                demands_hist_msgd_cd_all_high,
                obj_hist_msgd_cd_all_high,
                prices_hist_mogd_leontief_all_high,
                demands_hist_mogd_leontief_all_high,
                obj_hist_mogd_leontief_all_high,
                prices_hist_msgd_leontief_all_high,
                demands_hist_msgd_leontief_all_high,
                obj_hist_msgd_leontief_all_high) = run_test(num_buyers, num_goods, learning_rate, num_experiments, num_iters_linear, num_iters_cd, num_iters_leontief)



    # # # Save data
    obj_hist_mogd_linear_all_low = np.array(obj_hist_mogd_linear_all_low)
    obj_hist_msgd_linear_all_low = np.array(obj_hist_msgd_linear_all_low)
    obj_hist_mogd_cd_all_low = np.array(obj_hist_mogd_cd_all_low)
    obj_hist_msgd_cd_all_low = np.array(obj_hist_msgd_cd_all_low)
    obj_hist_mogd_leontief_all_low = np.array(obj_hist_mogd_leontief_all_low)
    obj_hist_msgd_leontief_all_low = np.array(obj_hist_msgd_leontief_all_low)

    obj_hist_mogd_linear_all_high = np.array(obj_hist_mogd_linear_all_high)
    obj_hist_msgd_linear_all_high = np.array(obj_hist_msgd_linear_all_high)
    obj_hist_mogd_cd_all_high = np.array(obj_hist_mogd_cd_all_high)
    obj_hist_msgd_cd_all_high = np.array(obj_hist_msgd_cd_all_high)
    obj_hist_mogd_leontief_all_high = np.array(obj_hist_mogd_leontief_all_high)
    obj_hist_msgd_leontief_all_high = np.array(obj_hist_msgd_leontief_all_high)

    obj_hist_mogd_linear_low =  pd.DataFrame( obj_hist_mogd_linear_all_low)
    obj_hist_msgd_linear_low =  pd.DataFrame( obj_hist_msgd_linear_all_low)
    obj_hist_mogd_cd_low =  pd.DataFrame(obj_hist_mogd_cd_all_low)
    obj_hist_msgd_cd_low =  pd.DataFrame( obj_hist_msgd_cd_all_low)
    obj_hist_mogd_leontief_low =  pd.DataFrame( obj_hist_mogd_leontief_all_low)
    obj_hist_msgd_leontief_low =  pd.DataFrame(obj_hist_msgd_leontief_all_low)

    obj_hist_mogd_linear_high =  pd.DataFrame( obj_hist_mogd_linear_all_high)
    obj_hist_msgd_linear_high =  pd.DataFrame( obj_hist_msgd_linear_all_high)
    obj_hist_mogd_cd_high =  pd.DataFrame(obj_hist_mogd_cd_all_high)
    obj_hist_msgd_cd_high =  pd.DataFrame( obj_hist_msgd_cd_all_high)
    obj_hist_mogd_leontief_high =  pd.DataFrame( obj_hist_mogd_leontief_all_high)
    obj_hist_msgd_leontief_high =  pd.DataFrame(obj_hist_msgd_leontief_all_high)

    obj_hist_mogd_linear_low.to_csv("data/obj/obj_hist_mogd_linear_low.csv")
    obj_hist_msgd_linear_low.to_csv("data/obj/obj_hist_msgd_linear_low.csv")
    obj_hist_mogd_cd_low.to_csv("data/obj/obj_hist_mogd_cd_low.csv")
    obj_hist_msgd_cd_low.to_csv("data/obj/obj_hist_msgd_cd_low.csv")
    obj_hist_mogd_leontief_low.to_csv("data/obj/obj_hist_mogd_leontief_low.csv")
    obj_hist_msgd_leontief_low.to_csv("data/obj/obj_hist_msgd_leontief_low.csv")

    obj_hist_mogd_linear_high.to_csv("data/obj/obj_hist_mogd_linear_high.csv")
    obj_hist_msgd_linear_high.to_csv("data/obj/obj_hist_msgd_linear_high.csv")
    obj_hist_mogd_cd_high.to_csv("data/obj/obj_hist_mogd_cd_high.csv")
    obj_hist_msgd_cd_high.to_csv("data/obj/obj_hist_msgd_cd_high.csv")
    obj_hist_mogd_leontief_high.to_csv("data/obj/obj_hist_mogd_leontief_high.csv")
    obj_hist_msgd_leontief_high.to_csv("data/obj/obj_hist_msgd_leontief_high.csv")

    prices_hist_mogd_linear_all_low = np.array(prices_hist_mogd_linear_all_low)
    prices_hist_msgd_linear_all_low = np.array(prices_hist_msgd_linear_all_low)
    prices_hist_mogd_cd_all_low = np.array(prices_hist_mogd_cd_all_low)
    prices_hist_msgd_linear_all_low = np.array(prices_hist_msgd_linear_all_low)
    prices_hist_mogd_leontief_all_low = np.array(prices_hist_mogd_leontief_all_low)
    prices_hist_msgd_leontief_all_low = np.array(prices_hist_msgd_leontief_all_low)

    prices_hist_mogd_linear_all_high = np.array(prices_hist_mogd_linear_all_high)
    prices_hist_msgd_linear_all_high = np.array(prices_hist_msgd_linear_all_high)
    prices_hist_mogd_cd_all_high = np.array(prices_hist_mogd_cd_all_high)
    prices_hist_msgd_linear_all_high = np.array(prices_hist_msgd_linear_all_high)
    prices_hist_mogd_leontief_all_high = np.array(prices_hist_mogd_leontief_all_high)
    prices_hist_msgd_leontief_all_high = np.array(prices_hist_msgd_leontief_all_high)

    prices_mogd_linear_low =  pd.DataFrame(prices_hist_mogd_linear_all_low)
    prices_ngd_linear_low =  pd.DataFrame(prices_hist_msgd_linear_all_low)
    prices_mogd_cd_low =  pd.DataFrame(prices_hist_mogd_cd_all_low)
    prices_ngd_cd_low =  pd.DataFrame(prices_hist_msgd_linear_all_low )
    prices_mogd_leontief_low =  pd.DataFrame( prices_hist_mogd_leontief_all_low)
    prices_ngd_leontief_low =  pd.DataFrame( prices_hist_msgd_leontief_all_low )

    prices_mogd_linear_high =  pd.DataFrame(prices_hist_mogd_linear_all_high)
    prices_ngd_linear_high =  pd.DataFrame(prices_hist_msgd_linear_all_high)
    prices_mogd_cd_high =  pd.DataFrame(prices_hist_mogd_cd_all_high)
    prices_ngd_cd_high =  pd.DataFrame(prices_hist_msgd_linear_all_high )
    prices_mogd_leontief_high =  pd.DataFrame( prices_hist_mogd_leontief_all_high)
    prices_ngd_leontief_high =  pd.DataFrame( prices_hist_msgd_leontief_all_high )

    prices_mogd_linear_low.to_csv("data/prices/prices_mogd_linear_low.csv")
    prices_ngd_linear_low.to_csv("data/prices/prices_ngd_linear_low.csv")
    prices_mogd_cd_low.to_csv("data/prices/prices_mogd_cd_low.csv")
    prices_ngd_cd_low.to_csv("data/prices/prices_ngd_cd_low.csv")
    prices_mogd_leontief_low.to_csv("data/prices/prices_mogd_leontief_low.csv")
    prices_ngd_leontief_low.to_csv("data/prices/prices_ngd_leontief_low.csv")

    prices_mogd_linear_high.to_csv("data/prices/prices_mogd_linear_high.csv")
    prices_ngd_linear_high.to_csv("data/prices/prices_ngd_linear_high.csv")
    prices_mogd_cd_high.to_csv("data/prices/prices_mogd_cd_high.csv")
    prices_ngd_cd_high.to_csv("data/prices/prices_ngd_cd_high.csv")
    prices_mogd_leontief_high.to_csv("data/prices/prices_mogd_leontief_high.csv")
    prices_ngd_leontief_high.to_csv("data/prices/prices_ngd_leontief_high.csv")

    #%%
    obj_mogd_linear_low = np.mean(obj_hist_mogd_linear_all_low, axis = 0)
    obj_msgd_linear_low = np.mean(obj_hist_msgd_linear_all_low, axis = 0)
    obj_mogd_cd_low = np.mean(obj_hist_mogd_cd_all_low, axis = 0)
    obj_msgd_cd_low = np.mean(obj_hist_msgd_cd_all_low, axis = 0)
    obj_mogd_leontief_low = np.mean(obj_hist_mogd_leontief_all_low, axis = 0)
    obj_msgd_leontief_low = np.mean(obj_hist_msgd_leontief_all_low, axis = 0)

    obj_mogd_linear_high = np.mean(obj_hist_mogd_linear_all_high, axis = 0)
    obj_msgd_linear_high = np.mean(obj_hist_msgd_linear_all_high, axis = 0)
    obj_mogd_cd_high = np.mean(obj_hist_mogd_cd_all_high, axis = 0)
    obj_msgd_cd_high = np.mean(obj_hist_msgd_cd_all_high, axis = 0)
    obj_mogd_leontief_high = np.mean(obj_hist_mogd_leontief_all_high, axis = 0)
    obj_msgd_leontief_high = np.mean(obj_hist_msgd_leontief_all_high, axis = 0)

    obj_mogd_leontief_low = obj_mogd_leontief_low[:-200]
    obj_mogd_leontief_high = obj_mogd_leontief_high[:-200]
    obj_msgd_leontief_low = obj_msgd_leontief_low[:-200]
    obj_msgd_leontief_high = obj_msgd_leontief_high[:-200]

    num_iters_linear = len(obj_mogd_linear_low)
    num_iters_cd = len(obj_mogd_cd_low)
    num_iters_leontief = len(obj_mogd_leontief_low)
    x_linear = np.linspace(1, num_iters_linear, num_iters_linear)
    x_cd = np.linspace(1, num_iters_cd, num_iters_cd)
    x_leontief = np.linspace(1, num_iters_leontief, num_iters_leontief)

    fig, axs = plt.subplots(2, 3) # Create a figure containing a single axes.
    # First row for experiments with low initial prices and
    # second row for experiments with high initial prices.

    # Add shift in plots to make the difference clearer
    axs[0,0].plot([iter for iter in range(num_iters_linear)], obj_mogd_linear_low, label = "Max-Oracle", alpha = 1, color = "b")
    axs[0,0].plot([iter for iter in range(num_iters_linear)], obj_msgd_linear_low, label = "Nested Gradient Descent Ascent", linestyle='dashed', alpha = 1, color = "orange")
    # axs[0,0].plot(x, (obj_mogd_linear[0]/15)/x + obj_mogd_linear[-1], color='green', linestyle='dashed', label = "1/T")
    axs[0,0].plot(x_linear, (obj_mogd_linear_low[0] - obj_mogd_linear_low[-1])*(x_linear**(-1/2)) + obj_mogd_linear_low[-1], color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[0,0].set_title("Linear Market", fontsize = "medium")
    axs[0,0].set_ylim(2100, 2500)

    axs[0,1].plot([iter for iter in range(num_iters_cd)], obj_mogd_cd_low , label = "Max-Oracle", color = "b")
    axs[0,1].plot([iter for iter in range(num_iters_cd)], obj_msgd_cd_low, label = "Nested Gradient Descent Ascent", linestyle='dashed', alpha = 1, color = "orange")
    # axs[0,1].plot(x, (obj_mogd_cd[0]/3)/x + obj_mogd_cd[-1], color='green', linestyle='dashed', label = "1/T")
    axs[0,1].plot(x_cd, (obj_mogd_cd_low[0] - obj_mogd_cd_low[-1])*(x_cd**(-1/2)) + obj_mogd_cd_low[-1], color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[0,1].set_title("Cobb-Douglas Market", fontsize = "medium")
    axs[0,1].set_ylim(-330, 200)

    axs[0,2].plot([iter for iter in range(num_iters_leontief)], obj_mogd_leontief_low, label = "Max-Oracle", color = "b")
    axs[0,2].plot([iter for iter in range(num_iters_leontief)], obj_msgd_leontief_low, label = "Nested Gradient Descent Ascent", linestyle='dashed', alpha = 1, color = "orange")
    # axs[1,0].plot(x, (obj_mogd_leontief[0]/4)/x + obj_mogd_leontief[-1], color='green', linestyle='dashed', label = "1/T")
    axs[0,2].plot(x_leontief, (obj_mogd_leontief_low[0] - obj_mogd_leontief_low[-1])*(x_leontief**(-1/2)) + obj_mogd_leontief_low[-1], color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[0,2].set_title("Leontief Market", fontsize = "medium")
    # axs[1, 1].axis('off')
    axs[0,2].set_ylim(-1600, -900)

    # Add shift in plots to make the difference clearer
    axs[1,0].plot([iter for iter in range(num_iters_linear)], obj_mogd_linear_high, label = "Max-Oracle", alpha = 1, color = "b")
    axs[1,0].plot([iter for iter in range(num_iters_linear)], obj_msgd_linear_high, label = "Nested Gradient Descent Ascent", linestyle='dashed', alpha = 0.8, color = "orange")
    # axs[0,0].plot(x, (obj_mogd_linear[0]/15)/x + obj_mogd_linear[-1], color='green', linestyle='dashed', label = "1/T")
    axs[1,0].plot(x_linear, (obj_mogd_linear_high[0] - obj_mogd_linear_high[-1]+5)*(x_linear**(-1/2)) + obj_mogd_linear_high[-1], color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[1,0].set_title("Linear Market", fontsize = "medium")
    axs[1,0].set_ylim(2115, 2145)

    axs[1,1].plot([iter for iter in range(num_iters_cd)], obj_mogd_cd_high , label = "Max-Oracle", alpha = 1, color = "b")
    axs[1,1].plot([iter for iter in range(num_iters_cd)], obj_msgd_cd_high, label = "Nested Gradient Descent Ascent", linestyle='dashed', alpha = 1, color = "orange")
    # axs[0,1].plot(x, (obj_mogd_cd[0]/3)/x + obj_mogd_cd[-1], color='green', linestyle='dashed', label = "1/T")
    axs[1,1].plot(x_cd, (obj_mogd_cd_high[0] - obj_mogd_cd_high[-1])*(x_cd**(-1/2)) + obj_mogd_cd_high[-1] , color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[1,1].set_title("Cobb-Douglas Market", fontsize = "medium")
    axs[1,1].set_ylim(-305, -290)

    axs[1,2].plot([iter for iter in range(num_iters_leontief)], obj_mogd_leontief_high, label = "Max-Oracle", alpha = 1, color = "b")
    axs[1,2].plot([iter for iter in range(num_iters_leontief)], obj_msgd_leontief_high, label = "Nested Gradient Descent Ascent", linestyle ='dashed', color = "orange")
    # axs[1,0].plot(x, (obj_mogd_leontief[0]/4)/x + obj_mogd_leontief[-1], color='green', linestyle='dashed', label = "1/T")
    axs[1,2].plot(x_leontief, (obj_mogd_leontief_high[0] - obj_mogd_leontief_high[-1])*(x_leontief**(-1/2)) + obj_mogd_leontief_high[-1] - 3, color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[1,2].set_title("Leontief Market", fontsize = "medium")


    for ax in axs.flat:
        ax.set(xlabel='Iteration Number', ylabel='Objective Value')
        ax.yaxis.set_ticks([])
    for ax in axs.flat:
        ax.label_outer()

    name = "obj_graphs"
    plt.savefig(f"graphs/{name}.jpg")
    plt.show()

