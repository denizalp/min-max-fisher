#%%
import fisherMinmax as fm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import consumerUtility as cu
#%%
# Objective Functions for linear, Cobb-Douglas, and Leontief

def get_obj_linear(prices, demands, budgets, valuations):
    # utils = np.zeros(budgets.shape[0])
    # for buyer in range(budgets.shape[0]):
    #     utils[buyer] = cu.get_linear_indirect_utill(prices, budgets[buyer], valuations[buyer,:])
    utils = np.sum(valuations*demands, axis = 1)
    return np.sum(prices) + budgets.T @ np.log(utils)

def get_obj_cd(prices, demands, budgets, valuations):
    # utils = np.zeros(budgets.shape[0])

    # for buyer in range(budgets.shape[0]):
    #     utils[buyer] = cu.get_CD_indirect_util(prices, budgets[buyer], valuations[buyer,:])

    utils = np.prod(np.power(demands, valuations), axis= 1)

    return np.sum(prices) - budgets.T @ np.log(utils)

def get_obj_leontief(prices, demands, budgets, valuations):
    # utils = np.zeros(budgets.shape[0])

    # for buyer in range(budgets.shape[0]):
    #     utils[buyer] = cu.get_CD_indirect_util(prices, budgets[buyer], valuations[buyer,:])

    utils = np.min(demands/valuations, axis = 1)
    return np.sum(prices) - budgets.T @ np.log(utils)

# Function that run Max-Oracle Gradient Descent and Nested Gradient Descent Ascent Tests and returns data
def run_test(num_buyers, num_goods, learning_rate, num_iters_outer, num_iters_inner, num_experiments):
    
    prices_hist_mogd_linear_all = []
    demands_hist_mogd_linear_all = []
    obj_hist_mogd_linear_all = []
    prices_hist_msgd_linear_all = []
    demands_hist_msgd_linear_all = []
    obj_hist_msgd_linear_all = []
    prices_hist_mogd_cd_all = []
    demands_hist_mogd_cd_all = []
    obj_hist_mogd_cd_all = []
    prices_hist_msgd_cd_all = []
    demands_hist_msgd_cd_all = []
    obj_hist_msgd_cd_all = []
    prices_hist_mogd_leontief_all = []
    demands_hist_mogd_leontief_all = []
    obj_hist_mogd_leontief_all = []
    prices_hist_msgd_leontief_all = []
    demands_hist_msgd_leontief_all = []
    obj_hist_msgd_leontief_all = []

    for experiment_num in range(num_experiments):
        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")

        valuations = np.random.rand(num_buyers, num_goods)*10 + 5
        budgets = np.random.rand(num_buyers)*10 + 5
        prices_0  = np.random.rand(num_goods)*5  + 20
        
        print(f"****** Market Parameters ******\nval = {valuations}\n budgets = {budgets}\n prices_0 ={prices_0}\n")
        print(f"*******************************")
        print(f"------ Linear Fisher Market ------\n")

        print(f"***** Max-Oracle Gradient Descent *****")
        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_linear(valuations, budgets, prices_0, learning_rate, num_iters_outer)
        
        prices_hist_mogd_linear_all.append(prices_mogd)
        demands_hist_mogd_linear_all.append(demands_mogd)
        objective_values = []
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_mogd_linear_all.append(objective_values)
        
        print(f"***** Nested Gradient Descent-Ascent *****")
        
        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_linear(valuations, budgets, prices_0, learning_rate, num_iters_outer, num_iters_inner)
        prices_hist_msgd_linear_all.append(prices_msgd)
        demands_hist_msgd_linear_all.append(demands_msgd)
        objective_values = []
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        
        obj_hist_msgd_linear_all.append(objective_values)
        
        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")

        print(f"------ Cobb-Douglas Fisher Market ------")
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T # Normalize valuations for Cobb-Douglas
        
        
        print(f"***** Max-Oracle Gradient Descent *****")

        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_outer)
        prices_hist_mogd_cd_all.append(prices_mogd)
        demands_hist_mogd_cd_all.append(demands_mogd)
        objective_values = []
        
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_mogd_cd_all.append(objective_values)
        
        print(f"***** Nested Gradient Descent Ascent *****")
        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_outer, num_iters_inner)
        prices_hist_msgd_cd_all.append(prices_msgd)
        demands_hist_msgd_cd_all.append(demands_msgd)
        objective_values = []
        
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_linear(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_msgd_cd_all.append(objective_values)

        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
    
        print(f"------ Leontief Fisher Market ------")
        
        print(f"***** Max-Oracle Gradient Descent *****")
        demands_mogd, prices_mogd, demands_hist_mogd, prices_hist_mogd = fm.max_oracle_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters_outer)
        prices_hist_mogd_leontief_all.append(prices_mogd)
        demands_hist_mogd_leontief_all.append(demands_mogd)
        objective_values = []
        for x, p in zip(demands_hist_mogd, prices_hist_mogd):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_mogd_leontief_all.append(objective_values)        
        
        print(f"***** Nested Gradient Descent Ascent *****")

        demands_msgd, prices_msgd, demands_hist_msgd, prices_hist_msgd = fm.ms_gd_leontief(valuations, budgets, prices_0, learning_rate, num_iters_outer, num_iters_inner)
        prices_hist_msgd_leontief_all.append(prices_msgd)
        demands_hist_msgd_leontief_all.append(demands_msgd)
        objective_values = []
        for x, p in zip(demands_hist_msgd, prices_hist_msgd):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_msgd_leontief_all.append(objective_values)

        print(f"Max Oracle Gradient Descent Prices: {prices_mogd}\nMulti-Step Gradient Descent Prices: {prices_msgd}\n")
        
    return (prices_hist_mogd_linear_all,
            demands_hist_mogd_linear_all,
            obj_hist_mogd_linear_all,
            prices_hist_msgd_linear_all,
            demands_hist_msgd_linear_all,
            obj_hist_msgd_linear_all,
            prices_hist_mogd_cd_all,
            demands_hist_mogd_cd_all,
            obj_hist_mogd_cd_all,
            prices_hist_msgd_cd_all,
            demands_hist_msgd_cd_all,
            obj_hist_msgd_cd_all,
            prices_hist_mogd_leontief_all,
            demands_hist_mogd_leontief_all,
            obj_hist_mogd_leontief_all,
            prices_hist_msgd_leontief_all,
            demands_hist_msgd_leontief_all,
            obj_hist_msgd_leontief_all)


# if __name__ == '__main__':

#%%
num_experiments = 3
num_buyers =  5
num_goods = 8
learning_rate =  0.9
num_iters_outer = 800
num_iters_inner = 2000

# results = 
(prices_hist_mogd_linear_all,
        demands_hist_mogd_linear_all,
        obj_hist_mogd_linear_all,
        prices_hist_msgd_linear_all,
        demands_hist_msgd_linear_all,
        obj_hist_msgd_linear_all,
        prices_hist_mogd_cd_all,
        demands_hist_mogd_cd_all,
        obj_hist_mogd_cd_all,
        prices_hist_msgd_cd_all,
        demands_hist_msgd_cd_all,
        obj_hist_msgd_cd_all,
        prices_hist_mogd_leontief_all,
        demands_hist_mogd_leontief_all,
        obj_hist_mogd_leontief_all,
        prices_hist_msgd_leontief_all,
        demands_hist_msgd_leontief_all,
        obj_hist_msgd_leontief_all) = run_test(num_buyers, num_goods, learning_rate, num_iters_outer, num_iters_inner, num_experiments)



# %%
obj_hist_mogd_linear_all = np.array(obj_hist_mogd_linear_all)
obj_hist_msgd_linear_all = np.array(obj_hist_msgd_linear_all)
obj_hist_mogd_cd_all = np.array(obj_hist_mogd_cd_all)
obj_hist_msgd_cd_all = np.array(obj_hist_msgd_cd_all)
obj_hist_mogd_leontief_all = np.array(obj_hist_mogd_leontief_all)
obj_hist_msgd_leontief_all = np.array(obj_hist_msgd_leontief_all)

obj_hist_mogd_linear =  pd.DataFrame( obj_hist_mogd_linear_all, )
obj_hist_ngd_linear =  pd.DataFrame( obj_hist_msgd_linear_all)
obj_hist_mogd_cd =  pd.DataFrame(obj_hist_mogd_cd_all)
obj_hist_ngd_cd =  pd.DataFrame( obj_hist_msgd_cd_all)
obj_hist_mogd_leontief =  pd.DataFrame( obj_hist_mogd_leontief_all)
obj_hist_ngd_leontief =  pd.DataFrame(obj_hist_msgd_leontief_all)


obj_hist_mogd_linear.to_csv("data/obj/obj_hist_mogd_linear.csv")
obj_hist_ngd_linear.to_csv("data/obj/obj_hist_ngd_linear.csv")
obj_hist_mogd_cd.to_csv("data/obj/obj_hist_mogd_cd.csv")
obj_hist_ngd_cd.to_csv("data/obj/obj_hist_ngd_cd.csv")
obj_hist_mogd_leontief.to_csv("data/obj/obj_hist_mogd_leontief.csv")
obj_hist_ngd_leontief.to_csv("data/obj/obj_hist_ngd_leontief.csv")

prices_hist_mogd_linear_all = np.array(prices_hist_mogd_linear_all)
prices_hist_msgd_linear_all = np.array(prices_hist_msgd_linear_all)
prices_hist_mogd_cd_all = np.array(prices_hist_mogd_cd_all)
prices_hist_msgd_linear_all = np.array(prices_hist_msgd_linear_all)
prices_hist_mogd_leontief_all = np.array(prices_hist_mogd_leontief_all)
prices_hist_msgd_leontief_all = np.array(prices_hist_msgd_leontief_all)

prices_mogd_linear =  pd.DataFrame(prices_hist_mogd_linear_all)
prices_ngd_linear =  pd.DataFrame(prices_hist_msgd_linear_all)
prices_mogd_cd =  pd.DataFrame(prices_hist_mogd_cd_all)
prices_ngd_cd =  pd.DataFrame(prices_hist_msgd_linear_all )
prices_mogd_leontief =  pd.DataFrame( prices_hist_mogd_leontief_all)
prices_ngd_leontief =  pd.DataFrame( prices_hist_msgd_leontief_all )

prices_mogd_linear.to_csv("data/prices/prices_mogd_linear")
prices_ngd_linear.to_csv("data/prices/prices_ngd_linear")
prices_mogd_cd.to_csv("data/prices/prices_mogd_cd")
prices_ngd_cd.to_csv("data/prices/prices_ngd_cd")
prices_mogd_leontief.to_csv("data/prices/prices_mogd_leontief")
prices_ngd_leontief.to_csv("data/prices/prices_ngd_leontief")

#%%
obj_mogd_linear = np.mean(obj_hist_mogd_linear_all, axis = 0)
obj_msgd_linear = np.mean(obj_hist_msgd_linear_all, axis = 0)
obj_mogd_cd = np.mean(obj_hist_mogd_cd_all, axis = 0)
obj_msgd_cd = np.mean(obj_hist_msgd_cd_all, axis = 0)
obj_mogd_leontief = np.mean(obj_hist_mogd_leontief_all, axis = 0)
obj_msgd_leontief = np.mean(obj_hist_msgd_leontief_all, axis = 0)

fig, axs = plt.subplots(2, 2) # Create a figure containing a single axes.
axs[0,0].plot([iter for iter in range(len(obj_mogd_linear))], obj_mogd_linear, label = "Max-Oracle")
axs[0,0].plot([iter for iter in range(len(obj_msgd_linear))], obj_msgd_linear, label = "Nested Gradient Descent Ascent")
axs[0,0].set_title("Linear Market", fontsize = "medium")

axs[0,1].plot([iter for iter in range(len(obj_mogd_cd))], obj_mogd_cd, label = "Max-Oracle")
axs[0,1].plot([iter for iter in range(len(obj_msgd_cd))], obj_msgd_cd, label = "Nested Gradient Descent Ascent")
axs[0,1].set_title("Cobb-Douglas Market", fontsize = "medium")


axs[1,0].plot([iter for iter in range(len(obj_mogd_leontief))], obj_mogd_leontief, label = "Max-Oracle")
axs[1,0].plot([iter for iter in range(len(obj_msgd_leontief))], obj_msgd_leontief, label = "Nested Gradient Descent Ascent")
axs[1,0].set_title("Leontief Market", fontsize = "medium")
axs[1, 1].axis('off')
for ax in axs.flat:
    ax.set(xlabel='Iteration Number', ylabel='Objective Value')
for ax in axs.flat:
    ax.label_outer()

handles, labels = axs[1,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')

plt.show()

# plt.savefig("graphs/linear_mogd.png")
# %%
