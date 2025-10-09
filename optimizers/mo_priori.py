import numpy as np
import datetime
import sys
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools

from population.population import population
from utils.initialization import random_initialize,lhs_initialize
import algorithms
# import case_studies.robotics as problem
from utils.selection import simple_selection
from sys_utils.logger import Tee_buffer as Tee
from metrics.plots import generate_plots_notf


np.set_printoptions(linewidth=np.inf)

def single_run(args):
    algorithm,function,n_vars,bounds,psize,iterations,seed_id = args

    np.random.seed(seed_id)

    p = random_initialize(psize,n_vars,bounds)
    pop = population(p)
    pop.objective_values, pop.constraint_values = function(pop.population)
    
    pop.best_idx = np.argmin(pop.objective_values)
    pop.wrost_idx = np.argmax(pop.objective_values)
    pop.best = pop.population[pop.best_idx]
    pop.worst = pop.population[pop.wrost_idx]

    pop.convergence_data = np.zeros([iterations,1]) 
    pop.convergence_data[0] = pop.objective_values[pop.best_idx]

    for i in range(iterations-1):
        new_p = algorithm(pop,bounds)
        new_ob, new_constr = function(new_p)

        compare = new_ob < pop.objective_values
        pop.population[compare] = new_p[compare]
        pop.objective_values[compare] = new_ob[compare]
        if pop.constraint_values is not None:
            pop.constraint_values[compare] = new_constr[compare]

        pop.best_idx = np.argmin(pop.objective_values)
        pop.wrost_idx = np.argmax(pop.objective_values)
        pop.best = pop.population[pop.best_idx]
        pop.worst = pop.population[pop.wrost_idx]

        pop.convergence_data[i+1] = pop.objective_values[pop.best_idx]
    if pop.constraint_values is not None:
        return pop.objective_values[pop.best_idx],pop.best,pop.constraint_values[pop.best_idx],pop.convergence_data
    else:
        return pop.objective_values[pop.best_idx],pop.best,None,pop.convergence_data

def generate_weights(n_obj, step=0.1):
    """Generate weight vectors for n_obj that sum to 1."""
    grid = np.arange(0, 1+step, step)
    weights = []
    for combo in itertools.product(grid, repeat=n_obj):
        if abs(sum(combo) - 1.0) < 1e-6:
            weights.append(combo)
    weights = np.array(weights)
    # Build sorting keys dynamically (negative for descending order)
    sort_keys = tuple(-weights[:, i] for i in range(n_obj))
    sort_idx = np.lexsort(sort_keys[::-1])  # reverse because lexsort uses last key as primary
    weights = weights[sort_idx]

    return weights

def a_priori_optimizer(function,n_vars,bounds,algo_name,psize,iterations,runs,file_path,std_seed=1):
            
    algorithm = algorithms.get[algo_name] 

    function_evals = np.arange(1, iterations+1, 1) * psize

    if std_seed:
        seed_id = np.arange(0, runs, 1)
    else:
        seed_id = np.random.randint(1, 1e6, runs)
        np.savetxt(f"{file_path}/seeds.csv", seed_id, delimiter=",", fmt="%d", header="Seeds", comments="")

    args = [(algorithm,function,n_vars,bounds,psize,iterations,i) for i in seed_id]
    output = [single_run(a) for a in args]

    results, solutions, constraint_values, convergence_data = zip(*output)
    results = np.array(results)
    solutions = np.stack(solutions)
    constraint_values = np.stack(constraint_values)
    mean_convergence_data = np.mean(np.stack(convergence_data), axis=0)

    return results,solutions,constraint_values

def a_priori(function,n_vars,bounds,n_obj,list_of_algos,list_of_psizes,max_evals,runs,tee_path):

    for algo_name in list_of_algos:

        legend = [f"{algo_name.upper()}"]

        print(f"\nAlgorithm: {algo_name.upper()}")

        for psize in list_of_psizes:
            iterations = max_evals//psize 
            print(f"\nP: {psize}, I: {iterations}")
            file_path = f"{tee_path}/{function.__name__.upper()}/{algo_name.upper()}/{psize}"
            os.makedirs(file_path, exist_ok=True)

            so_min = np.zeros([1,n_obj])
            so_sols = np.zeros([n_obj,n_vars])
            for i in range(n_obj):
                # print(i)
                def so_prob(pop):
                    f,g = function(pop)
                    violations = g > 0
                    penalties = np.where(violations, g**2, 0) 
                    return f[:,i]+np.sum(penalties,axis=1),g

                so_min[0,i],so_sols[i,:],_ = a_priori_optimizer(so_prob,n_vars,bounds,algo_name,psize,iterations,runs,file_path)

            print(f"\nso_solutions: \n{so_sols}")
            print(f"so_min:{so_min}")
            weights = generate_weights(n_obj,0.1)
            print(f"\nWeights: \n{weights}")
            # pareto_front = np.zeros([weights.shape[0],2])
            constraints = []
            solution = []
            
            for w in range(len(weights)):
                def prob(pop):
                    f,g = function(pop)
                    obj = (f[:,0]*weights[w,0]/so_min[:,0]
                            + f[:,1]*(weights[w,1])/so_min[:,1])
                    violations = g > 0
                    penalties = np.where(violations, g**2, 0)
                    return obj+np.sum(penalties,axis=1),g

                _,sol,cons = a_priori_optimizer(prob,n_vars,bounds,algo_name,psize,iterations,runs,file_path)
                
                constraints.append(cons)
                solution.append(sol)

            solutions = np.row_stack(solution)
            objective_values,_ = function(solutions)

            print(f"\nSolutions:\n{solutions}")
            print(f"Objectives:\n{objective_values}")

            headers = [f"f{i+1}" for i in range(objective_values.shape[1])]
            np.savetxt(f"{file_path}/pareto_front.csv", objective_values, delimiter=",", fmt="%.7e",
                            header=','.join(headers), comments="")
            
            generate_plots_notf(function.__name__,algo_name,psize,iterations,objective_values[:,:n_obj],legend,file_path)


    
    

