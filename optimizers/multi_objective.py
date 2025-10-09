import numpy as np
import datetime
import sys
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

from customclass.states import PopulationState
from utils.initialization import random_initialize,lhs_initialize
import algorithms
import algo_modifiers.population as modifiers

from utils.sorting import ranking_crowding_general as sorting_function
from utils.sorting import ranking_reference 
from utils.selection import random_selection as selector
from metrics.performance import pindicators, gen_pindicators
from metrics.plots import generate_plots_notf

from pymoo.problems import get_problem

def single_run(args):

    algorithm,function,n_vars,bounds,tf,psize,max_evals,selector,selection_pool,mod,seed_id = args

    np.random.seed(seed_id)

    p = random_initialize(psize,n_vars,bounds)

    population_state = PopulationState(p, function, max_evals, sorting_function, selection_pool)

    population_state.update_generation()

    population_state.best, population_state.worst = selector(population_state,selection_pool,psize)

    while(population_state.evals < population_state.max_evals):
        
        population_state.add_solutions(algorithm(population_state, bounds))

        population_state.add_solutions(mod(population_state.population, bounds))

        population_state.evaluate()

        population_state.update_generation()

        population_state.best, population_state.worst = selector(population_state,selection_pool,psize)

    ref_pts = np.ones([population_state.objective_values.shape[1]]) * 1.1

    _,metrics = gen_pindicators(population_state.pareto_objectives,ref_pts,tf)

    return metrics,population_state

def multi_objective_optimizer(function,n_vars,bounds,minmax,list_of_algos,list_of_psizes,modifier_name,
                    selection_pool,max_evals,runs,tee_path,tf=None,std_seed=True):

    for psize in list_of_psizes:

        print(f"\nP:{psize}")
    
        for algo_name in list_of_algos:
            
            algorithm = algorithms.get[algo_name]
            mod = modifiers.get[modifier_name]

            file_path = f"{tee_path}/{algo_name.upper()}/{psize}"
            os.makedirs(file_path, exist_ok=True)

            if std_seed:
                seed_id = np.arange(0,runs,1)
            else:
                seed_id = np.random.randint(0,1e6,runs)
                np.savetxt(f"{file_path}/seeds.csv", seed_id, delimiter=",", fmt="%d", header="Seeds", comments="")

            args = [(algorithm,function,n_vars,bounds,tf,psize,max_evals,selector,selection_pool,mod,i) for i in seed_id]

            with Pool(processes=10) as pool:
                output = pool.map(single_run, args) 
            
            metrics, pop_states = zip(*output)
            metrics = np.array(metrics)
            mean_res = np.mean(metrics, axis=0)
            std = np.std(metrics, axis=0)

            pop = pop_states[np.argmax(metrics[:,-1])]
            solutions = pop.pareto_pop
            objective_values = minmax*pop.pareto_objectives
            constraint_values = pop.pareto_constraints 

            if metrics.shape[1] == 2:
                print(f"{algo_name.upper()}; F1: {np.min(objective_values[:,0]):.4e} to {np.max(objective_values[:,0]):.4e}, " 
                        f"F2: {np.min(objective_values[:,1]):.4e} to {np.max(objective_values[:,1]):.4e}, "
                        f"Spc: {mean_res[-2]:.4e} [{std[-1]:.4e}], HV: {mean_res[-1]:.4e} [{std[-1]:.4e}]")
            
            else:        
                print(f"{algo_name.upper()}; GD: {mean_res[0]:.4e} [{std[0]:.4e}], " 
                    f"IGD: {mean_res[1]:.4e} [{std[1]:.4e}], Spc: {mean_res[2]:.4e} [{std[2]:.4e}], " 
                    f"Spr: {mean_res[3]:.4e} [{std[3]:.4e}], HV: {mean_res[4]:.4e} [{std[4]:.4e}]")


            if constraint_values.ndim == 1:
                headers = ([f"s{i+1}" for i in range(solutions.shape[1])] + 
                            ["g1"] + [f"f{i+1}" for i in range(objective_values.shape[1])])
            else:
                headers = ([f"s{i+1}" for i in range(solutions.shape[1])] + 
                            [f"g{i+1}" for i in range(constraint_values.shape[1])] + [f"f{i+1}" for i in range(objective_values.shape[1])])

            np.savetxt(f"{file_path}/solutions.csv", np.column_stack([solutions, constraint_values, objective_values]) , delimiter=",", fmt="%.7e",
                    header=','.join(headers), comments="")

            legend = [f"MO-{algo_name.upper()} Pareto Front"]

            generate_plots_notf(function.__name__,algo_name,psize,max_evals,objective_values,legend,file_path,tf)