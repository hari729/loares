import numpy as np
import datetime
import sys
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import json

from customclass.states import PopulationState
from utils.initialization import random_initialize,lhs_initialize
import algorithms
import modifiers.population

from utils.sorting import ranking_crowding_general as sorting_function
from utils.sorting import ranking_reference 
from utils.selection import random_selection as selector
from metrics.performance import pindicators, gen_pindicators
from metrics.plots import generate_plots_notf, convergence_plots

from pymoo.problems import get_problem

def single_run(args):

    algorithm,function,n_vars,bounds,tf,psize,max_evals,selector,selection_pool,pmods,seed_id = args

    np.random.seed(seed_id)

    interval_size = int(max_evals * 0.05)

    p = random_initialize(psize,n_vars,bounds)

    population_state = PopulationState(p, function, max_evals, sorting_function, selection_pool)

    population_state.update_generation()

    ref_pts = np.ones([population_state.objective_values.shape[1]]) * 1.1

    population_state.add_convergance_data(gen_pindicators(population_state.pareto_objectives,ref_pts,tf))

    population_state.best, population_state.worst = selector(population_state,selection_pool,psize)

    while(population_state.evals < population_state.max_evals):
        
        prev_evals = population_state.evals

        population_state.add_solutions(algorithm(population_state, bounds))
        
        for mod in pmods:
            population_state.add_solutions(mod(population_state, bounds))

        population_state.evaluate()

        population_state.update_generation()

        if (population_state.evals // interval_size) > (prev_evals // interval_size):
            population_state.add_convergance_data(gen_pindicators(population_state.pareto_objectives,ref_pts,tf))

        population_state.best, population_state.worst = selector(population_state,selection_pool,psize)


    metrics = gen_pindicators(population_state.pareto_objectives,ref_pts,tf)

    population_state.add_convergance_data(metrics)

    return metrics,population_state

def multi_objective_optimizer(function,n_vars,bounds,minmax,list_of_algos,list_of_psizes,pmodifier_list,
                    selection_pool,max_evals,runs,tee_path,threads,tf=None,std_seed=True):

    for psize in list_of_psizes:

        print(f"\nP:{psize}")
    
        for algo_name in list_of_algos:

            
            algorithm = algorithms.get[algo_name]
            pmods = [modifiers.population.get[name] for name in pmodifier_list]

            file_path = f"{tee_path}/{algo_name.upper()}/{psize}"
            os.makedirs(file_path, exist_ok=True)

            if std_seed:
                seed_id = np.arange(0,runs,1)
            else:
                seed_id = np.random.randint(0,1e6,runs)
                np.savetxt(f"{file_path}/seeds.csv", seed_id, delimiter=",", fmt="%d", header="Seeds", comments="")

            args = [(algorithm,function,n_vars,bounds,tf,psize,max_evals,selector,selection_pool,pmods,i) for i in seed_id]

            with Pool(processes=threads) as pool:
                output = pool.map(single_run, args) 
            
            metrics, pop_states = zip(*output)
            metrics = np.array(metrics)
            mean_res = np.mean(metrics, axis=0)
            std = np.std(metrics, axis=0)

            pop = pop_states[np.argmax(metrics[:,-1])]
            best_seed = seed_id[np.argmax(metrics[:,-1])]
            solutions = pop.pareto_pop
            objective_values = minmax*pop.pareto_objectives
            constraint_values = pop.pareto_constraints
            convergence_data = pop.get_convergence_data()

            if metrics.shape[1] == 2:
                print(f"{algo_name.upper()}; F1: {np.min(objective_values[:,0]):.4e} to {np.max(objective_values[:,0]):.4e}, " 
                        f"F2: {np.min(objective_values[:,1]):.4e} to {np.max(objective_values[:,1]):.4e}, "
                        f"Spc: {mean_res[-2]:.4e} [{std[-1]:.4e}], HV: {mean_res[-1]:.4e} [{std[-1]:.4e}]")
                convergence_headers = ["SPC","HV","evals"]

            else:        
                print(f"{algo_name.upper()}; GD: {mean_res[0]:.4e} [{std[0]:.4e}], " 
                    f"IGD: {mean_res[1]:.4e} [{std[1]:.4e}], Spc: {mean_res[2]:.4e} [{std[2]:.4e}], " 
                    f"Spr: {mean_res[3]:.4e} [{std[3]:.4e}], HV: {mean_res[4]:.4e} [{std[4]:.4e}]")

                convergence_headers = ["GD","IGD","SPC","SPR","HV","evals"]

            if constraint_values.ndim == 1:
                headers = ([f"s{i+1}" for i in range(solutions.shape[1])] + 
                            ["g1"] + [f"f{i+1}" for i in range(objective_values.shape[1])])
            else:
                headers = ([f"s{i+1}" for i in range(solutions.shape[1])] + 
                            [f"g{i+1}" for i in range(constraint_values.shape[1])] + [f"f{i+1}" for i in range(objective_values.shape[1])])
            
            pareto_data =  np.column_stack([solutions, constraint_values, objective_values])

            np.savetxt(f"{file_path}/solutions.csv", pareto_data , delimiter=",", fmt="%.7e",
                    header=','.join(headers), comments="")
            
            np.savetxt(f"{file_path}/convergence_data.csv", convergence_data , delimiter=",", fmt="%.7e",
                    header=','.join(convergence_headers), comments="")

            legend = [f"MO-{algo_name.upper()} Pareto Front"]

            generate_plots_notf(function.__name__,algo_name,psize,max_evals,objective_values,legend,file_path,tf)
            convergence_plots(function.__name__,algo_name,psize,max_evals,convergence_data,file_path)


            settings = {
                "problem": function.__name__,
                "n_vars": int(n_vars),
                "bounds": "[" + ",".join(f"[{b[0]},{b[1]}]" for b in bounds) + "]",
                "minmax": minmax.tolist(),
                "algo_name": algo_name,
                "psize": int(psize),
                "max_evals": int(max_evals),
                "modifiers": " + ".join(f"{p}" for p in pmodifier_list),
                "selection_pool": selection_pool,
                "sorting": sorting_function.__name__,
                "selector": selector.__name__,
                "runs": int(runs),
                "seeds": "[" + ",".join(f"{i}" for i in seed_id) + "]",
                "best_seed": int(best_seed),
                "metrics": {
                        "headers": convergence_headers,
                        "mean":  mean_res.tolist(),
                        "std": std.tolist(),
                }
            }

            with open(f"{file_path}/settings.json", "w") as f:
                json.dump(settings, f, indent=4)

            convergence_dict = {c_header: convergence_data[:,i] for i,c_header in enumerate(convergence_headers)}
            np.savez_compressed(
                f"{file_path}/convergence_data.npz",
                **convergence_dict
            )
           
            pareto_dict = {header: pareto_data[:,j] for j,header in enumerate(headers)}
            np.savez_compressed(
                f"{file_path}/pareto_data.npz",
                **pareto_dict
            )
