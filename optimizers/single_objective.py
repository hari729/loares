import numpy as np
import datetime
import sys
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool

from customclass.states import population
from utils.initialization import random_initialize,lhs_initialize
import algorithms
# import case_studies.robotics as problem
from utils.selection import simple_selection

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

def single_objective_optimizer(function,n_vars,bounds,list_of_algos,list_of_psizes,max_evals,runs,tee_path,minmax,plt_fe=True,std_seed=1,):

    for algo_name in list_of_algos:
            
            algorithm = algorithms.get[algo_name]

            print(f"\nAlgorithm: {algo_name.upper()}")
        
            for psize in list_of_psizes:

                iterations = max_evals//psize  

                if plt_fe:
                    function_evals = np.arange(1, iterations+1, 1) * psize
                    xlabel = "Function Evaluations"
                else:
                    function_evals = np.arange(1, iterations+1, 1)
                    xlabel = "Iterations"

                function_name = function.__name__

                file_path = f"{tee_path}/{algo_name}/{psize}"
                os.makedirs(file_path, exist_ok=True)

                if std_seed:
                    seed_id = np.arange(0, runs, 1)
                else:
                    seed_id = np.random.randint(1, 1e6, runs)
                    np.savetxt(f"{file_path}/seeds.csv", seed_id, delimiter=",", fmt="%d", header="Seeds", comments="")

                args = [(algorithm,function,n_vars,bounds,psize,iterations,i) for i in seed_id]

                with Pool(processes=10) as pool:
                    output = pool.map(single_run, args) 

                results, solutions, constraint_values, convergence_data = zip(*output)
                if minmax is None:
                    results = np.array(results)
                    mean_convergence_data = np.mean(np.stack(convergence_data), axis=0)
                else:
                    results = np.array(results) * minmax
                    mean_convergence_data = np.mean(np.stack(convergence_data), axis=0) * minmax
                solutions = np.stack(solutions)
                constraint_values = np.stack(constraint_values)
                

                conv_pt = np.empty([mean_convergence_data.shape[0]])
                conv_pt.fill(np.nan)

                conv_idx = np.argmax(mean_convergence_data == mean_convergence_data[-1])
                conv_pt[conv_idx] = mean_convergence_data[conv_idx]

                np.savetxt(f"{file_path}/results.csv", results, delimiter=",", fmt="%.7e", header="Best Value", comments="")
                np.savetxt(f"{file_path}/Mean_convergance_data.csv", np.column_stack([function_evals,mean_convergence_data,conv_pt]),
                            delimiter=",", fmt=("%d","%.7e","%.7e"), header="FEvals,Mean Best Values,Convergence point", comments="")

                if np.all(constraint_values != None):
                    if constraint_values.ndim == 1:
                        headers = ([f"s{i+1}" for i in range(solutions.shape[1])] + 
                                    ["g1"] + ["f"])
                    else:
                        headers = ([f"s{i+1}" for i in range(solutions.shape[1])] + 
                                    [f"g{i+1}" for i in range(constraint_values.shape[1])] + ["f"])

                    np.savetxt(f"{file_path}/solutions.csv", np.column_stack([solutions, constraint_values, results]) , delimiter=",", fmt="%.7e",
                            header=','.join(headers), comments="")
                else:
                    headers = [f"s{i+1}" for i in range(solutions.shape[1])] + ["f"]
                    np.savetxt(f"{file_path}/solutions.csv", np.column_stack([solutions, results]) , delimiter=",", fmt="%.7e",
                            header=','.join(headers), comments="")
                
                min_idx = np.argmin(results)

                print(f"\nP_size: {psize}, Iterations: {iterations}, Mean Result : {np.mean(results):.7e} [{np.std(results):.7e}]")
                print(f"Best result: {results[min_idx]:.7e} [R{min_idx}],  Worst result: {np.max(results):.7e}")
                print(f"Solution: {solutions[min_idx]}")
                if np.any(constraint_values != None):
                    print(f"Constraint values: {constraint_values[min_idx]}")
                print(f"MFE :{function_evals[conv_idx]}")          

                lg = [f"{algo_name.upper()}","Convergence Point"]
                plt.figure()
                plt.plot(function_evals, mean_convergence_data,linestyle='-',color='green',alpha=1)
                plt.plot(function_evals, conv_pt,linestyle='',marker = 'x',color='red',alpha=1)
                plt.legend(labels=lg, loc='right', fontsize=8)
                plt.grid(which='both',linestyle='--',alpha=0.7)
                plt.xlabel(xlabel)
                if runs == 1:
                    plt.ylabel("Best value")
                else:
                    plt.ylabel("Mean Best value")
                plt.tight_layout()
                plt.savefig(f"{file_path}/{function_name}_{algo_name}_{psize}_{iterations}.png", dpi=600, bbox_inches='tight')
                # plt.show()
                plt.close()