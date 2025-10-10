import datetime
import sys
import os
import numpy as np
from pathlib import Path
import argparse
import shutil

import problems
from optimizers.single_objective import single_objective_optimizer
from optimizers.multi_objective import multi_objective_optimizer
# from optimizers.mo_priori import a_priori
from sys_utils.logger import Tee_general as Tee

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run multi-objective optimization experiments.")
    parser.add_argument("--test_name", "-t", required=True, help="The name of the test.")
    parser.add_argument("--problem", "-p", type=str, required=True, help="<category>.<prob_name>")
    parser.add_argument("--selection-pool", "-sp", type=str, choices=["population","archive"], default="population",
                            help="Selection pool for best soltution, population or archive")

    args = parser.parse_args()
    test_name = args.test_name
    list_of_functions = [args.problem]
    selection_pool = args.selection_pool

    list_of_algos = ["bmr","bwr","bmwr"]
    list_of_psizes = []  # add psizes other than default here
    runs = 1
    modifier_name = "opposition"
    a_posterior = 1

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    project_root = Path(__file__).parent.resolve()
    final_path = project_root/f"results/{test_name}_{timestamp}" 
    temp_path = project_root/f"results/data_dump/{test_name}_{timestamp}" 
    os.makedirs(temp_path, exist_ok=True)
    tee = Tee(f"{temp_path}/{timestamp}.txt")
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(f"\nTest Name: {test_name}")
        print(f"\nRuns: {runs}")
        print(f"\nSelection Pool: {selection_pool}")


        for function_name in list_of_functions:

            temp_file_path = project_root/f"results/data_dump/{test_name}_{timestamp}/{function_name}" 
            os.makedirs(temp_file_path, exist_ok=True)

            function, n_vars, bounds, n_obj, minmax, max_evals, def_psize = problems.get(function_name)

            # max_evals = 40000 # override case definition
            list_of_psizes.append(def_psize)

            _, idx = np.unique(bounds, axis=0, return_index=True)
            ubounds = bounds[np.sort(idx)] 
            bounds_str = "[" + " , ".join(f"[{b[0]}, {b[1]}]" for b in ubounds) + "]"

            print(f"\nFunction: {function_name.upper()}, Variables: {n_vars}, Bounds: {bounds_str}")
            print(f"\nMax_Evals: {max_evals}")

            if n_obj > 1:
                tf = problems.get_true_front(function_name, n_vars)
                if a_posterior:
                    multi_objective_optimizer(function,n_vars,bounds,minmax,
                                                list_of_algos,list_of_psizes,modifier_name,
                                                selection_pool,max_evals,
                                                runs,temp_file_path,tf)
                else:
                    a_priori(function,n_vars,bounds,n_obj,list_of_algos,list_of_psizes,max_evals,runs,file_path)
            else:
                single_objective_optimizer(function,n_vars,bounds,list_of_algos,list_of_psizes,max_evals,runs,
                                            temp_file_path,minmax,plt_fe=False)

            list_of_psizes.pop()

        shutil.move(str(temp_path), str(final_path))
        print(f"\nResults saved successfully to: {final_path}")

    except KeyboardInterrupt:
        print("\nRun interrupted by user (Ctrl+C).")
        print(f"Temporary results kept in: {temp_path}\n")

    except Exception as e:
        import traceback
        print(f"\nRun failed — temporary results kept in: {temp_path}\n")
        traceback.print_exc()         