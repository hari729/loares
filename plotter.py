from metrics.plots import plot_mod
import numpy as np

if __name__ == "__main__":
    
    import case_studies.robotics as problem

    # function_name="drag_lift_inv"
    # psize=300
    # function,_,_,_,minmax,_ = problem.get[function_name]
    # pop = np.array([[50,30,3,40],[34.89,20,3,40],
    #                 [44.58,40,3,0],
    #                 [35.66,40,3,0],
    #                 [27.85,40,3,0],
    #                 [21.29,40,3,0],
    #                 [10.77,40,3,0]])
    # cpts,_ = minmax*function(pop)
    # xylabels = ["Drag","Lift"]

    # function_name = "mau"
    # function,_,_,_,_,_ = problem.get[function_name]
    # pop = np.array([[44.6174,9.0015,96.6784,6.1673]])
    # cpts,_ = function(pop)

    function_name = "auv_gep_safe"
    psize = 200
    cpts = np.array([[15.685,1/0.997],[20.436,1/1.085],[22.123,1/1.111]])
    xylabels = ["f1","f2"]

    test_name = "c3_gep_80000_20250908_094835"

    for algo_name in ["BMR","BWR","BMWR"]:
        csv_path = f"/home/hari/opti/v3.1/results/{test_name}/{algo_name}/{function_name.upper()}/{psize}/pareto_front.csv"
        legend = [f"MO-{algo_name}","Previous Optimum"]
        plot_mod(csv_path,legend,algo_name,cpts,xylabels)