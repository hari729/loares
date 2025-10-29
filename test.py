# from results.composite import composite
import numpy as np
from problems.robotics import gripper_c1

if __name__ == "__main__":
    
    population = np.array([[230.65628,182.67297,300,46.39774,43.90391,144.51126,2.03856],
                           [237.73246,187.10041,152.75955,33.30607,31.64811,232.63416,2.20343]])
    f,g = gripper_c1(population)
    print(f,g)
    # multi_objective_unconstrained(100,30,200000,30,"zdt1","e2","population")
    # print(len(np.arange(0,30,1)))

    # for i in range(10):
    #     print(i)

    # test_name = "plot_test"

    # timestamp = "20250712_185851"

    # list_of_algos = ["bmr","bwr"]

    # list_of_functions = ["zdt3"]

    # list_of_psizes = [500]

    # composite(test_name, timestamp, list_of_algos, list_of_functions, list_of_psizes)

    # composite_source = {}

    # composite_source["zdt1"] = dict(bmr = 150, bwr=200)
    # composite_source["zdt2"] = dict(bmr = 300, bwr=300)

    # list1,list2 = composite_source.items()
    # list3,list4 = list2.items()

    # print(list1,list2)
    # print(list3,list4)

    # t = np.array([[1,2,3],[1,2,3]])
    # print(np.mean(t, axis=1))

    # names = [f"dtlz{i}" for i in range(1,10)]
    # print(names)
