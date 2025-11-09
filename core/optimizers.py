
from multiprocessing import Pool

def optimizer(algo):
    while algo.tracker.remaining_evals() > 0:
        algo.advance()
    return algo.get_result()

def multi_thread_optimizer(algo_class, args, seed_list, threads=5):
    settings = [{"seed": s, **args} for s in seed_list]
    algos = [algo_class(**i) for i in settings]
    with Pool(processes=threads) as pool:
        output = pool.map(optimizer, algos)

    return output
