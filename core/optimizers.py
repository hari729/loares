from multiprocessing import Pool, Manager
from tqdm import tqdm
from time import sleep

def optimizer(algo):
    while algo.tracker.remaining_evals() > 0:
        # print(algo.tracker.remaining_evals())
        algo.advance()
    return algo.get_result()


def multi_thread_optimizer_dict(algos_list, threads=5):
    with Pool(processes=threads) as pool:
        output = pool.map(optimizer, algos_list)

    return output

def optimizer_with_progress(args):
    algo, pid, que = args

    while algo.tracker.remaining_evals() > 0:
        algo.advance()
        remaining_evals = algo.tracker.remaining_evals()
        que.put((pid, remaining_evals))
    return algo.get_result()

def multi_thread_optimizer(algos_list, threads=5):

    manager = Manager()
    que = manager.Queue()

    total_evals = [algo.problem.max_evals for algo in algos_list]

    bars = []
    args = []
    for i, algo in enumerate(algos_list):
        bars.append(tqdm(total=algo.problem.max_evals,
                            position=i, 
                            desc=algo.get_info()["name"],
                         ascii=('','-')))

        args.append((algo, i, que))

    with Pool(processes=threads) as pool:
        output = pool.map_async(optimizer_with_progress, args)

        remaining_bars = len(algos_list)
        while remaining_bars > 0:
            while not que.empty():
                id, remaining = que.get()
                done = total_evals[id] - remaining
                bars[id].n = done
                bars[id].refresh()

            if output.ready():
                remaining_bars =0

            sleep(0.05)

        pool.close()
        pool.join()

    for b in bars:
        b.close()

    return output.get()
