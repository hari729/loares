import numpy as np 

def random_reinit(problem, new, pb = 0.5):
    pop_size = new.shape[0]
    bounds = problem.bounds

    r3 = np.random.rand(pop_size, 1)
    r4 = np.random.rand(pop_size, 1)
    mask = r4 > pb

    random_init = bounds[None, :, 1] - r3  * (bounds[None, :, 1] - bounds[None,:, 0])
    new = np.where(mask, new, random_init)    
    new = np.clip(new,bounds[:,0],bounds[:,1])

    return new

def qopp_reinit(problem, new, pb = 0.5):
    pop_size = new.shape[0]
    bounds = problem.bounds

    r3 = np.random.rand(pop_size, 1)
    r4 = np.random.rand(pop_size, 1)
    mask = r4 > pb
    mid = np.mean(bounds, axis=1)
    opp = np.sum(bounds, axis=1) - new
    low = np.minimum(mid, opp)
    high = np.maximum(mid, opp)
    qopp = low + r3 * (high-low)
    new = np.where(mask, new, qopp)    
    new = np.clip(new,bounds[:,0],bounds[:,1])

    return new
