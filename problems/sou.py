import numpy as np

def sphere(pop):
    pop = np.atleast_2d(pop)
    return np.sum(pop**2,axis=1), None

def matyas(pop):
    pop = np.atleast_2d(pop)
    return 0.26*(sphere(pop))-0.48*(np.prod(pop,axis=1)), None

def sumsquares(pop):
    pop = np.atleast_2d(pop)
    i_vals = np.arange(1,pop.shape[1]+1)
    return np.sum(i_vals * pop**2,axis=1), None

def beale(pop):
    pop = np.atleast_2d(pop)
    t1 = (1.5 - pop[:,0] + pop[:,0]*pop[:,1])**2
    t2 = (2.25 - pop[:,0] + pop[:,0]*pop[:,1]**2)**2
    t3 = (2.625 - pop[:,0] + pop[:,0]*pop[:,1]**3)**2
    return t1+t2+t3, None
    
def easom(pop):
    pop = np.atleast_2d(pop)
    e_term = np.exp(-(pop[:,0]-np.pi)**2 - (pop[:,1]-np.pi)**2)
    return -np.cos(pop[:,0])*np.cos(pop[:,1])*e_term, None

def branin(pop):
    pop = np.atleast_2d(pop)
    t1 = (pop[:,1] - ((5.1/(4*np.pi**2))*pop[:,0]**2) + 5*(pop[:,0]/np.pi) - 6)**2
    t2 = 10*(1 - (1/(8*np.pi)))*np.cos(pop[:,0]) + 10
    return t1+t2, None

def colville(pop):
    x1,x2,x3,x4 = pop.T
    f = (
        100*(x1**2 - x2)**2 + (x1 - 1)**2 + (x3 - 1)**2 + 90*(x3**2 - x4)**2 
        + 10.1*((x2-1)**2 + (x4-1)**2) + 19.8*(x2-1)*(x4-1)
    )

    return f, None

def trid(pop):

    f = np.sum((pop-1)**2, axis=1) - np.sum(pop[:,1:]*pop[:,:-1], axis=1)

    return f, None

def zakharov(pop):
    i_vals = np.arange(1,pop.shape[1]+1)
    f = np.sum(pop**2, axis=1) + (np.sum(0.5*i_vals*pop, axis=1))**2 + (np.sum(0.5*i_vals*pop, axis=1))**4
    return f, None

def schwefel_1_2(pop):
    f = np.sum((np.cumsum(pop, axis=1))**2,axis=1)
    return f, None

def rosenbrock(pop):
    f = np.sum(100* (pop[:,1:] - pop[:,:-1]**2)**2 + (pop[:,:-1] - 1)**2, axis=1)
    return f, None

def dixon_price(pop):
    i_vals = np.arange(2,pop.shape[1]+1)
    f = (pop[:,0] - 1)**2 + np.sum(i_vals*(2*pop[:,1:]**2 - pop[:,:-1])**2, axis=1)
    return f, None

def bohachevsky1(pop):
    f = pop[:,0]**2 + 2*pop[:,1]**2 - 0.3*np.cos(3*np.pi*pop[:,0]) - 0.4*np.cos(4*np.pi*pop[:,1]) + 0.7
    return f,None

def bohachevsky2(pop):
    f = pop[:,0]**2 + 2*pop[:,1]**2 - 0.3*np.cos(3*np.pi*pop[:,0])*np.cos(4*np.pi*pop[:,1]) + 0.3
    return f,None

def bohachevsky3(pop):
    f = pop[:,0]**2 + 2*pop[:,1]**2 - 0.3*np.cos(3*np.pi*pop[:,0] + 4*np.pi*pop[:,1]) + 0.3
    return f,None  

def booth(pop):
    f = (pop[:,0] + 2*pop[:,1] -7)**2 + (2*pop[:,0] + pop[:,1] - 5)**2
    return f, None

def michalewicz(pop):
    i_vals = np.arange(1, pop.shape[1]+1)
    f = -np.sum(np.sin(pop)*(np.sin(i_vals*pop**2/np.pi))**20, axis=1)
    return f, None

def goldstein_price(pop):
    x1,x2 = pop.T
    f = ( (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))*
            (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)) )
    return f, None
    
def perm(pop, beta=0.5):
    d = pop.shape[1]
    i = np.arange(1, d+1).reshape(-1, 1)  # (d, 1)
    j = np.arange(1, d+1).reshape(1, -1)  # (1, d)

    # pop shape: (N, d)
    x_scaled = pop[:, None, :] / j  # shape: (N, 1, d) / (1, d) → (N, 1, d)
    inner = (j**i + beta) * ((x_scaled)**i - 1)  # shape: (N, d, d)
    f = np.sum(np.sum(inner, axis=2)**2, axis=1)  # sum over j, then over i
    return f, None

def ackley(pop):
    D = pop.shape[1]
    f = (   -20*np.exp(-0.2*np.sqrt(np.sum(pop**2, axis=1)/D)) 
            - np.exp((1/D)*np.sum(np.cos(2*np.pi*pop), axis=1)) + 20 + np.e )
    return f, None

get = {
    #"function_name":[function, no_variables, bounds]
    "sphere": [sphere, 30, np.array([[-100,100]]*30),1],
    "sumsquares": [sumsquares, 30, np.array([[-10,10]]*30),1],
    "beale" : [beale,2,np.array([[-4.5,4.5]]*2),1],
    "easom" : [easom,2,np.array([[-100,100]]*2),1],
    "matyas": [matyas, 2, np.array([[-10,10]]*2),1],
    "colville":[colville, 4, np.array([[-10,10]]*4),1],
    "trid6":[trid, 6, np.array([[-36,36]]*6),1],
    "trid10":[trid, 10, np.array([[-100,100]]*10),1],
    "zakharov":[zakharov, 10, np.array([[-5,10]]*10),1],
    "schwefel_1.2":[schwefel_1_2, 30, np.array([[-100,100]]*30),1],
    "rosenbrock":[rosenbrock, 30, np.array([[-5,10]]*30),1],
    "dixon-price":[dixon_price, 30, np.array([[-10,10]]*30),1],
    "branin": [branin,2,np.array([[-5,10],[0,15]]),1],
    "bohachevsky1":[bohachevsky1, 2, np.array([[-100,100]]*2),1],
    "bohachevsky2":[bohachevsky2, 2, np.array([[-100,100]]*2),1],
    "bohachevsky3":[bohachevsky3, 2, np.array([[-100,100]]*2),1],
    "booth":[booth, 2, np.array([[-10,10]]*2),1],
    "michalewicz2":[michalewicz, 2, np.array([[0,np.pi]]*2),1],
    "michalewicz5":[michalewicz, 5, np.array([[0,np.pi]]*5),1],
    "goldstein_price":[goldstein_price, 2, np.array([[-2,2]]*2),1],
    "perm": [perm, 4, np.array([[-4,4]]*4),1],
    "ackley":[ackley, 30, np.array([[-32,32]]*30),1],
}