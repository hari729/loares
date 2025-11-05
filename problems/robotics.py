import numpy as np
import scipy as sp
from optimizers.single_objective import single_run as so_opti
import algorithms

def mau(population):
    b,h,l,s = population.T
    f = np.zeros([population.shape[0],2])
    g = np.zeros([population.shape[0],1])

    f[:,0] = (6.4852 + 0.036*l + 0.023*s*s + 0.0025*b*h
                - 0.0007*b*l + 0.0007*h*l - 0.014*h*s
                - 0.0015*l*s)
    f[:,1] = 7.86e-3 * (b - 7) * (h - 3) * (l - 7)

    F = (-63.446 - 1.4887*(h-3)**2 + 1.1434*(b-7)*(h-3)
            + 0.0911*(b-7)*(l-7) + 0.3495*(h-3)*(l-7))
    g[:,0] = 630 - F

    return f,g

def drag_lift(population):
    x1,x2,x3,x4 = population.T
    x3 = np.rint(x3)
    # x2 = np.round(x2/10) * 10
    # x4 = np.round(x4/10) * 10
    lift = (-5.57529 + 0.12894*x1 + 0.21604*x2
            + 0.93850*x3 - 0.07689*x4 - 0.00011*x1**2
            - 0.00353*x2**2 - 0.07071*x3**2 + 0.00243*x4**2
            - 0.00018*x1*x2 + 0.01703*x1*x3 + 0.00295*x1*x4
            + 0.00356*x2*x3 + 0.0003*x2*x4 - 0.01148*x3*x4)
    drag = (-0.00745 + 8.10185*x1 + 0.00045*x2
            + 0.00124*x3 - 4.27546e-5 *x4 + 1.48611e-6 *x1**2
            - 5.7778e-6 *x2**2 + 0.0003*x3**2 + 1.25e-7 *x4**2
            + 1.52778e-6 *x1*x2 - 1.09722e-5 *x1*x3 + 2.43751*x1*x4
            - 6.94444e-5 *x2*x3 + 1.55556e-6 *x2*x4 - 5.51667e-6 *x3*x4)

    return np.column_stack([drag,-lift]), np.column_stack([-lift,-drag])

def drag_lift_inv(population):
    x1,x2,x3,x4 = population.T
    x3 = np.rint(x3)
    # x2 = np.round(x2/10) * 10
    # x4 = np.round(x4/10) * 10
    lift = (-5.57529 + 0.12894*x1 + 0.21604*x2
            + 0.93850*x3 - 0.07689*x4 - 0.00011*x1**2
            - 0.00353*x2**2 - 0.07071*x3**2 + 0.00243*x4**2
            - 0.00018*x1*x2 + 0.01703*x1*x3 + 0.00295*x1*x4
            + 0.00356*x2*x3 + 0.0003*x2*x4 - 0.01148*x3*x4)
    drag = (-0.00745 + 8.10185*x1 + 0.00045*x2
            + 0.00124*x3 - 4.27546e-5 *x4 + 1.48611e-6 *x1**2
            - 5.7778e-6 *x2**2 + 0.0003*x3**2 + 1.25e-7 *x4**2
            + 1.52778e-6 *x1*x2 - 1.09722e-5 *x1*x3 + 2.43751*x1*x4
            - 6.94444e-5 *x2*x3 + 1.55556e-6 *x2*x4 - 5.51667e-6 *x3*x4)

    return np.column_stack([-1/drag,1/lift]), np.column_stack([-lift,-drag])

def auv(population):
    x1,x2,x3,x4 = population.T
    f = (35.590 + -76.736*x1 -35.031*x2 - 1.303*x3 -2.227*x4
        + 198.888*x1**2 - 0.007*x1*x2 + 0.148*x1*x3 - 0.005*x1*x4
        + 30.732*x2**2 + 7.911*x2*x3 - 0.655*x2*x4
        - 0.183*x3**2 + 0.127*x3*x4 + 0.654*x4**2)
    g = (1.088 - 0.794*x1 - 0.903*x2 + 0.058*x3 + 0.04*x4
        + 0.302*x1**2 - 0.007*x1*x2 + 0.148*x1*x3 - 0.005*x1*x4
        + 0.09*x2**2 - 0.004*x2*x3 + 0.154*x2*x4
        - 0.010*x3**2 + 0.0002*x3*x4 - 0.009*x4**2)
    return np.column_stack([f,1/g]), np.full((population.shape[0], 1), -1)

# def auv_gep(population):
#     x1,x2,x3,x4 = population.T
#     f = (10**(np.arctan(np.arccos(np.maximum((1.6126-x4)*1.8852, np.tan(x1)))))
#             + np.cos(x1*2.0651)*x3 + 1/x2 + np.maximum(x4,-3.0430)
#             + np.arctan(np.log(5.4558 - np.tan(np.tan(8.4771 + x4)))**2))
#     g = (np.cbrt(np.exp(np.cbrt(np.cos(x3)*(1/-9.5433 - x2)))**2)
#             + np.cbrt(np.minimum(np.arctan(1/(8.6543-x4-((x3+x1)/2))),np.arctan(1/(x3+6.6160))))
#             + np.cbrt(np.arcsin(np.maximum(0.1969,x1)*0.1174**2 - np.minimum((x1+x2)/2,x2))))
#     return np.column_stack([f,1/g]), np.full((population.shape[0], 1), -1)

BIG = 1e6  # penalty constant

def safe_log(x):
    out = np.empty_like(x)
    mask = x > 0
    out[mask] = np.log(x[mask])
    out[~mask] = BIG  # invalid -> huge -> arctan(BIG) ~ π/2
    return out

def safe_arccos(x):
    out = np.empty_like(x)
    mask = (x >= -1) & (x <= 1)
    out[mask] = np.arccos(x[mask])
    out[~mask] = np.pi  # invalid -> max penalty
    return out

def safe_arcsin(x):
    out = np.empty_like(x)
    mask = (x >= -1) & (x <= 1)
    out[mask] = np.arcsin(x[mask])
    out[~mask] = 0.0  # invalid -> small g -> large 1/g
    return out

def auv_gep_safe(population):
    x1, x2, x3, x4 = population.T
    
    # Resistance surrogate
    f = (10**(np.arctan(safe_arccos(np.maximum((1.6126-x4)*1.8852, np.tan(x1)))))
         + np.cos(x1*2.0651)*x3
         + 1/x2
         + np.maximum(x4, -3.0430)
         + np.arctan(safe_log(5.4558 - np.tan(np.tan(8.4771 + x4)))**2))
    
    # Volume surrogate
    v = (np.cbrt(np.exp(np.cbrt(np.cos(x3)*(1/-9.5433 - x2)))**2)
         + np.cbrt(np.minimum(np.arctan(1/(8.6543-x4-((x3+x1)/2))),
                              np.arctan(1/(x3+6.6160))))
         + np.cbrt(safe_arcsin(np.maximum(0.1969, x1)*0.1174**2
                               - np.minimum((x1+x2)/2, x2))))
    
    # Clamp g to avoid division by zero
    v = np.clip(v, 1e-8, None)
    
    return np.column_stack([f, 1/v]), np.full((population.shape[0], 1), -1)

def vg_ft(population):
    a,b,c = population.T
    f = (9.40745e-1 - 4.29e-4*a + 4.8329e-2*b
        - 4.62393e-1*c - 1.63333e-5*a*b - 3.034e-4*a*c
        - 2.75488e-1*b*c - 4.36364e-8*a**2 
        + 8.291e-3*b**2 - 6.27706*c**2)
    t = (19.32717 + 9.702e-3*a - 7.70195*b
        + 112.72384*c + 3.3e-5*a*b - 1.0036e-2*a*c
        + 12.80639*b*c - 5.85859e-7*a**2 + 4.29091e-1*b**2 - 187.77056*c**2)
    return np.column_stack([f,t]), np.full((population.shape[0], 1), -1)

def gripper_c1(population):

    P = 100
    Y_min = 50
    Y_max = 100
    Yg = 150
    Z_max = 50
    Fg = 50
    psize = population.shape[0]
    f_array = np.zeros([psize,2])
    g = np.zeros([psize,8])
    Fk_array = np.zeros([psize,2])

    
    def y(population,z):      
        a,b,c,e,f,l,delta = population.T
        g = np.sqrt((l-z)**2 + e**2)
        phi = np.arctan2(e , (l-z))
        num1 = np.clip((a**2 + g**2 - b**2)/(2*a*g), -1, 1)
        num2 = np.clip((b**2 + g**2 - a**2)/(2*b*g), -1, 1)
        alpha = np.arccos(num1) + phi
        beta  = np.arccos(num2) - phi
        return 2*(e+f+c*np.sin(beta+delta))

    yZmax = y(population,Z_max)
    y0 = y(population,0)

    for i in range(0,population.shape[0]):

        a,b,c,e,f,l,delta = population[i,:].T
        
        def Fk(z):
            g = np.sqrt((l-z)**2 + e**2)
            phi = np.arctan(e / (l-z))
            num1 = np.clip((a**2 + g**2 - b**2)/(2*a*g), -1, 1)
            num2 = np.clip((b**2 + g**2 - a**2)/(2*b*g), -1, 1)
            alpha = np.arccos(num1) + phi
            beta  = np.arccos(num2) - phi
            f_i = P*b*np.sin(alpha + beta)/(2*c*np.cos(alpha))
            return f_i, None

        def Fk_max(z):
            f_i,_ = Fk(z)
            return -f_i, None

        algo = algorithms.get['bmr']
        res = so_opti([algo,Fk_max,1,np.array([[0,50]]),10,10,1])
        Fk_array[i,0] = - res[0]
        Fk_array[i,1],_,_,_ = so_opti([algo,Fk,1,np.array([[0,50]]),10,10,1])
    
       
    f_array[:,0] = Fk_array[:,0] - Fk_array[:,1]
    f_array[:,1] = P/(Fk_array[:,1])

    ai,bi,ci,ei,fi,li,deltai = population.T
    g[:,0] = Y_min - yZmax
    g[:,1] = yZmax
    g[:,2] = y0 - Y_max
    g[:,3] = Yg - y0
    g[:,4] = (ai + bi)**2 - li**2 - ei**2
    g[:,5] = (li - Z_max)**2 + (ai - ei)**2 - bi**2
    g[:,6] = li - Z_max
    g[:,7] = Fk_array[:,1] - Fg
    # g[:,7] =  1
    # print(f_array,g) 
    # print(Fk_array)
    # return f_array,np.full((psize, 1), -1)
    # print(g)
    return f_array, -g


get = {
    #"name":[function, n_vars, bounds, n_obj, minmax, max_evals, population_size]

    "mau": [mau, 4, np.array([[24.5,45.5],[9,17],[60,110],[0,15]]),2,np.array([1,1]),30000,100],
    "drag_lift": [drag_lift, 4, np.array([[10,50],[20,40],[1,3],[0,40]]),2,np.array([1,-1]),60000,300],
    "drag_lift_inv": [drag_lift_inv, 4, np.array([[10,50],[20,40],[1,3],[0,40]]),2,np.array([-1,1]),60000,300],
    "auv": [auv, 4, np.array([[0.148,0.223],[0.185,0.285],[1.5,4],[1.5,3]]),2,np.array([1,1]),80000,200],
    "auv_g": [auv_gep_safe, 4, np.array([[0.148,0.223],[0.185,0.285],[1.5,4],[1.5,3]]),2,np.array([1,1]),80000,200],
   "gripper_c1": [gripper_c1,7,np.array([[10,250],[10,250],[100,300],[0,50],[10,250],[100,300],[1,3.14]]),2,np.array([1,1]),10000,100]
}

def get_true_fronts(function_name,n_vars):
    return None

if __name__ == "__main__":
    # population = np.array([[1.332,2.7,0.25,7,6.08]])
    # f,g = fbg_tactile_sensor(population)
    # population = np.array([[3,3.16,6,3,4.97,1.4,1.68,0.5]])
    # f,g = jtorque_sensor(population)
    # population = np.array([[35,13,87,7.5],[44.6174,9.0015,96.6784,6.1673],[44.6,9,96.7,6.2]])
    # f, g = mau(population)
    # population = np.array([[50,30,3,40],[10.77,40,3,0]])
    # f,g = drag_lift(population)
    # population = np.array([[29,7,15,41,1.5]])
    # f,_ = cf_sensor(population)
    # population = np.array([6000,3.8,0.4])
    # f,_ = vg_ft(population)
    population = np.array([[230.65628,182.67297,300,46.39774,43.90391,144,51126,2.03856]])
    f,g = gripper_c1(popuation)
    print(f,g)

    
