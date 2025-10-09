import numpy as np
import scipy as sp

def wall_building_trajectory(population):
    population = np.atleast_2d(population)
    delta_t = abs(np.diff(population, axis=1))

    n = 3
    k = 7

    u0 = np.zeros([population.shape[0],k+1])
    u1 = np.ones([population.shape[0],k+1])

    ut = []
    for i in range(0, 3):
        ut.append(lambda t: 0 + (t - population[:,i])/np.sum(delta_t, axis=1))

    
def masonry_force_error(population):
    x,l,v,h = population.T

    f = np.zeros([population.shape[0],2])
    g = np.zeros([population.shape[0],2])

    f[:,0] = (6222 + 1.41*x - 50.5*l - 2.05*v + 51.9*h - 0.00174*x**2 + 1.667*l**2 +
                0.00187*v**2 + 0.382*h**2 + 0.0097*x*l + 0.0013*x*v - 0.0285*x*h +
                0.1472*l*v + 4.773*l*h + 0.2907*v*h)

    f[:,1] = (-6.32 + 0.0176*x - 0.419*l - 0.0219*v + 0.43*h - 0.000026*x**2 -
                0.0048*l**2 - 0.000059*v**2 - 0.00463*h**2 + 0.000055*x*l +
                0.000003*x*v - 0.000047*x*h + 0.0011*l*v + 0.00828*l*h + 0.000679*v*h)

    g[:,0] = f[:,0] - 9500
    g[:,1] = -f[:,1]

    return f, g

def fbg_tactile_sensor(population):
    a,b,c,d,L = population.T

    d = np.rint(d)
    
    Ks = (-516.51195 - 2286.07063*a - 21.03937*c + 2201.6575*b - 82.34433*d - 7114.075*a*c
            - 845.7875*a*b - 25.06438*a*d + 2117.45*b*c + 207.79687*c*d + 15.5725*b*d
            + 2874.325 *a*a + 22890.69375*c*c - 707.5375*b*b + 5.66569*d*d 
            + 1566.3125*a*a*c - 999.4375*a*a*b + 16993.75*a*c*c + 601.65625*a*b*b 
            - 14273.75*b*c*c - 568.15625*d*c*c - 17.32187*c*d*d)

    Ls = L + 6

    Kf = (72e3 * (np.pi*0.14**2)/4 )/Ls

    g = Ks - 2*Kf

    violations = g < 0

    penalties = np.where(violations, 100*Ls*np.abs(g), 0)
    
    return Ks*Ls+penalties, g

def jtorque_sensor(population):
    h1,h2,l1,l2,l3,b1,b2,b3 = population.T

    f = np.zeros([population.shape[0],1])

    f = (1042.89 - 188.64*h1 - 58.21*h2 + 280.95*l1 + 198.47*l3 
            + 90.18*b1 - 3017.97*b3 + 32.15*b2 + 20.31*h1*b2
            + 49.32*h2*b3 - 44.67*l1*b1 + 110.96*l1*b3 - 38.41*l1*b2
            - 15.57*l3*b1 + 59.14*l3*b3 - 14.41*l3*b2 + 209.57*b1*b3
            + 19.93*b1*b2 + 143.47*b3*b2 - 13.49*l3**2 - 24.01*b1**2
            + 239.85*b3**2 - 5.98*b2**2)

    g = np.zeros([population.shape[0],1])

    g = (686.35 - 108.36*h1 - 23*h2 - 71.01*l1 + 11.47*l2
            - 45.13*l3 - 29.11*b1 - 54.22*b3 - 7.89*b2 + 13.08*h1*l1
            + 3.47*h1*l3 + 3.52*h1*b1 - 2.35*h2*l2 - 1.4*h2*l3
            + 10.05*h2*b3 + 4.88*h2*b2 + 3.93*l1*l2 + 5.57*l1*l3 - 4*l1*b2
            + 1.44*l2*l3 - 2.75*l2*b1 - 5.61*l2*b3 - 4.33*l2*b2
            - 1.21*l3*b2 + 1.59*b1*b2 + 8.98*b3*b2 + 2.25*b1**2 + 2.27*b2**2)

    violations = g > 101

    penalties = np.where(violations, np.abs(g)*100, 0)

    return -f+penalties, g

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
    v = np.clip(g, 1e-8, None)
    
    return np.column_stack([f, 1/v]), np.full((population.shape[0], 1), -1)

def cf_sensor(population):
    lb,wb,hb,lc,wc = population.T
    cond_n = (0.006886*lb**2 + 0.0603*hb**2 + 0.005711*lc**2 + 0.806135*wc**2
                - 0.014728*lb*wb + 0.010668*lb*hb + 0.030426*lb*wc
                - 0.020664*wb*hb - 0.283394*wb*wc + 0.011607*hb*lc
                - 0.308332*hb*wc - 0.154665*lc*wc - 0.590118*lb
                + 1.55853*wb - 2.00243*hb - 0.41139*lc + 10.65159*wc
                + 19.26557)    
    return cond_n, np.full((population.shape[0], 1), -1)

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

get = {
    #"name":[function, n_vars, bounds, n_obj, minmax, max_evals]
    
    "fbg_tactile_sensor": [fbg_tactile_sensor, 5, np.array([[1.3,1.7],[2.6,3],[0.05,0.25],[3,7],[5,8]]),1,1,10000],
    "torque_sensor": [jtorque_sensor, 8, np.array([[3,4.2],[2,4.2],[5.2,6.2],[3,5.6],[3,5.6],[1.4,4],[1.6,5],[0.5,1.3]]),1,-1,10000],

    "mau": [mau, 4, np.array([[24.5,45.5],[9,17],[60,110],[0,15]]),2,1,30000],
    "drag_lift": [drag_lift, 4, np.array([[10,50],[20,40],[1,3],[0,40]]),2,np.array([1,-1]),60000],
    "drag_lift_inv": [drag_lift_inv, 4, np.array([[10,50],[20,40],[1,3],[0,40]]),2,np.array([-1,1]),60000],
    "auv": [auv, 4, np.array([[0.148,0.223],[0.185,0.285],[1.5,4],[1.5,3]]),2,1,80000],
    "auv_g": [auv_gep_safe, 4, np.array([[0.148,0.223],[0.185,0.285],[1.5,4],[1.5,3]]),2,1,80000],
    "cf_sensor": [cf_sensor, 5, np.array([[25,29],[7,29],[13,15],[38,41],[1.5,2.5]]),1,1,10000]
}

def get_true_fronts(function_name,n_vars):
    return None

if __name__ == "__main__":
    population = np.array([[1.332,2.7,0.25,7,6.08]])
    f,g = fbg_tactile_sensor(population)
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
    print(f,g)
    # print(g)

    