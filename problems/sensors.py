import numpy as np

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

def cf_sensor(population):
    lb,wb,hb,lc,wc = population.T
    cond_n = (0.006886*lb**2 + 0.0603*hb**2 + 0.005711*lc**2 + 0.806135*wc**2
                - 0.014728*lb*wb + 0.010668*lb*hb + 0.030426*lb*wc
                - 0.020664*wb*hb - 0.283394*wb*wc + 0.011607*hb*lc
                - 0.308332*hb*wc - 0.154665*lc*wc - 0.590118*lb
                + 1.55853*wb - 2.00243*hb - 0.41139*lc + 10.65159*wc
                + 19.26557)    
    return cond_n, np.full((population.shape[0], 1), -1)


get = {
    #"name":[function, n_vars, bounds, n_obj, minmax, max_evals]
    
    "fbg_tactile_sensor": [fbg_tactile_sensor, 5, np.array([[1.3,1.7],[2.6,3],[0.05,0.25],[3,7],[5,8]]),1,1,10000],
    "torque_sensor": [jtorque_sensor, 8, np.array([[3,4.2],[2,4.2],[5.2,6.2],[3,5.6],[3,5.6],[1.4,4],[1.6,5],[0.5,1.3]]),1,-1,10000],
    "cf_sensor": [cf_sensor, 5, np.array([[25,29],[7,29],[13,15],[38,41],[1.5,2.5]]),1,1,10000]

}