import numpy as np
from pymoo.util.normalization import normalize

def welded_beam(pop):

    x1,x2,x3,x4 = pop.T

    f = 1.10471*x2*x1**2 + 0.04811*x3*x4*(14+x2)

    E = 30e6 #psi
    s_max = 30000 #psi
    L = 14 #in
    G = 12e6 #psi
    P = 6000 #lb
    t_max = 13600 #psi
    d_max = 0.25 #in

    M = P * (L + ( x2 / 2 ))
    R = np.sqrt((x2**2 / 4) + ((x1+x3)/2)**2)
    d_s = (4*P*L**3)/(E*x4*x3**3)
    J = 2*((x1*x2*np.sqrt(2))*((x2**2 / 12) + ((x1 + x3)/2)**2))
    s_s = (6*P*L)/(x4*x3**2)
    Pc_s = ((4.013*E*np.sqrt((x4**6 * x3**2)/36) / L**2)*
            (1 - (x3/(2*L))*np.sqrt(E/(4*G)))  )

    t_1 = P / (np.sqrt(2)*x1*x2)
    t_2 = M*R/J
    t_s = np.sqrt(t_1**2 + 2 * t_1 * t_2 * (x2 / (2 * R)) + t_2**2)

    g = np.zeros([pop.shape[0],7])

    g[:,0] = t_s - t_max
    g[:,1] = s_s - s_max
    g[:,2] = x1 - x4
    g[:,3] = f - 5
    g[:,4] = 0.125 - x1
    g[:,5] = d_s - d_max
    g[:,6] = P - Pc_s

    violations = g > 0

    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1

    penalties = np.where(lt1, penalties**0.5, penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

def three_bar_truss(pop):

    l = 100 #cm
    s = 2 #kN/cm^2
    P = 2 #kN/cm^2

    f = l * (2*np.sqrt(2)*pop[:,0] + pop[:,1])

    g = np.zeros([pop.shape[0], 3])

    g[:,0] = P * ((np.sqrt(2)*pop[:,0] +  pop[:,1])/
                            (np.sqrt(2)*pop[:,0]**2 +  2*pop[:,0]*pop[:,1] + 1e-8)) - s
    g[:,1] = P * (pop[:,1])/(np.sqrt(2)*pop[:,0]**2 +  2*pop[:,0]*pop[:,1] + 1e-8) - s
    g[:,2] = P * 1/(np.sqrt(2)*pop[:,1] + pop[:,0] + 1e-8) - s

    violations = g > 0

    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1

    penalties = np.where(lt1, 200*penalties, 100*penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

def cantilever_beam(pop):
    
    f = 0.0624 * (np.sum(pop, axis=1))
    
    g = 61/pop[:,0]**3 + 37/pop[:,1]**3 + 19/pop[:,2]**3 + 7/pop[:,3]**3 + 1/pop[:,4]**3 - 1

    violations = g > 0

    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1

    penalties = np.where(lt1, 100*penalties, penalties**2)

    objective_values = f + penalties

    return objective_values, g

def gear_train(pop):

    pop = np.round(pop).astype(int)

    f = (1/6.931 - (pop[:,1]*pop[:,2])/(pop[:,0]*pop[:,3]))**2

    return f, None

def tension_compression_spring(pop):

    f = (pop[:,2] + 2) * pop[:,1]*pop[:,0]**2

    g = np.zeros([pop.shape[0], 4])

    g[:,0] = 1 - (pop[:,2]*pop[:,1]**3)/(71785*pop[:,0]**4 + 1e-8)
    g[:,1] = ((4*pop[:,1]**2 - pop[:,0]*pop[:,1])/(12566 *( pop[:,1]*pop[:,0]**3 - pop[:,0]**4) + 1e-8)
                + 1/(5108* pop[:,0]**2) - 1 + 1e-8)
    g[:,2] = 1 - (140.45*pop[:,0])/(pop[:,2]*pop[:,1]**2 + 1e-8)
    g[:,3] = (pop[:,0]+pop[:,1])/1.5 - 1

    violations = g > 0

    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1

    penalties = np.where(lt1, penalties, penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g    

def pressure_vessel(pop):

    pop[:,0] = np.round(pop[:,0] / 0.0625) * 0.0625
    pop[:,1] = np.round(pop[:,1] / 0.0625) * 0.0625

    f = ( 0.6224*pop[:,0]*pop[:,2]*pop[:,3] + 1.7781*pop[:,1]*pop[:,2]**2 
                + 3.1661*pop[:,3]*pop[:,0]**2 + 19.84*pop[:,2]*pop[:,0]**2 )

    g = np.zeros([pop.shape[0], 4])

    g[:,0] = -pop[:,0] + 0.0193*pop[:,2]
    g[:,1] = -pop[:,1] + 0.00954*pop[:,2]
    g[:,2] = -np.pi*pop[:,3]*pop[:,2]**2 - (4/3)*np.pi*pop[:,2]**3 + 1296000
    g[:,3] = pop[:,3] - 240

    violations = g > 0

    penalties = np.where(violations, g, 0)

    lt1 = penalties < 1

    penalties = np.where(lt1, 3e5*penalties**0.1, 3e5*penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g  

def speed_reducer(pop):

    f = (0.7854*pop[:,0]*pop[:,1]**2 * (3.3333*pop[:,2]**2 + 14.9334*pop[:,2] - 43.0934)
            - 1.508*pop[:,0]*(pop[:,5]**2 + pop[:,6]**2) + 7.4777*(pop[:,5]**3 + pop[:,6]**3)
            + 0.7854*(pop[:,3]*pop[:,5]**2 + pop[:,4]*pop[:,6]**2) )

    g = np.zeros([pop.shape[0],11])

    g[:,0] = 27/(pop[:,0]*pop[:,2]*pop[:,1]**2) - 1
    g[:,1] = 397.5/(pop[:,0] * pop[:,1]**2 * pop[:,2]**2) - 1
    g[:,2] = (1.93*pop[:,3]**3)/(pop[:,1]*pop[:,2]*pop[:,5]**4) - 1
    g[:,3] = (1.93*pop[:,4]**3)/(pop[:,1]*pop[:,2]*pop[:,5]**4) - 1
    g[:,4] = np.sqrt((745*pop[:,3]/(pop[:,1]*pop[:,2]))**2 + 16.9e6)/(110*pop[:,5]**3) - 1
    g[:,5] = np.sqrt((745*pop[:,4]/(pop[:,1]*pop[:,2]))**2 + 157.5e6)/(85*pop[:,6]**3) - 1
    g[:,6] = pop[:,1]*pop[:,2]/40 - 1
    g[:,7] = 5*pop[:,1]/pop[:,0] - 1
    g[:,8] = pop[:,0]/(12*pop[:,1]) - 1
    g[:,9] = (1.5*pop[:,5] + 1.9)/pop[:,3] - 1
    g[:,10] = (1.1*pop[:,6] + 1.9)/pop[:,4] - 1

    violations = g > 0

    penalties = np.where(violations, g, 0)

    # lt1 = penalties < 1

    # penalties = np.where(lt1, penalties, penalties**2)

    objective_values = f + np.sum(10000*penalties, axis=1)

    return objective_values, g

def i_beam(pop):

    b,h,tw,tf = pop.T

    f = 5000 / ((tw*(h-2*tf)**3)/12 + (b*tf**3)/6 + 2*b*tf*((h-tf)/2)**2)    
    
    g = np.zeros([pop.shape[0], 2])

    g[:,0] = 2 * b*tf + tw*(h-2*tf) - 300
    g[:,1] = ( 180000*h / (tw*(h-2*tf)**3 + 2*b*tf*(4*tf**2 + 3*h*(h-2*tf)))
               + 15000*b / ((h-2*tf)*tw**2 + 2*tf*b**3) - 6 )

    violations = g > 0
    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1
    penalties = np.where(lt1, 100 * penalties, 100 * penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

def tubular_column(pop):
    Sy = 500 #kgf/cm^2
    E = 0.85*1e6 #kgf/cm^2
    P = 2500 #kgf
    l = 250 #cm

    d,t = pop.T

    f = 9.8*d*t + 2*d

    g = np.zeros([pop.shape[0], 6])

    g[:,0] = P/(np.pi*d*t*Sy) - 1
    g[:,1] = 8*P*l**2/(np.pi**3 * E*d*t*(d**2 + t**2)) - 1
    g[:,2] = 2/d - 1
    g[:,3] = d/14 - 1
    g[:,4] = 0.2/t - 1
    g[:,5] = t/0.8 - 1

    violations = g > 0
    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1
    penalties = np.where(lt1, 100 * penalties, 200 * penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

def piston_lever(pop):
    Q = 10000 #lbs
    L = 240 #in
    Mmax = 1.8e6 #lbs.in
    P = 1500 #psi
    theta = np.deg2rad(45)

    h,b,x,d = pop.T

    R = np.abs(-x*(x*np.sin(theta) + h) + h*(b-x*np.cos(theta)))/np.sqrt((x-b)**2 + h**2)
    F = (np.pi*P*d**2) / 4
    L1 = np.sqrt((x-b)**2 + h**2)
    L2 = np.sqrt((x*np.sin(theta) + h)**2 + (b-x*np.cos(theta))**2)

    f = 0.25*np.pi*d**2 * (L2 - L1)

    g = np.zeros([pop.shape[0],4])

    g[:,0] = Q*L*np.cos(theta) - R*F
    g[:,1] = -Mmax + Q*(L-x)
    g[:,2] = 1.2*(L2 - L1) - L1
    g[:,3] = d/2 - b

    violations = g > 0
    penalties = np.where(violations, g, 0)

    lt1 = penalties < 1
    penalties = np.where(lt1, 100 * penalties, 10*penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

def corrugated_bulkhead(pop):

    b,h,l,t = pop.T

    l_h = l**2 - h**2
    invalid = l_h <= 0
    l_h = np.where(invalid, 1e-6, l_h)
    l_h = np.sqrt(l_h)

    f = 5.885*t*(b+l)/(b + l_h )

    g = np.zeros([pop.shape[0],6])

    l_h = np.where(invalid, 1000, l_h)

    g[:,0] = t*h*(0.4*b + l/6) - 8.94*(b + l_h)
    g[:,1] = t*h**2 * (0.2*b + l/12) - 2.2*(8.94*(b + l_h))**(4/3)
    g[:,2] = t - 0.0156*b - 0.15
    g[:,3] = t - 0.0156*l - 0.15
    g[:,4] = t - 1.05
    g[:,5] = l - h

    violations = g < 0
    penalties = np.where(violations, np.abs(g), 0)

    lt1 = penalties < 1
    penalties = np.where(lt1, 100 * penalties, penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

def car_side_impact(pop):

    s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11 = pop.T

    s8 = np.where(np.abs(s8 - 0.192) < np.abs(s8 - 0.345), 0.192, 0.345)
    s9 = np.where(np.abs(s9 - 0.192) < np.abs(s9 - 0.345), 0.192, 0.345)

    f = 1.98+4.9*s1+6.67*s2+6.98*s3+4.01*s4+1.78*s5+2.73*s7

    g = np.zeros([pop.shape[0], 10])

    g[:,0] = 1.16 - 0.3717*s2*s4 - 0.00931*s2*s10 - 0.484*s3*s9 + 0.01343*s6*s10 - 1
    g[:,1] = ( 0.261 - 0.0159*s1*s2 - 0.188*s1*s8 - 0.019*s2*s7 + 0.0144*s3*s5 + 0.0008757*s5*s10
               + 0.08045*s6*s9 + 0.00139*s8*s11 + 0.00001575*s10*s11 - 0.32 )
    g[:,2] = (0.214 + 0.00817*s5 - 0.131*s1*s8 - 0.0704*s1*s9 + 0.03099*s2*s6
               - 0.018*s2*s7 + 0.0208*s3*s8 + 0.121*s3*s9 - 0.00364*s5*s6
               + 0.0007715*s5*s10 - 0.0005354*s6*s10 + 0.00121*s8*s11 + 0.00184*s9*s10 - 0.02*s2**2 - 0.32 )
    g[:,3] = 0.74 - 0.61*s2 - 0.163*s3*s8 + 0.001232*s3*s10 - 0.166*s7*s9 + 0.227*s2**2 - 0.32
    g[:,4] = 28.98 + 3.818*s3 - 4.2*s1*s2 + 0.0207*s5*s10 + 6.63*s6*s9 - 7.7*s7*s8 + 0.32*s9*s10 - 32
    g[:,5] = (33.86 + 2.95*s3 + 0.1792*s10 - 5.057*s1*s2 - 11*s2*s8
                - 0.0215*s5*s10 - 9.98*s7*s8 + 22*s8*s9 - 32) 
    g[:,6] = 46.36 - 9.9*s2 - 12.9*s1*s8 + 0.1107*s3*s10 - 32
    g[:,7] = 4.72 - 0.5*s4 - 0.19*s2*s3 - 0.0122*s4*s10 + 0.009325*s6*s10 + 0.000191*s11**2 - 4
    g[:,8] = 10.58 - 0.674*s1*s2 - 1.95*s2*s8 + 0.02054*s3*s10 - 0.0198*s4*s10 + 0.028*s6*s10 - 9.9
    g[:,9] = 16.45 - 0.489*s3*s7 - 0.843*s5*s6 + 0.0432*s9*s10 - 0.0556*s9*s11 - 0.000786*s11**2 - 15.7

    violations = g > 0
    penalties = np.where(violations, g, 0)

    lt1 = penalties < 1
    penalties = np.where(lt1, 100*penalties, penalties**2)

    objective_values = f + np.sum(penalties, axis=1)

    return objective_values, g

get = {
    #"function_name":[function, no_variables, bounds]
    "welded_beam": [welded_beam, 4, np.array([[0.1,2],[0.1,10],[0.1,10],[0.1,2]])],
    "three_bar_truss": [three_bar_truss, 2, np.array([[0,1]] * 2)],
    "cantilever_beam": [cantilever_beam, 5, np.array([[0.01,100]]*5)],
    "gear_train": [gear_train, 4, np.array([[12,60]]*4)],
    "tension_compression_spring": [tension_compression_spring, 3, np.array([[0.05,2],[0.25,1.3],[2,15]])],
    "pressure_vessel": [pressure_vessel, 4, np.array([[0.0625,99],[0.0625,99],[10,200],[10,200]])],
    "speed_reducer": [speed_reducer, 7, np.array([[2.6,3.6],[0.7,0.8],[17,28],[7.3,8.3],[7.3,8.3],[2.9,3.9],[5,5.5]])],
    "i_beam": [i_beam, 4, np.array([[10,50],[10,80],[0.9,5],[0.9,5]])],
    "tubular_column": [tubular_column, 2, np.array([[2,14],[0.2,0.8]])],
    "piston_lever": [piston_lever, 4, np.array([[0.05,500],[0.05,500],[0.05,120],[0.05,500]])],
    "corrugated_bulkhead": [corrugated_bulkhead, 4, np.array([[0,100],[0,100],[0,100],[0,5]])],
    "car_side_impact": [car_side_impact, 11, np.array([[0.5,1.5],[0.5,1.5],[0.5,1.5],[0.5,1.5],[0.5,1.5],
                                                        [0.5,1.5],[0.5,1.5],[0.192,0.345],[0.192,0.345],
                                                        [-30,30],[-30,30]]) ]
}


def get_true_fronts(function_name,n_vars):
    return None