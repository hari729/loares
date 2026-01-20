import numpy as np

test_case = np.array([5,6,7,7,1,3,3,3,5])
test_delta = np.diff(test_case)/test_case[:-1]

print(test_case)
print(test_delta)

convergence_pt = np.where(np.abs(test_delta) < 0.1)[0]
print(convergence_pt)

if len(convergence_pt)>0:
    idx = np.where(np.diff(convergence_pt) == 1)[0][0]
    print(idx)
    print(convergence_pt[idx])
    print(test_case[convergence_pt[idx]])
    # convergence[m] = [mean[m][convergence_pt+1][idx],mean['evals'][convergence_pt+1][idx]]
    # convergence[m] = [mean[m][convergence_pt+1][0],mean['evals'][convergence_pt[0]+1]]
else:
    convergence[m] = [np.nan, np.nan]
