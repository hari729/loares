import numpy as np

def null(new_p, bounds):
    return new_p

def opposition(new_p, bounds):
    opp_p = np.sum(bounds, axis=1) - new_p
    return opp_p


# def q_opposition(new_p, bounds)

get = {
    "null":null,
    "opposition":opposition,
}


if __name__ == "__main__":

    new_p = np.array([[5,4,3,2]])

    bounds = np.array([[1,6],[2,10],[1,5],[1,10]])

    print(opposition(new_p,bounds))