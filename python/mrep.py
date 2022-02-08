from math import comb
import numpy as np
from scipy.linalg import null_space

def bernstein(i, degree, u):
    ''' Returns a Berstein polynomial

        i = polynomial number
        degree = degree
        u = sample points
    '''
    return comb(degree, i) * np.power(u, i) * np.power(1 - u, degree - i)

def sample_curve(b, n=100, u=None):
    ''' Samples a Bezier curve

        b = sample points (shape is [degree + 1, dimension])
        n = number of points to generate (in the range 0,1)
        u = point or points to sample at
    '''
    degree = b.shape[0] - 1
    dimension = b.shape[1]
    if u is None:
        u = np.linspace(0, 1, n)
    elif isinstance(u, float) or isinstance(u, int):
        u = np.array([u], dtype=np.float64)
    out = np.zeros([len(u), dimension])
    for i in range(degree + 1):
        bs = bernstein(i, degree, u).reshape(-1,1)
        out += np.hstack([bs] * dimension) * np.vstack([b[i,:]] * len(u))
    return out

m = np.array([[0,0,0,0,0,0],[-1,1,-1,3/2,1/2,-1],[0,-1,0,-1,-1,0]])
mx = np.array([[2,-2,2,-3,-1,2],[0,0,0,0,0,1],[0,0,0,0,1,0]])
my = np.array([[2,-1,2,-2,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0]])
mz = np.array([[0,0,2,-1,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]])
def M(p):
    return m + p[0] * mx + p[1] * my + p[2] * mz

b = np.array([[0, 0,0], [1/3,0,0], [2/3,1/3,0], [1,1,1]])
out = sample_curve(b)
#for o in out: print(np.linalg.svd(M(o))[1])

def S_v(v, b):
    ''' Builds the S_v matrix using the B^v_j equation on p3
    '''
    degree = b.shape[0] - 1
    dimension = b.shape[1]

    out = np.zeros((degree + v + 1, (dimension + 1) * (v + 1)))
    for axis in range(dimension + 1):
        if axis == 0:
            c = np.ones_like(b[:,axis])
        else:
            c = b[:,axis - 1]
        for j in range(v + 1):
            for i in range(degree + 1):
                out[i + j, j + axis * (v + 1)] += comb(v, j) * comb(degree, i) / comb(degree + v, i + j) * c[i]
    return out


b = np.array([[0, 0,0], [1/3,0,0], [2/3,1/3,0], [1,1,1]])
out = sample_curve(b)
s = S_v(2, b)
null = null_space(s)
#null = np.array([[0,0,0],[-1,-1,-1],[1,1,1],[0,0,1],[1,1,0],[0,1,0],[1,0,0],[1,0,0]])
m = null[:3,:]
mx = null[3:6,:]
my = null[6:9,:]
mz = null[9:12,:]

def M(p):
    return m + p[0] * mx + p[1] * my + p[2] * mz
#for o in out: print(np.linalg.svd(M(o))[1])
'''
S has size (d + v + 1) x 4 * (v + 1)
It should therefore have rank (d + v + 1) and nullity 4 * (v + 1) - (d + v + 1)
=> 3*(v + 1) - d
'''

out = sample_curve(b, n=10)
for i in range(10):
    P = out[i]
    Mp = M(P)
    MpT = Mp.T
    n = null_space(MpT)
    print(n[1] / (2*n[0] + n[1]))
