from math import comb
from guptri_py import *

import numpy as np
from scipy.linalg import null_space, eigvals
import matplotlib.pyplot as plt

def bernstein(i, degree, u):
    ''' Returns a Berstein polynomial

        i = polynomial number
        degree = degree
        u = sample points
    '''
    return comb(degree, i) * np.power(u, i) * np.power(1 - u, degree - i)

def sample_curve(b, n=100):
    ''' Samples a Bezier curve

        b = sample points (shape is [degree + 1, dimension])
        n = number of points to generate (in the range 0,1)
    '''
    degree = b.shape[0] - 1
    dimension = b.shape[1]
    u = np.linspace(0, 1, n)
    out = np.zeros([n, dimension])
    for i in range(degree + 1):
        bs = bernstein(i, degree, u).reshape(-1,1)
        out += np.hstack([bs] * dimension) * np.vstack([b[i,:]] * n)
    return out

def sample_surface(b, n=10, u=None, v=None):
    ''' Samples a Bezier surface

        b = sample points (shape is [degree + 1, degree + 1, dimension])
        n = number of points to generate on each axis (in the range 0,1)
    '''
    degree1 = b.shape[0] - 1
    degree2 = b.shape[1] - 1
    dimension = b.shape[2]
    if u is None:
        u = np.linspace(0, 1, n)
    elif isinstance(u, float):
        u = np.array([u])
    if v is None:
        v = np.linspace(0, 1, n)
    elif isinstance(v, float):
        v = np.array([v])
    out = np.zeros([len(u), len(v), dimension])
    for i in range(degree1 + 1):
        for j in range(degree2 + 1):
            wu = bernstein(i, degree1, u)
            wv = bernstein(j, degree2, v)
            w = wu.reshape(-1,1).dot(wv.reshape(1,-1))
            s = np.dstack([w] * dimension)
            for k in range(dimension):
                s[:,:,k] *= b[i,j,k]
            out += s
    return out

def parse_bpt(data):
    ''' Parses a BPT file, which is a primitive B-rep format

        Returns a list of patches
    '''
    lines = data.split('\n')
    count = int(lines[0])
    i = 1
    patches = []
    for _ in range(count):
        (n, m) = map(int, lines[i].split(' '))
        i += 1
        patch = []
        for _ in range(n + 1):
            row = []
            for _ in range(m + 1):
                row.append(list(map(float, lines[i].split(' '))))
                i += 1
            patch.append(row)
        patches.append(np.array(patch))
    return patches

def S_v(b, v=None):
    ''' Builds the S_v matrix using the equation on p4
    '''
    degree1 = b.shape[0] - 1
    degree2 = b.shape[1] - 1
    dimension = b.shape[2]
    if v == None:
        # Pick a v that ensures the drop-of-rank property, based on Section 3.2
        v = (2 * min(degree1, degree2) - 1, max(degree1, degree2) - 1)

    stride = (v[0] + 1) * (v[1] + 1)
    out = np.zeros(((degree1 + v[0] + 1) * (degree2 + v[1] + 1),
                    4 * stride))
    for axis in range(dimension + 1):
        if axis == 0:
            c = np.ones_like(b[:,:,axis])
        else:
            c = b[:,:,axis - 1]
        for k in range(v[0] + 1):
            v_k = comb(v[0], k)
            for l in range(v[1] + 1):
                v_l = comb(v[1], l)
                for i in range(degree1 + 1):
                    for j in range(degree2 + 1):
                        # B_{i+k} * B_{j+l}
                        row = (j + l) + (i + k) * (v[1] + degree2 + 1)
                        out[row, l + k * (v[1] + 1) + axis * stride] += \
                            v_k * v_l * comb(degree1, i) * comb(degree2, j) \
                            / (comb(v[0] + degree1, i + k) * comb(v[1] + degree2, j + l)) \
                            * c[i,j]
    return out

def build_M(b):
    ''' Builds a generator function that takes X, Y, Z and returns M

        b are control points on a Bézier surface
    '''
    s = S_v(b)
    null = null_space(s)
    i = int(null.shape[0] / 4)
    return lambda x,y,z: \
        null[:i,:] + null[i:2*i,:] * x \
                   + null[2*i:3*i,:] * y \
                   + null[3*i:4*i,:] * z

def parameterize_ray(M, o, d):
    ''' Combines a MRep M with a ray R given by an origin o and direction d
        Returns A, B matrices such that M(R(t)) = A - tB
    '''
    A = M(*o)
    B = M(o[0] + d[0], o[1] + d[1], o[2] + d[2]) - A
    return A, B

def draw_patches(patches, n=10):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    bounds = []
    for p in patches:
        samples = sample_surface(p, n)
        x = samples[:,:,0]
        y = samples[:,:,1]
        z = samples[:,:,2]
        bounds.append([np.min(x), np.min(y), np.min(z), np.max(x), np.max(y), np.max(z)])
        ax.scatter(samples[:,:,0], samples[:,:,1], samples[:,:,2])
    bounds = np.array(bounds)
    mins = np.min(bounds[:,:3], axis=0)
    maxs = np.max(bounds[:,3:], axis=0)
    scale = np.max(maxs - mins)
    centers = (mins + maxs) / 2

    ax.set_xlim(centers[0] - scale, centers[0] + scale)
    ax.set_ylim(centers[1] - scale, centers[1] + scale)
    ax.set_zlim(centers[2] - scale, centers[2] + scale)
    plt.show()

with open('teapot.bpt') as f:
    patches = parse_bpt(f.read())

def reduce_pencil_easy(A, B):
    while np.linalg.matrix_rank(B) != B.shape[1]:
        (u1, e1, vt1) = np.linalg.svd(B)
        bv1 = np.matmul(B, vt1.T)
        i = np.min(np.where(np.max(np.abs(bv1), axis=0) < 1e-8))
        av1 = np.matmul(A, vt1.T)
        A1 = av1[:,i:]
        (u2, e2, vt2) = np.linalg.svd(A1)
        A = np.matmul(np.matmul(u2.T, A), vt1.T)
        B = np.matmul(np.matmul(u2.T, B), vt1.T)
        mask = np.abs(A) < 1e-8
        if mask.sum() == 0:
            return False
        mask = np.where(mask)
        i = mask[0][0]
        j = mask[1][0]
        A = A[i:,:j]
        B = B[i:,:j]
    if B.shape[0] == B.shape[1]:
        return A, B
    else:
        return reduce_pencil_easy(A.T, B.T)

def pencil_eigenvalues(A, B):
    out = reduce_pencil_easy(A, B)
    if out:
        A, B = out
        eigs = eigvals(A, B)
        return eigs
    else:
        # https://arxiv.org/pdf/1805.07657.pdf is another alternative, maybe
        # faster than GUPTRI but requires you to put the matrix into KCF form
        # yourself, and I don't have time for that!
        S, T, P, Q, kstr = guptri(A, B)
        blocks = np.cumsum(kcf_blocks(kstr), axis=1).astype(int)
        S = S[blocks[0,1]:blocks[0,2], blocks[1,1]:blocks[1,2]]
        T = T[blocks[0,1]:blocks[0,2], blocks[1,1]:blocks[1,2]]
        return S.diagonal() / T.diagonal()

def preimages(M, P):
    ''' Calculates the pre-image of the point P given an M-rep M

        This uses the original M-Rep paper, not the later ray-tracing
        paper, because the latter is inexplicable.  As such, it's not as
        robust and doesn't account for multiple pre-images.
    '''
    # Use last column of SVD to approximate null space of M_v(P)^T
    (u,_,_) = np.linalg.svd(M(*P))
    n = u[:,-1]
    # Hard-coded ratios based on the v used for our patches
    u = n[1] / (2*n[0] + n[1])
    v = n[3] / (5*n[0] + n[3])
    if u >= 0 and u <= 1 and v >= 0 and v <= 1:
        return (u, v)
    else:
        return False

#draw_patches(patches)

b = np.array([[[1,0,0],[1,0,1],[0,0,1]],[[1,1,0],[1,1,1],[0,1,0]],[[1,1,2],[1,2,2],[0,2,2]]])
#patches = [b]
#b = patches[0]
#draw_patches([b], 40)
hits = []
for (i,b) in enumerate(patches):
    print(i)
    M = build_M(b)

    bounds_min = b.reshape(-1, 3).min(axis=0)
    bounds_max = b.reshape(-1, 3).max(axis=0)

    # Test one point
    for i in range(10):
        surf_point = sample_surface(b)[i,i]
        preimages(M, surf_point)
    surf_point = sample_surface(b)[8,8]
    print("surf point:", surf_point)
    ray_origin = np.copy(surf_point)
    ray_dir = np.array([1,0,0])
    ray_origin += ray_dir # add a little offset, as a treat
    eigs = pencil_eigenvalues(*parameterize_ray(M, ray_origin, ray_dir))
    for e in eigs:
        if abs(e.imag) < 1e-8:
            pt = ray_origin - ray_dir * e.real
            if np.all((pt >= bounds_min) * (pt <= bounds_max)):
                hits.append(pt)
                print(pt)
    print()

################################################################################
'''
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for p in patches:
    M = build_M(p)
    def check(x, y, z):
        return np.linalg.matrix_rank(M(x, y, z))

    samples = sample_surface(p)

    #assert(null.shape[1] == 9)
    #for (x, y, z) in p[0,:,:]:
    #    print(null.shape, check(x,y,z))
    base_rank = np.linalg.matrix_rank(M(100,100,100))
    failed = False
    for row in samples:
        for (x,y,z) in row:
            if np.linalg.matrix_rank(M(x,y,z)) == min(null.shape[0]/4, null.shape[1]):
                print(base_rank, np.linalg.svd(M(x,y,z))[1])
                failed = True
                #panic("at the disco")

    if failed:
        print(p)
        ax.scatter(samples[:,:,0], samples[:,:,1], samples[:,:,2])
#ax.set_xlim(-3, 3)
#ax.set_ylim(-3, 3)
#ax.set_zlim(-1, 5)
plt.show()
'''