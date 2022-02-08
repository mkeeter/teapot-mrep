from mrep import *

import os

import numpy as np
import scipy.linalg
import matplotlib
import pylab as plt

SAVE_CURVE_FIG = False
SAVE_RANK_FIGS = False
SAVE_CURVE_ANIM = False
SAVE_PROJ_FIGS = False
SAVE_RAYTRACE_FIGS = True

def image_path(s):
    return os.path.join('../../../Web/projects/mrep/', s)

b = np.array([[0,0],[1,2],[2,1],[3,3]])

if SAVE_CURVE_FIG:
    plt.figure(figsize=(4, 4), dpi=150)
    plt.plot(*sample_curve(b).T)
    plt.plot(b[:,0], b[:,1])
    plt.axis('square')
    plt.scatter([], [])
    plt.scatter(b[:,0], b[:,1])
    plt.savefig(image_path('bezier.svg'), bbox_inches='tight', transparent=True)

s = S_v(2, b)
null = null_space(s)
m0 = null[:3,:]
mx = null[3:6,:]
my = null[6:9,:]
def M(p):
    return m0 + p[0] * mx + p[1] * my

if SAVE_RANK_FIGS:
    p = np.linspace(-0.2, 3.2, 200)
    def e(p):
        return np.product(np.linalg.svd(M(p))[1])
    out = np.zeros((len(p), len(p)))
    for (i, x) in enumerate(p):
        for (j, y) in enumerate(p):
            out[-j - 1,i] = e([x,y])
    plt.figure(figsize=(4, 3), dpi=150)
    plt.imshow(out, extent=(-0.2, 3.2, -0.2, 3.2))
    plt.plot(*sample_curve(b).T, 'w')
    plt.colorbar()
    plt.savefig(image_path('rank.svg'), bbox_inches='tight', transparent=True)

    plt.close()
    plt.figure(figsize=(4, 3), dpi=150)
    plt.imshow(out, extent=(-0.2, 3.2, -0.2, 3.2), norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.savefig(image_path('rank_log.svg'), bbox_inches='tight', transparent=True)

if SAVE_CURVE_ANIM:
    plt.figure(figsize=(3, 3), dpi=300)
    num_frames = 100
    for (i, s) in enumerate(np.linspace(0, 1, num_frames)):
        plt.clf()
        phi = np.array([bernstein(i, m0.shape[0] - 1, s) for i in range(m0.shape[0])])
        z = list(zip(phi.dot(m0), phi.dot(mx), phi.dot(my)))
        plt.plot(*sample_curve(b).T)
        for (co, cx, cy) in z:
            def y(x):
                return -(co + cx * x) / cy
            plt.plot([-1, 4], [y(-1), y(4)])

        (x, y) = sample_curve(b, u=s).T
        plt.plot(x, y, 'k.')
        plt.xlim([-0.2, 3.2])
        plt.ylim([-0.2, 3.2])
        plt.yticks([0,1,2,3])
        plt.savefig(image_path('anim{:03d}.png'.format(i)), bbox_inches='tight', transparent=True)
    import subprocess
    subprocess.run('convert -delay 2 -dispose previous anim*.png sweep@2x.gif', shell=True, check=True, cwd=image_path(''))
    subprocess.run('rm anim*.png', shell=True, check=True, cwd=image_path(''))


if SAVE_PROJ_FIGS:
    p = np.linspace(-0.2, 3.2, 200)
    d = 2
    def s(p):
        _, _, U = np.linalg.svd(M(p).T)
        n = U[-1,:] # equivalent to null_space(M(p).T), but more general
        return n[1] / (d  * n[0] + n[1])

    for i in np.linspace(0, 1, 100):
        pt = sample_curve(b, u=i)[0, :]
        print(i, end='\t')
        print(s(pt))
    out = np.zeros((len(p), len(p)))
    for (i, x) in enumerate(p):
        for (j, y) in enumerate(p):
            out[-j - 1,i] = s([x,y])
    out[out < -1] = -1
    out[out > 1] = 1
    plt.figure(figsize=(4, 3), dpi=150)
    plt.imshow(out, extent=(-0.2, 3.2, -0.2, 3.2))
    plt.plot(*sample_curve(b).T, 'w')
    plt.colorbar()
    plt.show()

if SAVE_RAYTRACE_FIGS:
    from mrep3 import parse_bpt, prepare, raytrace, surface_derivs
    with open('../teapot.bpt') as f:
        patches = parse_bpt(f.read())
    implicit_patches = prepare(patches)

    camera_pos = np.array([3,3,3])
    camera_look = np.array([0.07,0.1,1.4])
    camera_dir = camera_look - camera_pos
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    camera_up = np.array([0,0,1])
    camera_x = np.cross(camera_dir, camera_up)
    camera_x = camera_x / np.linalg.norm(camera_x)
    camera_scale = 6
    image_size = 400
    out_dist = np.zeros((image_size, image_size), dtype=np.float64)
    out_rgb = np.zeros((image_size, image_size, 3), dtype=np.float64)
    out_uv = np.zeros((image_size, image_size, 3), dtype=np.float64)
    max_search = np.zeros((image_size, image_size), dtype=int)
    actual_search = np.zeros((image_size, image_size), dtype=int)

    from datetime import datetime
    start = datetime.now()
    for x in range(out_dist.shape[0]):
        print("{}/{}".format(x + 1, out_dist.shape[0]))
        for y in range(out_dist.shape[1]):
            pos = camera_pos + \
                camera_x * (x/out_dist.shape[0] - 0.5) * camera_scale + \
                camera_up * (y/out_dist.shape[1] - 0.5) * camera_scale
            dist, index, uv, ms, acs = raytrace(pos, camera_dir, implicit_patches)
            max_search[out_dist.shape[0] - y - 1, x] = ms
            actual_search[out_dist.shape[0] - y - 1, x] = acs
            if index is not None:
                out_dist[out_dist.shape[0] - y - 1, x] = dist
                out_uv[out_dist.shape[0] - y - 1, x,:2] = uv
                norm = surface_derivs(patches[index], u=uv[1], v=uv[0])
                out_rgb[out_dist.shape[0] - y - 1, x,:] = norm
    print(datetime.now() - start)

    def plot_colorbar(data, filename):
        cmap = matplotlib.cm.viridis
        plt.figure(figsize=(8, 6), dpi=150)
        norm = matplotlib.colors.BoundaryNorm(range(0,9), cmap.N)
        plt.imshow(data)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(image_path(filename), bbox_inches='tight', transparent=True)
        plt.close()
    #plot_colorbar(max_search, 'max_search.svg')
    #plot_colorbar(actual_search, 'actual_search.svg')

    #plt.imsave(image_path('patch_uv.png'), out_uv)
    #plt.imsave(image_path('final.png'), np.abs(out_rgb))

pts = sample_curve(b)
