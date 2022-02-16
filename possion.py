# Python3
# https://github.com/jayroxis/poisson
# Code cleaned up from https://github.com/daleroberts/poisson

import numpy as np
from numpy.matlib import zeros, ones
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay as Delaunay
from scipy.interpolate import LinearNDInterpolator


def mesh(xs, ys, npoints):
    # randomly choose some points
    rng = np.random.RandomState(1234567890)
    rx = rng.uniform(xs[0], xs[1], size=npoints)
    ry = rng.uniform(ys[0], ys[1], size=npoints)
    
    # only take points in domain
    nx, ny = [], []
    for x,y in zip(rx,ry):
        if in_domain(x,y):
            nx.append(x)
            ny.append(y)
            
    # Delaunay triangulation
    points = np.stack((np.array(nx), np.array(ny)), axis=1)
    tri = Delaunay(points)
    return tri
            
    
def A_e(v):
    # take vertices of element and return contribution to A
    G = np.vstack((ones((1,3)), v.T)).I * np.vstack((zeros((1,2)), np.eye(2)))
    return np.linalg.det(np.vstack((ones((1,3)), v.T))) * G * G.T / 2

  
def b_e(v):
    # take vertices of element and return contribution to b
    vS = v.sum(axis=0)/3.0 # Centre of gravity
    return f(vS) * ((v[1,0]-v[0,0])*(v[2,1]-v[0,1])-(v[2,0]-v[0,0])*(v[1,1]-v[0,1])) / 6.0

  
def poisson(tri, boundary):
    # get elements and vertices from mesh
    elements = tri.simplices
    vertices = tri.points
    
    # number of vertices and elements
    N = vertices.shape[0]
    E = elements.shape[0]
    
    #Loop over elements and assemble LHS and RHS 
    A = zeros((N,N))
    b = zeros((N,1))
    for j in range(E):
        index = (elements[j,:]).tolist()
        A[np.ix_(index,index)] += A_e(vertices[index,:])
        b[index] += b_e(vertices[index,:])
        
    # find the 'free' vertices that we need to solve for    
    free = list(set(range(len(vertices))) - set(boundary))
    
    # initialise solution to zero so 'non-free' vertices are by default zero
    u = zeros((N,1))
    
    # solve for 'free' vertices.
    u[free] = np.linalg.solve(A[np.ix_(free,free)], b[free])
    return np.array(u)
    
    
def f(v):
    # the RHS f
    x, y = v
    f = 2.0 * np.cos(10.0 * x) * np.sin(10.0 * y) + np.sin(10.0 * x * y)
    return 1

  
def in_domain(x,y):
    # is a point in the domain?
    return np.sqrt(x ** 2 + y ** 2) <= 1
  

# demo
if __name__ == "__main__":
    xs = (-1.,1.)
    ys = (-1.,1.)
    npoints = 1000

    # generate mesh and determine boundary vertices
    tri = mesh(xs, ys, npoints)
    boundary = tri.convex_hull
    boundary = boundary.flatten().tolist()

    # solve Poisson equationmatlib ones
    u  = poisson(tri, boundary).flatten()

    # interpolate values and plot a nice image
    interpolator = LinearNDInterpolator(tri.points, u)
    X, Y = np.meshgrid(np.linspace(xs[0], xs[1]), np.linspace(ys[0], ys[1]))
    z = interpolator(X, Y)
    z = np.where(np.isinf(z), 0.0, z)
    extent = (xs[0], xs[1], ys[0], ys[1])

    plt.ioff()
    plt.clf()
    plt.imshow(np.nan_to_num(z), interpolation='bilinear', extent=extent, origin='lower', cmap='rainbow')
    plt.show()
    plt.savefig('sol.png', bb_inches='tight')
