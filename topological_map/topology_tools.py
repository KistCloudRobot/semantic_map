import numpy as np
import scipy.spatial as spatial

def generate_graph():
    # settings
    minx = 0; maxx = 10; sizex = maxx - minx
    miny = 0; maxy = 10; sizey = maxy - miny
    num_node = 50  # graph the number of nodes
    min_dist = 0.5  # minimum distance between two nodes

    # initialize
    node_pnt = np.ndarray((num_node, 2))

    # Sample random nodes
    node_pnt[0:1, :] = np.multiply(np.random.rand(1, 2), [sizex, sizey]) + [minx, miny]
    idx = 1
    for ii in range(1, num_node * 2):
        pnt = np.multiply(np.random.rand(1, 2), [sizex, sizey]) + [minx, miny]
        y = spatial.distance.cdist(node_pnt[0:idx, :], pnt)
        if min(y) > min_dist:
            # TODO: if check whether a point is not occupied
            node_pnt[idx:idx + 1, :] = pnt
            idx = idx + 1

        if idx > num_node:
            print("end")
            break
    # connect edges
    for ii in range(0,num_node):
