import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt

class TopoMap():
    def __init__(self):
        # settings
        self.minx = 0
        self.maxx = 10
        self.sizex = self.maxx - self.minx
        self.miny = 0
        self.maxy = 10
        self.sizey = self.maxy - self.miny

        self.num_node = 50  # graph the number of nodes
        self.min_dist = 0.5  # minimum distance between two nodes
        self.prm = self.makePRM()

    def makePRM(self): # probabilistic roadmap
        # initialize
        node_pnt = np.ndarray((self.num_node, 2))

        # Sample random nodes
        node_pnt[0:1, :] = np.multiply(np.random.rand(1, 2), [self.sizex, self.sizey]) + [self.minx, self.miny]
        idx = 1
        for ii in range(1, self.num_node * 2):
            pnt = np.multiply(np.random.rand(1, 2), [self.sizex, self.sizey]) + [self.minx, self.miny]
            y = spatial.distance.cdist(node_pnt[0:idx, :], pnt)
            if min(y) > self.min_dist:
                # TODO: if check whether a point is not occupied
                node_pnt[idx:idx + 1, :] = pnt
                idx = idx + 1

            if idx > self.num_node:
                print("end")
                break
        # Edges
        


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

    # Make an voronoi diagram
    vor = spatial.Voronoi(node_pnt)
    return vor

def PRM():

def GetRoomInfo(v):
    print("Getroominfo")


if __name__ == '__main__':
    vor = generate_graph()
    fig=spatial.voronoi_plot_2d(vor)
    plt.show(fig)




