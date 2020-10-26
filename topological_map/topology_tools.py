import sys
import os
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from topological_map.ReadConceptNet import *

class TopoMap():
    def __init__(self):
        # settings - environment
        self.minx = 0
        self.maxx = 10
        self.sizex = self.maxx - self.minx
        self.miny = 0
        self.maxy = 10
        self.sizey = self.maxy - self.miny
        # settings - topological map
        self.num_node = 100  # graph the number of nodes
        self.min_dist = 0.5  # minimum distance between two nodes
        self.prm_k = 5
        self.room_classes = ['kitchen','living_room','bedroom','garage','gym']
        self.num_room = len(self.room_classes)

        # scoring function
        self.CONST_DIST = 1
        self.CONST_CONF = 0.5

        # initialize
        self.prm = self.makePRM() # make PRM
        self.compute_dist() # compute distance between nodes
        self.obj_node= {} #{vertex: object}
        self.obj_class ={}
        self.node_room = np.zeros((self.num_node,self.num_room))# probability
        self.score_room_obj={} # (room, obj): score


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

        # Make Edges
        # distance matrix
        distmat = np.ndarray((self.num_node, self.num_node))
        for ii in range(0, self.num_node):
            for jj in range(ii, self.num_node):
                dist = np.sum((node_pnt[ii, :]-node_pnt[jj, :])**2)
                distmat[ii, jj] = dist
                distmat[jj, ii] = dist

        # edges
        edges = np.zeros((self.num_node, self.num_node))
        edges_dict = {}
        for ii in range(0, self.num_node):
            kdist = list(distmat[ii, 0:self.prm_k+1])
            kmax = max(kdist)
            klist = list(range(0, self.prm_k+1))
            for jj in range(self.prm_k+1, self.num_node):
                if kmax > distmat[ii, jj]:
                    imax = kdist.index(kmax)
                    kdist.pop(imax)
                    klist.pop(imax)
                    kdist.append(distmat[ii, jj])
                    klist.append(jj)
                    kmax = max(kdist)

            for kk in range(0, self.prm_k+1):
                edges[ii, klist[kk]] = kdist[kk]
                edges[klist[kk], ii] = kdist[kk]
                edges_dict[(ii, klist[kk])] = kdist[kk]


        print("Make PRM")
        #plt.imshow(edges)
        #plt.show()

        self.nodes= node_pnt
        self.edges_mat = edges
        self.edges_dict = edges_dict

    def compute_dist(self):
        # dijkstra
        print("Computing Distance")
        dist_mat = np.zeros((self.num_node, self.num_node))
        for ii in range(0,self.num_node):
            dist = [np.inf]*self.num_node
            unvisited = list(range(0, self.num_node))
            dist[ii] = 0

            while len(unvisited) != 0:
                cur_node = dist.index(min([dist[dd] for dd in unvisited]))
                near_idx = np.nonzero(self.edges_mat[cur_node,:])[0]
                for jj in near_idx:
                    if self.edges_mat[cur_node, jj]+ dist[cur_node] < dist[jj]:
                        dist[jj] = self.edges_mat[cur_node, jj]+ dist[cur_node]

                unvisited.remove(cur_node)

            dist_mat[ii:ii+1,:] = dist

        self.dist_mat = dist_mat
#        plt.imshow(dist_mat)
#        plt.show()

    def update_obj(self, obj_pose, obj_class):
        # input: {"ID": pose, 1:[1,2]}
        print("update object")
        for key, val in obj_pose.items():
            # if a new object class is detected, compute room-object score
            if obj_class[key] not in self.obj_class.values():
                for ii in range(0, self.num_room):
                    self.score_room_obj[(self.room_classes[ii], obj_class[key])]\
                        = CNetGetScore(self.room_classes[ii], obj_class[key], 'in')

            self.obj_node[key] = np.argmin((self.nodes[:, 0]-val[0])**2 + (self.nodes[:, 1]-val[1])**2)
            self.obj_class[key] = obj_class[key]

    def draw_map(self):
        print("Drawing")
        # draw topological map
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1])
        for ii in range(0, self.num_node):
            for jj in range(ii, self.num_node):
                if self.edges_mat[ii, jj]>0:
                    plt.plot([self.nodes[ii, 0],self.nodes[jj, 0]],
                    [self.nodes[ii, 1], self.nodes[jj, 1]], color="b", mfc="r",marker="o")
        # draw object
        for key, val in self.obj_node.items():
            plt.text(self.nodes[val,0]+0.1,self.nodes[val,1]+0.1,key)
        plt.show()

    # TODO: here and fcn_scoring
    def update_nodeclass(self): # update probability
        for ii in range(0, self.num_node):
            # probability initialize
            prob = np.zeros(1, self.num_room)
            # filtering object near nodes

            for jj in range(0, self.num_room):

    def fcn_scoring(self, dist, conf): # distance between vertices, confidence level
        # dist, conf: integer or numpy array
        score = np.divide(np.exp(-dist/self.CONST_DIST),(1+np.exp(-conf/self.CONST_CONF)))
        return score


            print("")



if __name__ == '__main__':
    topomap = TopoMap()

    obj_pose = {1:[1,1], 2:[1,1.5],3:[0.5,0.5], 4:[3,3],5:[4,3.5]}
    obj_class = {1:"fridge", 2:"table", 3:"cup", 4:"sofa", 5: "coffee_table"} # ID -class
    topomap.update_obj(obj_pose, obj_class)
    topomap.draw_map()

    print("end?")
    #fig=spatial.voronoi_plot_2d(vor)
    #plt.show(fig)




