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
        self.CONST_DIST = 10
        self.CONST_CONF = 2

        # initialize
        self.prm = self.makePRM() # make PRM
        self.compute_dist() # compute distance between nodes
        self.obj_node= {} #{vertex: object}
        self.obj_class ={}
        self.prob_node_room = np.zeros((self.num_node,self.num_room))# probability
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
                    self.score_room_obj[(self.room_classes[ii], obj_class[key])]= 1/(1+np.exp(-CNetGetScore(obj_class[key], self.room_classes[ii], 'in')/self.CONST_CONF))

            self.obj_node[key] = np.argmin((self.nodes[:, 0]-val[0])**2 + (self.nodes[:, 1]-val[1])**2)
            self.obj_class[key] = obj_class[key]
        self.num_obj = len(self.obj_node)


    def draw_map(self, draw_room):
        print("Drawing")
        # draw topological map
        for ii in range(0, self.num_node):
            for jj in range(ii, self.num_node):
                if self.edges_mat[ii, jj]>0:
                    plt.plot([self.nodes[ii, 0],self.nodes[jj, 0]],
                    [self.nodes[ii, 1], self.nodes[jj, 1]], color="b")
        # Draw nodes
        plt_node = []
        for ii in range(0, self.num_node):
            plt_node.append(plt.plot(self.nodes[ii, 0], self.nodes[ii, 1], c="r", mfc="r", marker="o"))

        # draw object
        for key, val in self.obj_node.items():
            plt.text(self.nodes[val,0]+0.1,self.nodes[val,1]+0.1,key)

        # test - distance
        #for ii in range(0, self.num_node):
        #    for jj in range(0, self.num_node):
        #        linecolor = (np.exp(-self.dist_mat[jj, ii]/self.CONST_DIST), 0, 0)
        #        plt.setp(plt_node[jj], mfc=linecolor, c=linecolor)
        #    plt.title(str(ii))
        #    plt.pause(5)

        if draw_room:
            for ii in range(0, self.num_room):
                for jj in range(0,self.num_node):
                    linecolor = (self.prob_node_room[jj,ii],0,0)
                    plt.setp(plt_node[jj], mfc=linecolor, c=linecolor)
                plt.title(self.room_classes[ii])
                plt.pause(10)


        plt.show()

    # TODO: here and fcn_scoring
    def update_nodeclass(self): # update probability
        prob = np.ones((self.num_node, self.num_room)) # initialize
        # compute
        for ii in range(0, self.num_obj):
            prob_room = np.zeros((self.num_node, self.num_room))
            for jj in range(0, self.num_room):
                print(self.room_classes[jj]+" "+self.obj_class[ii])
                prob_room[:, jj:jj+1] = self.fcn_scoring(self.dist_mat[:,self.obj_node[ii]:self.obj_node[ii]+1],
                                                     self.score_room_obj[(self.room_classes[jj],self.obj_class[ii])])
            prob = np.multiply(prob, prob_room)

            for kk in range(0, self.num_node):
                plt.plot(prob_room[kk, :])

        # normalize
        for ii in range(0, self.num_node):
            plt.plot(self.prob[ii,:])
        self.prob_node_room = np.copy(np.divide(prob, np.tile(np.sum(prob, axis=1, keepdims=True), (1, self.num_room))))
        print("end:update node_class")

    def fcn_scoring(self, dist, conf): # distance between vertices, confidence level
        # dist, conf: integer or numpy array
        score = np.exp(-dist/self.CONST_DIST)*conf
        return score



if __name__ == '__main__':
    topomap = TopoMap()

    obj_pose = {0:[1,1], 1:[1,1.5], 2:[0.5,0.5], 3:[3,3], 4:[4,3.5], 5:[8,8]}
    obj_class = {0:"fridge", 1:"table", 2:"food", 3:"sofa", 4: "coffee_table", 5: "balls"} # ID -class
    topomap.update_obj(obj_pose, obj_class)
    topomap.update_nodeclass()
    topomap.draw_map(draw_room = True)



    print("end?")
    #fig=spatial.voronoi_plot_2d(vor)
    #plt.show(fig)




