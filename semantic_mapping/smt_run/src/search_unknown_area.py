#!/usr/bin/python

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2

from smt_run.msg import FloatArray
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def _numpy_to_multiarray(np_array):
    multiarray = Float32MultiArray()
    multiarray.layout.dim = \
        [MultiArrayDimension(
            'dim%d' % i,
            np_array.shape[i],
            np_array.shape[i] * np_array.dtype.itemsize)
         for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape([1, -1])[0].tolist()
    return multiarray


class Search_unknown_class():
    def __init__(self):
        self.top_range = [2.2, 2.5]  # min, max
        self.bottom_range = [0.2, 1.0]

        self.voxel_size = 0.05
        map_size = np.array([20, 20, self.voxel_size])
        map_xyz = map(int, map_size/self.voxel_size)
        self.map_offset = np.array([10, 10, 0])/self.voxel_size
        self.map = np.empty(map_xyz, list)

        # Subscribe
        self.map_pointcloud = PointCloud2()
        rospy.Subscriber("/hdl_graph_slam/map_points", PointCloud2,
                         self.pointcloud_callback)

        # Publish
        self.point_pub = rospy.Publisher(
            "/smt_run/points", PointCloud2, queue_size=3)
        self.unknown_area_pub = rospy.Publisher(
            "/smt_run/unknown_area", FloatArray, queue_size=1)

        rospy.loginfo("Start search unknown area algorithm..")

    def pointcloud_callback(self, pointcloud_data):
        self.map_pointcloud = pointcloud_data

    def find_unknown_area(self):
        if self.map_pointcloud == PointCloud2():
            return

        point_cloud = np.array(
            pc2.read_points_list(self.map_pointcloud,
                                 skip_nans=True,
                                 field_names=("x", "y", "z")))

        check_point = [0, -1, 1]
        for point in point_cloud:
            point_z = point[2]
            if point_z < self.bottom_range[0] or\
                    point_z > self.top_range[1]:
                continue
            voxel = map(int, point / self.voxel_size + self.map_offset)
            if point_z < self.bottom_range[1]:
                self.map[voxel[0]][voxel[1]][0] = 1
            elif self.top_range[0] < point_z:
                for i in check_point:
                    for j in check_point:
                        self.map[voxel[0]+i][voxel[1]+j][0] = None

        diff_voxels = []
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j][0]:
                    diff_voxels.append(
                        np.array(([i, j, 0]-self.map_offset)*self.voxel_size))

        self.point_pub.publish(pc2.create_cloud_xyz32(
            self.map_pointcloud.header, diff_voxels))

        # Clustering
        dbscan_clustering = DBSCAN(eps=0.2,
                                   min_samples=8).fit(diff_voxels)
        dbscan_result = dbscan_clustering.labels_

        max_label = max(dbscan_result)
        if max_label == -1:
            return

        voxel_group = [np.empty((0, 3))]*(max_label+1)
        for i, xyz in enumerate(diff_voxels):
            label_value = dbscan_result[i]
            if label_value != -1:  # "-1" value is noisy data
                voxel_np = np.array(xyz).reshape(1, 3)
                voxel_group[label_value] = np.concatenate(
                    (voxel_group[label_value], voxel_np))

        pub_list = FloatArray()
        for i in voxel_group:
            multiarray = _numpy_to_multiarray(i)
            pub_list.lists.append(multiarray)
        self.unknown_area_pub.publish(pub_list)


def main():
    search_area_class = Search_unknown_class()

    # rate = rospy.Rate(0.2)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        search_area_class.find_unknown_area()
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('search_unknown_area')
    main()
