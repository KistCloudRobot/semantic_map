#!/usr/bin/python
import rospy
import random
import std_msgs.msg
import numpy as np

from visualization_msgs.msg import MarkerArray, Marker


class Viewer_class():
    def __init__(self, base_frame='/map', class_names={}):
        self.class_names = class_names
        self.ID = 0

        self.voxel_size = 0.05
        self.map_offset = np.array([10, 10, 1.0])/self.voxel_size

        self.header = std_msgs.msg.Header()
        self.header.frame_id = base_frame

        self.class_color = {}
        for _class in class_names:
            r = random.randint(0, 255) / 255.0
            g = random.randint(0, 255) / 255.0
            b = random.randint(0, 255) / 255.0
            self.class_color[_class] = [r, g, b]

        self.random_color = {}
        for i in range(20):
            r = random.randint(0, 255) / 255.0
            g = random.randint(0, 255) / 255.0
            b = random.randint(0, 255) / 255.0
            self.random_color[i] = [r, g, b]

        # Publish data
        self.pub_objs = rospy.Publisher(
            "/smt_run/markerarray", MarkerArray, queue_size=3)

    def del_markerarray(self):
        del_msg_markerarray = MarkerArray()
        del_msg_marker = Marker()
        self.header.stamp = rospy.Time.now()
        del_msg_marker.header = self.header
        del_msg_marker.action = Marker.DELETEALL
        del_msg_markerarray.markers.append(del_msg_marker)

        self.pub_objs.publish(del_msg_markerarray)

    def view_objs(self, obj_data):
        self.del_markerarray()
        msg_markerarray = MarkerArray()
        self.ID = 0
        for obj in obj_data:
            obj_class = obj["class"]
            for voxel in obj["voxel"]:
                voxel_point = (voxel - self.map_offset)*self.voxel_size
                marker = self.msg_marker_voxel(obj_class, voxel_point)
                msg_markerarray.markers.append(marker)
        self.pub_objs.publish(msg_markerarray)

    def msg_marker_voxel(self, obj_class, voxel_point):
        msg_marker = Marker()
        # header
        self.header.stamp = rospy.Time.now()
        msg_marker.header = self.header
        # color
        marker_color = self.class_color[obj_class]
        msg_marker.color.r = marker_color[0]
        msg_marker.color.g = marker_color[1]
        msg_marker.color.b = marker_color[2]
        msg_marker.color.a = 1
        # pose
        msg_marker.pose.position.x = voxel_point[0]
        msg_marker.pose.position.y = voxel_point[1]
        msg_marker.pose.position.z = voxel_point[2]
        msg_marker.pose.orientation.w = 1
        # scale
        msg_marker.scale.x = 0.05
        msg_marker.scale.y = 0.05
        msg_marker.scale.z = 0.05
        # etc
        msg_marker.ns = obj_class
        msg_marker.id = self.ID
        msg_marker.type = Marker.CUBE
        msg_marker.action = Marker.ADD
        self.ID += 1

        return msg_marker
