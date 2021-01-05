#!/usr/bin/python
import tf
import cv2
import math
import rospy
import numpy as np

from cv_bridge import CvBridge
from scipy.spatial import cKDTree
from sensor_msgs.msg import CameraInfo
from smt_run.msg import FloatArray
from geometry_msgs.msg import TransformStamped
from mask_rcnn_ros.srv import MaskRcnn, MaskRcnnRequest


class Detect_class():
    def __init__(self,
                 base_link='/base_link',
                 camera_link='/camera_rgb_optical_frame',
                 class_names={}):

        self.class_names = class_names
        self.bridge = CvBridge()
        self.voxel_list = []

        # Camera inform
        self.image_scale = 1/3.0
        cam_info_data = rospy.wait_for_message(
            "/camera/color/camera_info", CameraInfo)
        self.image_size = np.array([0, 0])  # width, height
        self.image_size[0] = cam_info_data.width * self.image_scale
        self.image_size[1] = cam_info_data.height * self.image_scale
        self.fx = cam_info_data.K[0] * self.image_scale
        self.fy = cam_info_data.K[4] * self.image_scale
        self.cx = (self.image_size[0]+1)/2
        self.cy = (self.image_size[1]+1)/2
        self.image_center = np.array([self.cx, self.cy])

        # Create voxel map
        map_size = np.array([20, 20, 6.0])
        self.voxel_size = 0.05
        map_xyz = map(int, map_size/self.voxel_size)
        self.map_offset = np.array([10, 10, 1.0])/self.voxel_size
        self.map = np.empty(map_xyz, list)

        # Search unknown area

        # Detect variables
        self.w_threshold = 20 * self.image_scale
        self.obj_threshold = 0.8
        self.dist_threshold = 2
        self.class_threshold = 0.8
        self.update_threshold = 0.3

        # Transformation matrix
        self.tf_ros = tf.TransformerROS()
        self.tf_listener = tf.TransformListener()
        self.tf_map2odom = np.eye(4)
        self.tf_base2cam = self.transformation_matrix(base_link,
                                                      camera_link)

        # Subscriber
        rospy.Subscriber('/hdl_graph_slam/odom2pub',
                         TransformStamped, self.tf_callback)
        rospy.Subscriber('/smt_run/unknown_area',
                         FloatArray, self.unk_callback)

        # Mask Rcnn service
        self.plan_req = MaskRcnnRequest()
        self.mask_srv = rospy.ServiceProxy("/mask_rcnn_service", MaskRcnn)

        rospy.loginfo("Start detect object algorithm..")

    def _multiarray_to_numpy(self, pytype, dtype, multiarray):
        dims = map(lambda x: x.size, multiarray.layout.dim)
        result = np.array(multiarray.data,
                          dtype=pytype).reshape(dims).astype(dtype)
        return result

    def unk_callback(self, data):
        for i in data.lists:
            c = self._multiarray_to_numpy(float, np.float32, i)
            print(c)

    def tf_callback(self, data):
        pose = data.transform.translation
        pose_xyz = [pose.x, pose.y, pose.z]
        quaternion = data.transform.rotation

        self.tf_map2odom = tf.transformations.quaternion_matrix(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.tf_map2odom[0:3, 3] = pose_xyz

    def transformation_matrix(self, frame_A, frame_B):
        while 1:
            try:
                (translation, rotation) = self.tf_listener.lookupTransform(
                    frame_A, frame_B, rospy.Time(0))
                trans_matrix = self.tf_ros.fromTranslationRotation(
                    translation, rotation)
                return trans_matrix
            except tf.Exception:
                pass

    def cal_tf_odom2cam(self, localization):
        pose = localization.pose.position
        pose_xyz = [pose.x, pose.y, pose.z]
        quaternion = localization.pose.orientation
        euler = tf.transformations.euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        tf_map2base = tf.transformations.euler_matrix(
            euler[0], euler[1], euler[2])
        tf_map2base[0:3, 3] = pose_xyz
        return np.dot(tf_map2base, self.tf_base2cam)

    def detect_object(self, image, depth, localization):
        result = []
        # Compute Mask R-CNN data
        self.plan_req.image = image
        cnn_data = self.mask_srv(self.plan_req).result

        # Calculate transformation matrix from the map to camera pose
        tf_odom2cam = self.cal_tf_odom2cam(localization.pose)
        tf_map2cam = np.dot(self.tf_map2odom, tf_odom2cam)

        cv_depth = self.bridge.imgmsg_to_cv2(depth, depth.encoding)
        reshape_cv_depth = cv2.resize(cv_depth, dsize=(0, 0),
                                      fx=self.image_scale,
                                      fy=self.image_scale,
                                      interpolation=cv2.INTER_LINEAR)
        np.nan_to_num(reshape_cv_depth, copy=False)
        for num in range(len(cnn_data.scores)):
            class_score = cnn_data.scores[num]
            if class_score < self.class_threshold:
                continue

            cnn_masks = cnn_data.masks[num]
            cv_mask = self.bridge.imgmsg_to_cv2(cnn_masks)/255.0
            reshape_cv_mask = cv2.resize(cv_mask, dsize=(0, 0),
                                         fx=self.image_scale,
                                         fy=self.image_scale,
                                         interpolation=cv2.INTER_LINEAR)
            depth_image = reshape_cv_depth * reshape_cv_mask
            cv_pixel = np.nonzero(depth_image)
            pixel_size = len(cv_pixel[0])

            if pixel_size == 0:
                continue

            tf_cam2obj = np.eye(4)
            obj_class = cnn_data.class_names[num]
            obj_score = np.full((len(self.class_names), 1),
                                (1 - class_score) / len(self.class_names))
            obj_score[self.class_names.index(obj_class)] = class_score

            # Calculate w1
            kernel = np.ones((7, 7), np.uint8)
            erosion_image = cv2.erode(reshape_cv_mask, kernel, iterations=1)
            w1_image = (reshape_cv_mask + erosion_image)/2

            # Calculate w2
            depth_max = np.max(depth_image)
            depth_min = np.min(depth_image[cv_pixel])
            c2_image = (depth_image - depth_min) / (depth_max - depth_min)
            w2_image = (c2_image-1)*(c2_image-1)*reshape_cv_mask

            # Calculate w3
            cnn_boxes = cnn_data.boxes[num]
            box_pose = np.array(
                [cnn_boxes.x_offset + cnn_boxes.width/2,
                 cnn_boxes.y_offset + cnn_boxes.height/2])*self.image_scale

            c3 = (math.sqrt(np.dot((box_pose - self.image_center),
                                   (box_pose - self.image_center).T))) / \
                (math.sqrt(np.dot((self.image_size - self.image_center),
                                  (self.image_size - self.image_center).T)))
            w3 = -math.pow(c3, 2) + 1
            w3_image = w3*reshape_cv_mask

            w_image = w1_image*w2_image*w3_image

            count_obj = 0.0
            count_mask = 0.0

            # cv2.imshow("Image from my node", w_image)
            # cv2.waitKey(3)
            obj_voxel = []
            for num in range(pixel_size):
                pix_x = cv_pixel[0][num]
                pix_y = cv_pixel[1][num]
                # 2D image pixel to 3D point
                point_cz = depth_image[pix_x][pix_y]
                point_cx = point_cz*(pix_y-self.cx)/self.fx
                point_cy = point_cz*(pix_x-self.cy)/self.fy
                tf_cam2obj[0:3, 3] = [point_cx, point_cy, point_cz]

                # Trasnform camera frame to global(map) frame
                tf_map2obj = np.dot(tf_map2cam, tf_cam2obj)
                voxel = map(int, tf_map2obj[0:3, 3] /
                            self.voxel_size + self.map_offset)

                # map : [class score, w]
                w = w_image[pix_x][pix_y]
                map_voxel = self.map[voxel[0]][voxel[1]][voxel[2]]
                if map_voxel:
                    pre_class_score = map_voxel[0]
                    pre_w = map_voxel[1]
                    self.map[voxel[0]][voxel[1]][voxel[2]][0] = (
                        w*obj_score + pre_w * pre_class_score)/(w+pre_w)
                    self.map[voxel[0]][voxel[1]][voxel[2]][1] = pre_w + w
                else:
                    self.voxel_list.append(voxel)
                    self.map[voxel[0]][voxel[1]][voxel[2]] = [obj_score, w]

                if self.map[voxel[0]][voxel[1]][voxel[2]][1] > \
                        self.w_threshold:
                    count_mask += 1.0
                    if self.class_names[
                        self.map[voxel[0]][voxel[1]][voxel[2]][0].argmax()]\
                            == obj_class:
                        count_obj += 1.0
                        obj_voxel.append(voxel)

            if count_mask != 0 and (count_obj/count_mask) > self.obj_threshold:
                obj_voxel = np.array(obj_voxel)
                obj_voxel = np.unique(obj_voxel, axis=0)
                detect_obj = {}
                detect_obj["class"] = obj_class
                detect_obj["voxel"] = obj_voxel
                obj_point = (obj_voxel - self.map_offset)*self.voxel_size
                detect_obj["mean"] = np.mean(obj_point.T, axis=1)
                result.append(detect_obj)
        return result

    def update_data(self, detect_object, pre_object):
        result = []
        for obj in pre_object:
            pre_voxel = []
            count = 0.0
            for voxel in obj["voxel"]:
                voxel_class = self.class_names[
                    self.map[voxel[0]][voxel[1]][voxel[2]][0].argmax()]
                if voxel_class == obj["class"]:
                    count += 1.0
                    pre_voxel.append(voxel)

            voxel_accuracy = count/len(obj["voxel"])
            if voxel_accuracy > 0.95:
                result.append(obj)
            elif voxel_accuracy > 0.7:
                obj["voxel"] = np.array(pre_voxel)
                result.append(obj)

        for dt_obj in detect_object:
            list2update = []
            up_obj_class = dt_obj["class"]
            for num, pre_obj in enumerate(result):
                d_mean = dt_obj["mean"]
                p_mean = pre_obj["mean"]
                dist_d2p = math.sqrt(
                    np.dot((d_mean - p_mean), (d_mean - p_mean).T))

                if dist_d2p > self.dist_threshold or \
                        pre_obj["class"] != dt_obj["class"]:
                    continue

                # Calculate the relation cost between objects
                d_voxel = dt_obj["voxel"]
                p_voxel = pre_obj["voxel"]
                d_points = (d_voxel - self.map_offset)*self.voxel_size
                p_points = (p_voxel - self.map_offset)*self.voxel_size

                p_dist, _ = cKDTree(d_points).query(p_mean, k=1)
                d_dist, _ = cKDTree(p_points).query(d_mean, k=1)

                dist_model = (p_dist+d_dist)-dist_d2p
                if dist_model > self.update_threshold:
                    continue
                list2update.append([num, pre_obj])

            len_list2update = len(list2update)
            if len_list2update != 0:
                for i in reversed(range(len_list2update)):
                    del result[list2update[i][0]]

                up_obj_voxel = dt_obj["voxel"]
                for i in range(len_list2update):
                    up_obj_voxel = np.concatenate(
                        (up_obj_voxel, list2update[i][1]["voxel"]))
                up_obj_voxel = np.unique(up_obj_voxel, axis=0)

                up_obj_point = (up_obj_voxel - self.map_offset)*self.voxel_size
                up_obj_mean = np.mean(up_obj_point.T, axis=1)

                update_obj = {}
                update_obj["class"] = up_obj_class
                update_obj["voxel"] = up_obj_voxel
                update_obj["mean"] = up_obj_mean
                result.append(update_obj)
            else:
                result.append(dt_obj)
        return result
