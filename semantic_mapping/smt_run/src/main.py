#!/usr/bin/python
import rospy
# import pickle
import message_filters

from sensor_msgs.msg import Joy, Image
from nav_msgs.msg import Odometry

from viewer import Viewer_class
from detect_obj import Detect_class

from timeit import default_timer as timer

coco_class = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

is_update = False
button_update = False
image_data = Image()
depth_data = Image()
pose_estimate = Odometry()


def callback(depth_sub, image_sub, localization_sub):
    global depth_data, image_data, pose_estimate, is_update
    depth_data = depth_sub
    image_data = image_sub
    pose_estimate = localization_sub
    is_update = True


def callback_joy(data):
    global button_update
    joy_data = data.buttons
    if not button_update:
        if joy_data[1] == 1:
            button_update = True


def main():
    global is_update, button_update

    detect_class = Detect_class(base_link='/base_link',
                                camera_link='/camera_rgb_optical_frame',
                                class_names=coco_class)
    coco_class.append('unknown')
    viewer_class = Viewer_class(base_frame='/map', class_names=coco_class)

    # Subscriber data
    image_sub = message_filters.Subscriber(
        "/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber(
        "/camera/depth/image_raw", Image)
    localization_sub = message_filters.Subscriber(
        "/odom", Odometry)

    # image, depth, localization
    subs = message_filters.ApproximateTimeSynchronizer(
        [depth_sub, image_sub, localization_sub], 1, 0.05)
    subs.registerCallback(callback)

    # joystick data
    rospy.Subscriber("/bluetooth_teleop/joy", Joy, callback_joy)

    pre_obj = []
    while not rospy.is_shutdown():
        if button_update and is_update:
            start_time = timer()
            dt_obj = detect_class.detect_object(
                image_data, depth_data, pose_estimate)
            end_time = timer()
            detect_time = end_time - start_time

            start_time = timer()
            pre_obj = detect_class.update_data(dt_obj, pre_obj)
            end_time = timer()
            update_time = end_time - start_time

            rospy.loginfo('Detect :     %s', detect_time)
            rospy.loginfo('Update :     %s', update_time)
            print('------------------------------------')

            # unknown_area = []
            # unknown_area = search_area_class.find_unknown_area(point_list)

            viewer_class.view_objs(pre_obj)

            button_update = False
            is_update = False

# rate = rospy.Rate(1)
# while not rospy.is_shutdown():
#     bool_obj, detect_objs = detect_class.detect_object()
#     if bool_obj == True:
#         # pre_objs = detect_class.update_data(detect_objs, pre_objs)
#         # total_objs = detect_class.reorganize_objs(pre_objs, unknown_area)
#         viewer_class.del_markerarray()
#         viewer_class.view_objs(detect_objs)
#         rate.sleep()
# bool_area, area = search_area_class.find_unknown_area()
# if bool_area == True:
#     unknown_area = area

# if button_update:
#     with open('./objects.pkl', 'wb') as f:
#         for obj in total_objs:
#             pickle.dump(obj, f)
#     rospy.loginfo("Save objects..")
#     button_update = False
# # Read pickle data..
# data_list = []
# with open('./objects.pkl', 'rb') as f:
#     while True:
#         try:
#             data = pickle.load(f)
#         except EOFError:
#             break
#         data_list.append(data)

# rosrun map_server map_saver -f mymap map:=/projected_map
# rosrun map_server map_server mymap.yaml # be careful 'nan' value


if __name__ == '__main__':
    rospy.init_node('smt_run')
    main()
