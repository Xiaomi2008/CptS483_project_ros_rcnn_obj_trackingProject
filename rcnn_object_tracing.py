#!/usr/bin/env python
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
import numpy as np
import roslib
import tf
import PyKDL as kdl
from intro_to_robotics.image_converter import ToOpenCV, depthToOpenCV

global Net
Net = 0
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

# Messages
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

#roslib.load_manifest('odom_publisher')
PI=3.14159265359
Mag_stop_size =8000000 # size of fire hydrant
def vis_opencv_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return


    for i in inds:
        bbox = dets[i, :4].astype(int)
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0],  bbox[1]), ( bbox[2],bbox[3]), (0,0,255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(bbox[0], bbox[1] - 2), font, 0.5,(255,255,255),1)


# def demo_video(net,v_f):
#     v_file=os.path.join(cfg.DATA_DIR,'demo',v_f)
#     cap = cv2.VideoCapture(v_file)
#     CONF_THRESH = 0.8
#     NMS_THRESH = 0.3
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#
#         scores, boxes = im_detect(net, frame)
#         #cv2.imshow('frame',frame)
#         for cls_ind, cls in enumerate(CLASSES[1:]):
#             cls_ind += 1 # because we skipped background
#             cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#             cls_scores = scores[:, cls_ind]
#             dets = np.hstack((cls_boxes,
#                               cls_scores[:, np.newaxis])).astype(np.float32)
#             keep = nms(dets, NMS_THRESH)
#             dets = dets[keep, :]
#         #im = np.zeros((512,512,3), np.uint8)
#             vis_opencv_detections(frame, cls, dets, thresh=CONF_THRESH)
#         #vis_detections(im, cls, dets, thresh=CONF_THRESH)
#     #display the original image with the centroid drawn on the image
#         cv2.imshow("processing result", frame)

#this function does our image processing
#returns the location and "size" of the detected object
def process_image(image):
    #convert color space from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #create bounds for our color filter
    lower_bound = np.array([0, 10, 10])
    upper_bound = np.array([10,255,255])

    #execute the color filter, returns a binary black/white image
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    #display the results of the color filter
    cv2.imshow("image_mask", mask)

    #calculate the centroid of the results of the color filer
    M = cv2.moments(mask)
    location = None
    magnitude = 0
    if M['m00'] > 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        magnitude = M['m00']
        location = (cx-320, cy-240) #scale so that 0,0 is center of screen
        #draw a circle image where we detected the centroid of the object
        cv2.circle(image, (cx,cy), 3, (0,0,255), -1)

    #display the original image with the centroid drawn on the image
    cv2.imshow("processing result", image)

    #waitKey() is necessary for making all the cv2.imshow() commands work
    cv2.waitKey(1)
    return location, magnitude


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return




    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
	cv2.rectangle(im, (bbox[0],  bbox[1]), ( bbox[2],bbox[3]), (0,0,255), 2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(bbox[0], bbox[1] - 2), font, 0.5,(0,0,255),2)
	#display the original image with the centroid drawn on the image
    #cv2.imshow("processing result", im)

    #waitKey() is necessary for making all the cv2.imshow() commands work
    #cv2.waitKey(1)

        # ax.add_patch(
            # plt.Rectangle((bbox[0], bbox[1]),
                          # bbox[2] - bbox[0],
                          # bbox[3] - bbox[1], fill=False,
                          # edgecolor='red', linewidth=3.5)
            # )
        # ax.text(bbox[0], bbox[1] - 2,
                # '{:s} {:.3f}'.format(class_name, score),
                # bbox=dict(facecolor='blue', alpha=0.5),
                # fontsize=14, color='white')

    # ax.set_title(('{} detections with '
                  # 'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  # thresh),
                  # fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
def process_frame(im):
    global Net
	# scores shape =(N, 21) , wherer N is number of proposed regions, 21 is the number of class
	# boxs  =(N,21*4) , every box has  4 coodinates (Left,top, righ, bottom)
    if Net == 0:
        Net=init_rcnn()
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', '001763.jpg')
    #im = cv2.imread(im_file)
    #scores, boxes = im_detect(net, im)
    #print type(im)
    #print (im.shape)
    #print (net)
    #cv2.imshow("processing result", im)
    #cv2.waitKey(1)
    #return

    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(Net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #scores, boxes = im_detect(net, im)
    #vis_opencv_detections()
	 # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
	#cv2.imshow("processing result", image)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_opencv_detections(im, cls, dets, thresh=CONF_THRESH)
    cv2.imshow("processing result", im)
    cv2.waitKey(1)

class Node:
    def __init__(self):
        #register a subscriber callback that receives images
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        self.done =False
        #create a publisher for sending commands to turtlebot
        self.movement_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)

        #self.odom_sub = rospy.Subscriber("odom", Odometry, self.Position)
        self.wheel_r = 0.07
        self.wheel_axis_l =0.23

    #this function wll get called every time a new image comes in
    #all logic occurs in this function
    def image_callback(self, ros_image):
        # convert the ros image to a format openCV can use
        cv_image = np.asarray(ToOpenCV(ros_image))
        process_frame(cv_image)
        return

        #run our vision processing algorithm to pick out the object
        #returns the location (x,y) of the object on the screen, and the
        #"size" of the discovered object. Size can be used to estimate
        #distance
        #None/0 is returned if no object is seen


        #location, magnitude = process_frame(cv_image)


        #print magnitude

        #log the processing results

        #rospy.logdebug("image location: {}\tmagnitude: {}".format(location, magnitude))
        #self.reaction_controller(location,magnitude)




            #print location
            #print magnitude

        ###########
        # Insert turtlebot controlling logic here!
        ###########
        #cmd = Twist()


        #publish command to the turtlebot
        #self.movement_pub.publish(cmd)

    def reaction_controller(self,location, magnitude):
        if location == None:
            theta = PI/100
            self.turn(theta,0.15)

        elif magnitude <Mag_stop_size and location != None:
            #print location
            #print magnitude

            theta = location[0]/30 * PI/8;



            if abs(theta) >PI/8:
                #print 'thetha is {}'.format(theta)
                if theta>0:
                    print 'Object in on the Right'
                else:
                    print 'Object is on the Left'
                self.turn(theta/32,0.1)
                return
            else:
                print 'Object is on the Center'
                self.goFoward(0.05)

    def spinwheels(self, u1,u2,time):
        #r=0.2  # diameter of wheel, asuming left and right are the same
        #l=1
        linear_val=(self.wheel_r/2)*(u1+u2);
        ang_val=self.wheel_r/(2*self.wheel_axis_l)*(u1-u2);
        twsit_msg =Twist()
        twsit_msg.linear.x =linear_val
        twsit_msg.angular.z=ang_val
        rate=40
        r = rospy.Rate(rate)
        count =0
        finished =False
        total_count =time*rate
        #print time
        while not finished:
            #self.cmd_vel.publish(move_cmd)
            self.movement_pub.publish(twsit_msg)
            count=count+1
            r.sleep()
            if count >total_count:
                finished = True
        twsit_stop_msg =Twist()
        twsit_msg.linear.x =linear_val
        twsit_msg.angular.z=ang_val
        self.movement_pub.publish(twsit_msg)
        self.done =True
        self.stop_turtlebot()

    def stop_turtlebot(self):
        twsit_msg =Twist()
        twsit_msg.linear.x =0
        twsit_msg.angular.z=0
        self.movement_pub.publish(twsit_msg)
    def goFoward(self,distance):
        v=0.2
        time =distance/v
        u1=v/self.wheel_r
        u2=u1
        self.spinwheels(u1,u2,time)
    def turn(self, degree, speed):
        if degree >0:
           D = 1
        else:
           D = -1

        v=(PI)*D*speed
        time=degree*0.2/v
        #u1=0

        u2=v*(self.wheel_axis_l/self.wheel_r)
        u1=-u2
        self.spinwheels(u1,u2,time)
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    #parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
    #                    choices=NETS.keys(), default='vgg16')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args
def init_rcnn():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #print prototxt
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))


    #if args.cpu_mode:
    #    caffe.set_mode_cpu()
    #else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_file = os.path.join(cfg.DATA_DIR, 'demo', '001763.jpg')
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    return net

if __name__ == "__main__":
    net = None
    rospy.init_node("video_tracking")
    node = Node()
    #this function loops and returns when the node shuts down
    #all logic should occur in the callback function
    rospy.spin()
