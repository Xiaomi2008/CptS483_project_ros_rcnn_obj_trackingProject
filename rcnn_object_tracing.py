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
Mag_stop_size =8000000 # size of object  to stop move toward to
def vis_opencv_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return inds
    red =(0,0,255)   # box color =Red
    blue=(0,255,25)   # text color =Blue
    for i in inds:
        bbox = dets[i, :4].astype(int)
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0],  bbox[1]), ( bbox[2],bbox[3]), red, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(bbox[0], bbox[1] - 2), font, 0.7,blue,1)
    return inds




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


# def vis_detections(im, class_name, dets, thresh=0.5):
#     """Draw detected bounding boxes."""
#     inds = np.where(dets[:, -1] >= thresh)[0]
#     if len(inds) == 0:
#         return

#     for i in inds:
#         bbox = dets[i, :4]
#         score = dets[i, -1]
# 	cv2.rectangle(im, (bbox[0],  bbox[1]), ( bbox[2],bbox[3]), (0,0,255), 2)
# 	font = cv2.FONT_HERSHEY_SIMPLEX
# 	cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(bbox[0], bbox[1] - 2), font, 0.5,(0,0,255),2)

def process_frame(im):
    global Net
    if Net == 0:
        Net=init_rcnn()

    timer = Timer()
    timer.tic()
    # scores shape =(N, 21) , wherer N is number of proposed regions, 21 is the number of class
    # boxs  =(N,21*4) , every box has  4 coodinates (Left,top, righ, bottom)
    scores, boxes = im_detect(Net, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3

    # Only detect 4 object classes in robot projects
    det_dict ={'person':None,'bottle':None,'chair':None,'tvmonitor':None}


    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if cls in det_dict.keys():
            inds=vis_opencv_detections(im, cls, dets, thresh=CONF_THRESH)
            if len(inds) >0:
                det_dict[cls]=dets[inds,:]

    # return the coodinates and probabilities of all objects in 4 classes
    return det_dict

class object_finder:
    def __init__(self,object_seq):
        self.objectSeq =object_seq
        assert(type(object_seq)==list)
        self.valid_cls=['person','bottle','chair','tvmonitor']
        self.curent_seq_idx =0
        self.curret_object=self.objectSeq[ self.curent_seq_idx]
    def next_obj(self):
        if self.curent_seq_idx<len(self.objectSeq)-1:
            self.curent_seq_idx+=1
            self.curret_object=self.objectSeq[ self.curent_seq_idx]
    def get_curentSeq_closed_obj_coods(self,det_dict):
        dets=self.get_current_obj_dets(det_dict)
        return self.find_closest_obj(dets)
    def get_current_obj_dets(self,det_dict):
        return det_dict[self.curret_object]
    def get_current_tracking_obj(self):
        return self.objectSeq[ self.curent_seq_idx]
    def find_closest_obj(self,dets):
        location =None
        magnitude =None
        if dets !=None:
            num_obj=len(dets)
            areas=[(dets[i][2]-dets[i][0])*(dets[i][3]-dets[i][1]) for i in range(num_obj)]
            inds =sorted(range(len(areas)), key=areas.__getitem__)
            largest_det =dets[inds[-1]]
            center_cx =largest_det[0] + (largest_det[2]-largest_det[0])/2
            center_cy =largest_det[1] + (largest_det[3]-largest_det[1])/2

            magnitude =areas[inds[-1]]
            location  =(int(center_cx)-320, int(center_cy)-240)
        return location, magnitude




class Node:
    def __init__(self):
        #register a subscriber callback that receives images
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        self.done =False
        #create a publisher for sending commands to turtlebot
        self.movement_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.state = "TRACKING"
        #self.odom_sub = rospy.Subscriber("odom", Odometry, self.Position)
        self.wheel_r = 0.07
        self.wheel_axis_l =0.23
        object_finding_seq=['person']
        self.obj_finder =object_finder(object_finding_seq)
        self.previous_location =None
        self.previous_mag      =None
        self.frame_count       =0

    #this function wll get called every time a new image comes in
    #all logic occurs in this function
    def image_callback(self, ros_image):
        # convert the ros image to a format openCV can use
        cv_image = np.asarray(ToOpenCV(ros_image))
        #print (cv_image.shape)
        #location, magnitude=process_frame(cv_image)
        det_dicts =process_frame(cv_image)
        location, magnitude=self.obj_finder.get_curentSeq_closed_obj_coods(det_dicts)
        if location!=None:
            cv2.circle(cv_image, (location[0]+320,location[1]+240), 10, (0,255,0), -1)
        cv2.imshow("processing result", cv_image)
        cv2.waitKey(1)

        ###########
        # Insert turtlebot controlling logic here!
        ###########
        cmd = Twist()

        #here we have a little state machine for controlling the turtlebot
        #possible states:
        #TRACKING: we are currently moving towards an object
        #print the current state
        #rospy.loginfo("state: {}".format(self.state))

        #self.reaction_controller(location, magnitude)

        #return

        if self.state == "TRACKING":
            #check if we can't see an object
            if(location ==None and self.previous_location ==None):
                #if we can't see an object, then we should search for one
                self.state = "SEARCHING"
                return
            elif (location ==None and self.previous_location !=None):
                location =self.previous_location
            #else...



            self.previous_location =location

            self.frame_count  +=1
            if self.frame_count  >60:
                self.frame_count=0
                self.previous_location =None
            #go forward
            cmd.linear.x = 0.15

            #this is a basic proportional controller, where the magnitude of
            #rotation is based on how far away the object is from center
            #we apply a 10px hysteresis for smoothing
            #print(location)
            if(location[0] > 10):
                cmd.angular.z = -0.002 * location[0]
            elif(location[0] < 10):
                cmd.angular.z = -0.002 * location[0]

            #check if we are close to the object
            obj_mag = 0
            if self.obj_finder.get_current_tracking_obj() == 'bottle':
                obj_mag = 13000
            elif self.obj_finder.get_current_tracking_obj() == 'person':
                obj_mag = 125000
            else:
                obj_mag = 75000
            print ('{}: {}'.format(self.obj_finder.get_current_tracking_obj(), magnitude))
            if magnitude > obj_mag:
                #calculate a time 3 seconds into the future
                #we will rotate for this period of time
                self.rotate_expire = rospy.Time.now() + rospy.Duration.from_sec(3)
                #set state to rotating
                self.state = "SEARCHING"
                self.obj_finder.next_obj()
                self.previous_location =None

        # elif self.state == "ROTATING_AWAY":
        #     #check if we are done rotating
        #     if rospy.Time.now() < self.rotate_expire:
        #         cmd.angular.z = -0.5
        #     else: #after we have rotated away, search for new target
        #         self.state = "SEARCHING"

        elif self.state == "SEARCHING":
            #here we just spin  until we see an object
            cmd.angular.z = -0.5
            if location is not None:
                #when we see an object, start tracking it!
                self.state = "TRACKING"

        #this state is currently unused, but we could use it for exiting
        elif self.state == "STOP":
            rospy.signal_shutdown("done tracking, time to exit!")





        #publish command to the turtlebot
        self.movement_pub.publish(cmd)

    def reaction_controller(self,location, magnitude):
        if location == None:
            theta = PI/100
            self.turn(theta,0.15)
        #elif magnitude==0:
        #elif magnitude <Mag_stop_size and location != None:
            #print location
            #print magnitude

         #theta = location[0]/30 * PI/8;



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
                        choices=NETS.keys(), default='vgg16')

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
    Net =0
    rospy.init_node("video_tracking")
    node = Node()
    #this function loops and returns when the node shuts down
    #all logic should occur in the callback function
    rospy.spin()
