#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

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

class object_finder:
    def __init__(self,object_seq):
        self.objectSeq =object_seq
        assert(type(object_seq)==list)
        self.valid_cls=['person','bottle','chair','tvmonitor']
        self.curent_seq_idx =0
        self.curret_object=self.objectSeq[self.curent_seq_idx]
    def next_obj(self):
        if self.curent_seq_idx<len(self.objectSeq)-1:
            self.curent_seq_idx+=1
    def get_curentSeq_closed_obj_coods(self,det_dict):
        dets=self.get_current_obj_dets(det_dict)
        return self.find_closest_obj(dets)
    def get_current_obj_dets(self,det_dict):
        return det_dict[self.curret_object]
    def find_closest_obj(self,dets):
        location =None
        magnitude =None
        if dets !=None:
            num_obj=len(dets)
            print ('{0} of objects'.format(num_obj))
            print dets.shape
            #print dets[i][0],dets[i][2],dets[i][1],dets[i][3]
            areas=[((dets[i][2]-dets[i][0])*(dets[i][3]-dets[i][1])) for i in range(num_obj)]
            inds =sorted(range(len(areas)), key=areas.__getitem__)
            #print areas
            #print inds
            largest_det =dets[inds[-1]]
            center_cx =largest_det[0] + (largest_det[2]-largest_det[0])/2
            center_cy =largest_det[1] + (largest_det[3]-largest_det[1])/2


            magnitude =areas[inds[-1]]
            location  =(int(center_cx-320), int(center_cy-240))

        return location, magnitude

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print("len(inds)==0")
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
	
	
def vis_opencv_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) > 0:
        for i in inds:
            bbox = dets[i, :4].astype(int)
            score = dets[i, -1]
            cv2.rectangle(im, (bbox[0],  bbox[1]), ( bbox[2],bbox[3]), (0,0,255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(bbox[0], bbox[1] - 2), font, 0.5,(255,255,255),1)
    return inds

def demo_video(net,v_f):
    v_file=os.path.join(cfg.DATA_DIR,'demo',v_f)
    cap = cv2.VideoCapture(v_file)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    det_dict ={'person':None,'bottle':None,'chair':None,'tvmonitor':None}
    object_finding_seq=['person','bottle','chair']
    obj_finder =object_finder(object_finding_seq)
    while(cap.isOpened()):
        ret, frame = cap.read()

        scores, boxes = im_detect(net, frame)
        #cv2.imshow('frame',frame)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        #im = np.zeros((512,512,3), np.uint8)
            #vis_opencv_detections(frame, cls, dets, thresh=CONF_THRESH)
            if cls in det_dict.keys():
                inds=vis_opencv_detections(frame, cls, dets, thresh=CONF_THRESH)
                print cls, len(inds)
                if len(inds) >0:
                    det_dict[cls]=dets[inds]

        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
    #display the original image with the centroid drawn on the image
        location, magnitude=obj_finder.get_curentSeq_closed_obj_coods(det_dict)
        print location
        if location!=None:
            cx=location[0]
            cy=location[1]
            cv2.circle(frame, (cx+320,cy+240), 10, (0,255,0), -1)
            print (cx,cy)
        cv2.imshow("processing result", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    print type(im)

    #cv2.imshow("processing result", im)

    #print("drawing img")
    #waitKey() is necessary for making all the cv2.imshow() commands work
    #cv2.waitKey(0)

    #return 1, 1

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #print(scores)
    #print(boxes)
    #
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    #im = np.zeros((512,512,3), np.uint8)
        vis_opencv_detections(im, cls, dets, thresh=CONF_THRESH)
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
    #display the original image with the centroid drawn on the image
    cv2.imshow("processing result", im)

    print("drawing img")
    #waitKey() is necessary for making all the cv2.imshow() commands work
    cv2.waitKey(0)

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
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    print prototxt
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])
    print caffemodel

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    #args.cpu_mode =True

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['coke.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
    v_name='MVI_0021.MOV'
    demo_video(net, v_name)

    #plt.show()
