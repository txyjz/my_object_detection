import argparse
import pprint
import os
import cv2
import torch
import numpy as np

from model.faster_rcnn.config import cfg
from scipy.misc import imread
from model.faster_rcnn.vgg.vgg16 import vgg16
from torch.autograd import Variable
from core.preprocess import get_image_blob
from core.box_ops import bbox_transform_inv
from core.box_ops import clip_to_image
from core.nms.nms_wrapper import nms
from model.utils.net_utils import vis_detections


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_78cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load model',
                        default="data")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=625, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


np.random.seed(cfg.RNG_SEED)

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
    args = parse_args()
    print('Using config:')
    pprint.pprint(cfg)
    
    input_dir = os.path.join(args.load_dir, args.net, args.dataset)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(
        input_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    
    pascal_classes = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])
    # initilize the network here.
    
    faster_rcnn = vgg16(21)
    faster_rcnn.set()
    print("==> load checkpoint {}".format(load_name))
    checkpoint = torch.load(load_name)
    faster_rcnn.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    
    print('load model successfully!')
    print("load checkpoint %s" % (load_name))
    
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_shape = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)
    
    # ship to cuda
    args.cuda = True
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_shape = im_shape.cuda()
        gt_boxes = gt_boxes.cuda()
    
    im_data = Variable(im_data, volatile=True)
    im_shape = Variable(im_shape, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)
    
    if args.cuda > 0:
        cfg.CUDA = True
        faster_rcnn.cuda()
    
    max_per_image = 100
    thresh = 0.05
    vis = True
    
    faster_rcnn.eval()
    
    img_list = os.listdir(args.image_dir)
    n_imgs = len(img_list)
    
    print('Loaded Photo: {} images.'.format(n_imgs))
    
    for img in img_list:
        im_file = os.path.join(args.image_dir, img)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb->bgr
        im = im_in[:, :, ::-1]
        blobs, im_scales = get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_shape_np = np.array([[im_blob.shape[1], im_blob.shape[2]]])
        
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_shape_pt = torch.from_numpy(im_shape_np)
        
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        im_shape.data.resize_(im_shape_pt.size()).copy_(im_shape_pt)
        
        (rois, cls_prob,
         bbox_pred, rpn_loss_cls,
         rpn_loss_box, rcnn_loss_cls,
         rcnn_loss_bbox, rois_label) = faster_rcnn(im_data, im_shape, gt_boxes)
        
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
    
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_to_image(pred_boxes, im_shape, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        
        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        
        if vis:
            im2show = np.copy(im)
        for i in range(1, len(pascal_classes)):
            idx = torch.nonzero(scores[:, i] > thresh).view(-1)
            # if detected:
            if idx.numel() > 0:
                print("detected.")
                cls_scores = scores[:, i][idx]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[idx][:, i * 4:i * 4 + 4]
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, pascal_classes[i], cls_dets.cpu().numpy(), 0.5)
        
        if vis:
            result_path = os.path.join(args.image_dir, img[:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
