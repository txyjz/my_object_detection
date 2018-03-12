import torch.nn as nn
import torch.nn.functional as F
import abc
import torch

from model.faster_rcnn.config import cfg
from torch.autograd import Variable
from model.rpn.proposal_target_layer import ProposalTargetLayer
from model.rpn.region_proposal_network import RegionProposalNetwork
from core.roi_pooling.modules.roi_pool import _RoIPooling
from core.roi_align.modules.roi_align import RoIAlignAvg
from model.utils.net_utils import _smooth_l1_loss


class _FasterRCNN(nn.Module, metaclass=abc.ABCMeta):
    """base class of faster_rcnn."""
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        
        # define rpn
        self.RCNN_rpn = RegionProposalNetwork(self.base_out_channels)
        self.RCNN_proposal_target = ProposalTargetLayer(n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
    
    def forward(self, im_data, im_shape, gt_boxes):
        batch_size = im_data.size(0)
        
        gt_boxes = gt_boxes.data
        
        # feed image data to base model to obtain base feature map
        feat = self.RCNN_base(im_data)
        
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_reg = self.RCNN_rpn(feat, im_shape, gt_boxes)
        
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        
        rois = Variable(rois)
        
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(feat, rois.view(-1, 5))
        else:
            raise ValueError("Only `align` and `pool` RoIPooling supported.")
        
        # feed pooled features to top model
        pooled_feat_flat = pooled_feat.view(pooled_feat.size(0), -1)
        fc7 = self.RCNN_top(pooled_feat_flat)

        # bbox deltas
        bbox_pred = self.RCNN_bbox_pred(fc7)
        
        if self.training:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
            
        # object classification probability
        cls_score = self.RCNN_cls_score(fc7)
        cls_prob = F.softmax(cls_score)
        
        rcnn_loss_cls = 0
        rcnn_loss_reg = 0
        
        if self.training:
            # classification loss
            rcnn_loss_cls = F.cross_entropy(cls_score, rois_label)
            
            # bounding box regression L1 loss
            rcnn_loss_reg = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        
        return (rois, cls_prob, bbox_pred, rpn_loss_cls,
                rpn_loss_reg, rcnn_loss_cls, rcnn_loss_reg, rois_label)
    
    def _init_weight(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
    
    @abc.abstractmethod
    def _init_module(self):
        pass
    
    def set(self):
        self._init_module()
        self._init_weight()
