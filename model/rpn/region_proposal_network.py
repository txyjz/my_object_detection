"""
One implementation of Region Proposal Network described in:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from core import box_ops
from core.generate_anchorsCP import AnchorGenerator
from .proposal_layer import ProposalLayer
from model.faster_rcnn.config import cfg
from model.rpn.proposal_target_layer import ProposalTargetLayer
from model.rpn.anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

class RegionProposalNetwork(nn.Module):
    """
    The implementation of Region Proposal Network.

    Args:
        in_channels:
        out_channels:
        aspect_ratios:
        anchor_scales:

    Returns:

    """
    
    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 sliding_window_size=3):
        super().__init__()
        
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.num_classes = 21
        
        self.RPN_Conv = nn.Conv2d(in_channels,
                                  mid_channels,
                                  sliding_window_size,
                                  stride=1, padding=1)
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self.cls_out_channels = self.num_anchors * 2
        self.reg_out_channels = self.num_anchors * 4
        self.RPN_cls_score = nn.Conv2d(mid_channels,
                                       self.cls_out_channels,
                                       1, stride=1, padding=0)
        self.RPN_bbox_pred = nn.Conv2d(mid_channels,
                                       self.reg_out_channels,
                                       1, stride=1, padding=0)
        self.proposal_layer = ProposalLayer(scales=(8, 16, 32), aspect_ratios=(0.5, 1, 2))
        self.anchor_generator = AnchorGenerator(self.anchor_scales, self.anchor_ratios)
        self.anchor_target_layer = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)
    
    def forward(self, base_feat, im_shape, gt_boxes):
        """
        Args:
            base_feat: with shape [N, C, H, W]
        """
        batch_size, _, h, w = base_feat.shape
        k = self.num_anchors
        
        # with shape [N, C, H, W]
        mid_feat = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # with shape [N, 2k, H, W]
        scores = self.RPN_cls_score(mid_feat)
        scores_reshape = self.reshape(scores, 2)
        rpn_cls_prob = F.softmax(scores_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob, self.cls_out_channels)
        
        # with shape [N, 4k, H, w]
        box_deltas = self.RPN_bbox_pred(mid_feat)
        
        input = (rpn_cls_prob.data, box_deltas.data)
        rois = self.proposal_layer(input, im_shape, h, w)
        
        self.rpn_loss_cls = 0
        self.rpn_loss_reg = 0
        
        if self.training:
            rpn_data = self.anchor_target_layer((scores.data, gt_boxes, im_shape))
    
            # compute classification loss
            rpn_cls_score = scores_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)
    
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))
        
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
    
            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
    
            self.rpn_loss_reg = _smooth_l1_loss(box_deltas, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
        
        return rois, self.rpn_loss_cls, self.rpn_loss_reg
    
    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    
    t = torch.randn(4, 4)
    v = Variable(t)
