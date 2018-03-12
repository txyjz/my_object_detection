"""Proposal layer."""
import torch
import torch.nn as nn
import numpy as np

from core.box_ops import bbox_transform_inv
from core.box_ops import clip_to_image
from core.box_ops import filter_boxes
from core.box_ops import sort_by_scores
from core.nms.nms_wrapper import nms
from core.generate_anchorsCP import AnchorGenerator
from model.faster_rcnn.config import cfg
from core.generate_anchors import generate_anchors


class ProposalLayer(nn.Module):
    def __init__(self,
                 scales,
                 aspect_ratios):
        super().__init__()
        self.anchor_generator = AnchorGenerator(scales, aspect_ratios)
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(aspect_ratios))).float()
        self._num_anchors = self._anchors.size(0)
    
    def forward(self, inputs, im_shape, feat_height, feat_width):
        scores = inputs[0][:, self._num_anchors:, :, :]
        bbox_deltas = inputs[1]
        batch_size = scores.shape[0]
        
        # TODO
        pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH
        feat_stride = cfg.FEAT_STRIDE[0]
        min_size = cfg.TEST.RPN_MIN_SIZE
        # if self.training:
        #     pre_nms_topN = self.train_pre_nms_topN
        #     post_nms_topN = self.train_post_nms_topN
        #     min_size = cfg.TRAIN.RPN_MIN_SIZE
        # else:
        #     pass
        
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, feat_width) * feat_stride
        shift_y = np.arange(0, feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()
        
        A = self._num_anchors
        K = shifts.size(0)
        
        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        
        # anchors = self.anchor_generator.generate(feat_width, feat_height, feat_stride)
        
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        
        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        
        # 2. clip predicted boxes to image
        proposals = clip_to_image(proposals, im_shape, batch_size)
        
        # 3. remove predicted boxes with either height or width < threshold
        # TODO
        # keep = filter_boxes(proposals, min_size)
        # proposals = proposals[:, keep]
        # fg_scores = fg_scores[:, keep]
        
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
    
        for i in range(batch_size):
            proposals_single = proposals[i]
            scores_single = scores[i]
            
            # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            proposals_single, scores_single = sort_by_scores(
                proposals_single, scores_single, pre_nms_topN)
            
            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep = keep.long().view(-1)
            
            if post_nms_topN > 0:
                keep = keep[0:post_nms_topN]
            proposals_single = proposals_single[keep, :]
            scores_single = scores_single[keep, :]
            
            # padding 0 at the end.
            n_proposals = proposals_single.shape[0]
            output[i, :, 0] = i
            output[i, :n_proposals, 1:] = proposals_single
        
        return output


if __name__ == '__main__':
    pro = ProposalLayer()
    print(callable(pro))
    pro()
