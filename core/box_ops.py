"""Some operations to bbox."""

import torch

from core.bbox_list import BoxList


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    
    return boxes


def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        
        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1, -1).expand_as(gt_heights))
    
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights
        
        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')
    
    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), 2)
    
    return targets


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    
    if anchors.dim() == 2:
        
        N = anchors.size(0)
        K = gt_boxes.size(1)
        
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        
        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        
        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)
        
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)
        
        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0
        
        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua
        
        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    
    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)
        
        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()
        
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        
        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        
        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)
        
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)
        
        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0
        
        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        
        overlaps = iw * ih / ua
        
        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')
    
    return overlaps


def bbox_transform_inv(boxes, bbox_deltas, batch_size):
    """Add bbox_deltas to boxes, as described in
      `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`
       Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun."""
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights
    
    dx = bbox_deltas[:, :, 0::4]
    dy = bbox_deltas[:, :, 1::4]
    dw = bbox_deltas[:, :, 2::4]
    dh = bbox_deltas[:, :, 3::4]
    
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    
    pred_boxes = bbox_deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
    
    return pred_boxes


def clip_to_image(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    im_shape = im_shape.data
    for i in range(batch_size):
        # x1>=0
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        # y1>=0
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        # x2<h
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        # y2<w
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
    hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
    keep = (ws >= min_size) & (hs >= min_size)
    
    return keep.long().squeeze()


def sort_by_scores(boxes, scores, n_pre_nms):
    _, order = torch.sort(scores, 0, True)
    if n_pre_nms > 0:
        order = order[:n_pre_nms]
    
    boxes = boxes[order, :]
    scores = scores[order].view(-1, 1)
    
    return boxes, scores


def intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes

    Returns:
      a numpy.ndarray with shape [N, M] representing pairwise intersections
    """
    xmin1, ymin1, xmax1, ymax1 = np.split(boxlist1.get(), 4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = np.split(boxlist2.get(), 4, axis=1)
    all_pairs_max_xmin = np.maximum(xmin1, np.transpose(xmin2))
    all_pairs_min_xmax = np.maximum(xmax1, np.transpose(xmax2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    all_pairs_max_ymin = np.maximum(ymin1, np.transpose(ymin2))
    all_pairs_min_ymax = np.maximum(ymax1, np.transpose(ymax2))
    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    return intersect_widths * intersect_heights


def bbox_iou(boxes1, boxes2):
    """Compute pair-wise intersection-over-union between box collections.

    Args:
        boxes1, holding boxes with shape(

    Returns:
        a numpy.ndarray with shape [N, M] representing pairwise iou scores.
    """
    intersections = intersection(boxlist1, boxlist2)
    areas1 = boxlist1.areas
    areas2 = boxlist2.areas
    unions = (
        np.expand_dims(areas1, 1) + np.expand_dims(areas2, 0) - intersections)
    return np.where(
        np.equal(intersections, 0.0),
        np.zeros_like(intersections), np.true_divide(intersections, unions))


def vstack(boxes1, boxes2):
    boxes = torch.stack((boxes1, boxes2))
    return boxes
