"""
Generates anchors on the fly as described in:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
"""
import numpy as np
import torch


class AnchorGenerator(object):
    def __init__(self,
                 scales=(8, 16, 32),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 base_anchor_size=16,
                 feat_stride=1):
        self._scales = scales
        self._aspect_ratios = aspect_ratios
        self._base_anchor_size = base_anchor_size
    
    def num_anchors_per_location(self):
        return [len(self._scales) * len(self._aspect_ratios)]
    
    def generate(self, feat_height, feat_width, feat_stride):
        x_centers = np.arange(feat_width) * feat_stride
        y_centers = np.arange(feat_height) * feat_stride
        
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)
        x_centers = x_centers.reshape(-1)
        y_centers = y_centers.reshape(-1)
        
        bbox_centers = np.stack((x_centers, y_centers), axis=1)
        bbox_centers = bbox_centers.reshape([-1, 2])
        
        bbox_sizes = _construct_base_anchor_sizes(
            self._scales,
            self._aspect_ratios,
            self._base_anchor_size)
        
        def convert_bbox_center_to_corner(centers, sizes):
            centers = np.expand_dims(centers, axis=1)
            sizes = np.expand_dims(sizes, axis=0)
            half_sizes = .5 * sizes
            return np.concatenate(
                ((centers - half_sizes).reshape([-1, 2]),
                 (centers + half_sizes).reshape([-1, 2])),
                axis=1)
        
        # [hwk, 4] [xmin, ymin, xmax, ymax]
        anchors = np.round(
            convert_bbox_center_to_corner(bbox_centers, bbox_sizes))
        
        anchors = anchors[np.newaxis, :, :]
        anchors = torch.from_numpy(anchors).float().cuda()
        return anchors


def _construct_base_anchor_sizes(
    scales=(0.5, 1.0, 2.0),
    aspect_ratios=(0.5, 1.0, 2.0),
    base_anchor_size=16):
    # scales, aspect_ratios = np.meshgrid(scales, aspect_ratios)
    base_area = base_anchor_size * base_anchor_size
    
    # s = scales.ravel()
    # r = aspect_ratios.ravel()
    # print(s, r)
    
    base_ws = np.sqrt(base_area / np.array(aspect_ratios))
    base_hs = base_ws * aspect_ratios
    
    scales_grid, w_gird = np.meshgrid(scales, base_ws)
    bbox_widths = scales_grid * w_gird
    
    scales_grid, h_grid = np.meshgrid(scales, base_hs)
    bbox_heights = scales_grid * h_grid
    
    # return [[w1,h1],[w2,h2]...[wn,hn]]
    return np.stack((bbox_widths, bbox_heights), axis=2).reshape([-1, 2])

if __name__ == '__main__':
    gen = AnchorGenerator()
    anchors = gen.generate(1, 1, 16)
    pass