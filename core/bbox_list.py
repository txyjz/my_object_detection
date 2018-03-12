"""A list that stores a collection of boxes."""

import numpy as np


class BoxList(object):
    def __init__(self, boxes):
        """Constructor of BoxList.
        Args:
            boxes: (numpy.array) with shape (n, 4), each element
                is corresponding to box's corner coordinates:
                `xmin, ymin, xmax, ymax`.
        """
        assert boxes.shape[-1] == 4

        self._data = boxes

    def get_ctr_coordinates_and_sizes(self):
        boxes = self._data
        ws, hs = self.sizes
        x_ctr = boxes[:, 0] + hs / 2
        y_ctr = boxes[:, 2] + ws / 2
        return [x_ctr, y_ctr, ws, hs]

    @property
    def sizes(self):
        boxes = self.get()
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        return ws, hs

    @property
    def areas(self):
        ws, hs = self.sizes
        return ws * hs

    def get(self):
        return self._data

    def set(self, boxes):
        self._data = boxes

    def set_mask(self, mask):
        self._data = self._data[mask]
