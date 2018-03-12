import torch.nn as nn
import torchvision.models as models

from model.faster_rcnn.faster_rcnn import _FasterRCNN


class vgg16(_FasterRCNN):
    """vgg16"""
    
    def __init__(self, n_classes=21):
        self.n_classes = n_classes
        self.base_out_channels = 512
        super().__init__(n_classes)
    
    def _init_module(self):
        vgg = models.vgg16()
        
        # define extractor:
        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        # fix the layers before conv3
        for layer in range(10):
            for parm in self.RCNN_base[layer].parameters():
                parm.requires_grad = False
        
        # define top:
        self.RCNN_top = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        
        # define the final cls layer:
        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        
        # define the final reg layer
        self.RCNN_bbox_pred = nn.Linear(4096, self.n_classes * 4)


if __name__ == '__main__':
    vgg16()
