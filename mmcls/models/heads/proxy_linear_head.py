# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.linalg as la
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import HEADS
from mmcls.models.heads.cls_head import ClsHead


@HEADS.register_module()
class ProxyLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_features,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 temperature_scale=0.05,
                 *args,
                 **kwargs):
        super(ProxyLinearClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.out_features = out_features
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.refactor = nn.Linear(self.in_channels, self.out_features)

        self.fc = nn.Parameter(torch.Tensor(self.in_channels, self.num_classes))
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        self.scale = temperature_scale

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]

        x = F.normalize(self.refactor(x), dim=-1)
        cls_score = x.matmul(F.normalize(self.fc, dim=-1).t())

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        if isinstance(x, tuple):
            x = x[-1]
        x = F.normalize(self.refactor(x), dim=-1)
        cls_score = x.matmul(F.normalize(self.fc, dim=-1).t()) * self.scale

        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
