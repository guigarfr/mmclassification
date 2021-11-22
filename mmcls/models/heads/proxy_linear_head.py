# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
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

        self.refactor = nn.Linear(
            self.in_channels, self.out_features, bias=False)

        self.standardize = nn.LayerNorm(
            self.in_channels, elementwise_affine=False)

        self.fc = nn.Parameter(torch.Tensor(
            self.num_classes, self.out_features))

        stdv = 1. / math.sqrt(self.fc.size(1))
        self.fc.data.uniform_(-stdv, stdv)

        self.scale = temperature_scale

    def get_feats(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        x.view(x.size(0), -1)
        x = self.standardize(x)
        x = self.refactor(x)
        return F.normalize(x)

    def simple_test(self, x):
        """Test without augmentation."""
        x = self.get_feats(x)
        x = F.linear(x, F.normalize(self.fc))
        pred = F.softmax(x, dim=1) if x is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        # x = self.get_feats(x)
        x = F.linear(x, F.normalize(self.fc)) / self.scale

        losses = self.loss(x, gt_label, **kwargs)
        return losses
