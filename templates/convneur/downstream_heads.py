"""
Downstream Heads for ConvNeur Backbone
Implements:
1. ConvNeurForDetection: Bounding Box prediction on top of multi-scale features.
2. ConvNeurForSegmentation: Dense pixel mask prediction (UPerNet / FPN Lite).

NOTE ON BENCHMARKING & SURROGATES:
In this template, Penn-Fudan is employed as a lightweight, small-scale surrogate sanity check
(~170 images) to verify multi-scale feature propagation and convergence of downstream heads.
Penn-Fudan DOES NOT directly represent or benchmark large-scale tasks such as ADE20K (semantic segmentation)
or MS COCO (object detection). Metrics produced on Penn-Fudan must not be reported as ADE20K/COCO performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeurForSegmentation(nn.Module):
    """
    Semantic segmentation head on top of ConvNeur backbone (FPN-lite architecture).
    Used with small-scale surrogate dataset (Penn-Fudan) for sanity check of segmentation capabilities.
    """
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        # dims for m1 are [40, 80, 160, 320]
        c1, c2, c3, c4 = backbone.stages[0][0].local_dw.in_channels, \
                         backbone.stages[1][0].local_dw.in_channels, \
                         backbone.stages[2][0].local_dw.in_channels, \
                         backbone.stages[3][0].local_dw.in_channels

        # Top-down FPN-lite lateral connections
        self.lat4 = nn.Conv2d(c4, 64, 1)
        self.lat3 = nn.Conv2d(c3, 64, 1)
        self.lat2 = nn.Conv2d(c2, 64, 1)

        # Smooth and segmentation head
        self.smooth = nn.Conv2d(64, 64, 3, padding=1)
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        # Get multi-scale features from ConvNeur
        features = self.backbone.forward_features(x)
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)

        out = self.smooth(p2)
        logits = self.seg_head(out)
        # Upsample directly to original input resolution
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


class ConvNeurForDetection(nn.Module):
    """
    Object detection head on top of ConvNeur backbone.
    Used with small-scale surrogate datasets (Penn-Fudan / VOC subset) for sanity check of detection pipeline.
    Predicts normalized bounding boxes [cx, cy, w, h] and class/objectness score.
    """
    def __init__(self, backbone, num_classes=20, num_queries=1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        c4 = backbone.stages[3][0].local_dw.in_channels  # 320 for m1

        # Global spatial pooling + query MLP
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Linear(c4, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        # Box regressor: normalized (cx, cy, w, h) in [0, 1]
        self.box_head = nn.Sequential(
            nn.Linear(128, 4 * num_queries),
            nn.Sigmoid()
        )
        # Objectness / class score
        self.score_head = nn.Linear(128, num_classes * num_queries)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        f4 = features[-1]
        feat = self.pool(f4).flatten(1)
        hidden = self.shared(feat)

        boxes = self.box_head(hidden).view(-1, self.num_queries, 4)
        if self.num_classes > 1:
            scores = self.score_head(hidden).view(-1, self.num_queries, self.num_classes)
        else:
            scores = self.score_head(hidden).view(-1, self.num_queries)
        return boxes, scores
