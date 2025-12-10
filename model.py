# model.py
import torch
import torch.nn as nn

class PointNetLike(nn.Module):
    def __init__(self, num_classes=2, num_points=64):
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        
        # PointNet
        self.point_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU()
        )
        
        # Scalar
        self.scalar_mlp = nn.Sequential(
            nn.Linear(3, 32), # 3 scalars here: area, perimeter and diameter
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, point_set, scalars):
        """
        Args:
            point_set: (B, N, 2)
            scalars:   (B, 3)
        """

        # PointNet: per-point MLP + max pooling
        B, N, _ = point_set.shape
        x = self.point_mlp(point_set)          # (B, N, 1024)
        x = torch.max(x, dim=1)[0]             # (B, 1024)
        
        # Scalar MLP
        s = self.scalar_mlp(scalars)           # (B, 64)
       
        # 融合
        # 这里的大概意思是：模型一共有PointNet和Scalar两个分支分别负责判定轮廓的几何形状和几何尺寸
        # 两个分支在这里融合到一起，从而使模型可以综合形状和尺寸来判断轮廓的分类
        # 不过说实话这个地方我也不是很懂，反正ai给我什么我就写什么

        combined = torch.cat([x, s], dim=1)    # (B, 1024+64 = 1088)
        logits = self.classifier(combined)     # (B, num_classes)
        return logits