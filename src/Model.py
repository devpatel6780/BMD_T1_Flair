# ########################################
# # best r2 0.32
# ########################################

# import torch
# import torch.nn as nn


# class SimpleCNN(nn.Module):
#     """
#     Stronger CNN for BMD regression.
#     Input : [B, 3, 256, 256]
#     Output: [B, 1]
#     """

#     def __init__(self, in_ch=3, dropout=0.35):
#         super().__init__()

#         def block(cin, cout):
#             return nn.Sequential(
#                 nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(cout),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(cout),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#             )

#         # 256 -> 64 -> 228 -> 32 -> 32
#         self.features = nn.Sequential(
#             block(in_ch, 32),
#             block(32, 64),
#             block(64, 128),
#             block(128, 256),
#         )

#         # Global pooling
#         self.pool = nn.AdaptiveAvgPool2d(1)  # [B,256,1,1]

#         self.regressor = nn.Sequential(
#             nn.Flatten(),          # [B,256]
#             nn.Linear(256, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout * 0.7),
#             nn.Linear(64, 1),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)
#         x = self.regressor(x)
#         return x


# if __name__ == "__main__":
#     model = SimpleCNN(in_ch=5)
#     dummy = torch.randn(4, 3, 256, 256)
#     out = model(dummy)
#     print("Input:", dummy.shape)
#     print("Output:", out.shape)
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total params: {total:,} | Trainable: {trainable:,}")


# src/Model.py

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN for BMD regression.
    Input : [B, in_ch, 256, 256]
    Output: [B, 1]
    """

    def __init__(self, in_ch=3, dropout=0.35):  # Increased from 0.15 to reduce overfitting
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            block(in_ch, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),          # [B,256]
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    in_ch = 3
    model = SimpleCNN(in_ch=in_ch)
    dummy = torch.randn(4, in_ch, 256, 256)
    out = model(dummy)
    print("Input:", dummy.shape)
    print("Output:", out.shape)
