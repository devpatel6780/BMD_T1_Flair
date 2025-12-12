import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for BMD regression.
    Input : [B, 3, 256, 256]  (3-slice 2.5D)
    Output: [B, 1]
    """

    def __init__(self, in_ch=3, dropout=0.30):
        super().__init__()

        # Feature extractor (downsample to 16x16)
        self.features = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Head: use global pooling (more stable than flattening 16*16*256)
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B,256,1,1]

        self.regressor = nn.Sequential(
            nn.Flatten(),                     # -> [B,256]
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN(in_ch=3)
    dummy = torch.randn(4, 3, 256, 256)
    out = model(dummy)
    print("Input:", dummy.shape)
    print("Output:", out.shape)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
