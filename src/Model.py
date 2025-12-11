import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN model for BMD regression prediction
    Input: [batch, 1, 256, 256]
    Output: [batch, 1] BMD prediction value
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Regression head (16 x 16 x 256 = 65536)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


if __name__ == '__main__':
    # Test model
    model = SimpleCNN()

    # Print model structure
    print(model)

    # Test forward pass
    dummy_input = torch.randn(4, 1, 256, 256)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
