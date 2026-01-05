import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.DataLoader import create_dataloaders
from src.Model import SimpleCNN


def train():
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Hyperparameters
    epochs = 50
    batch_size = 16
    learning_rate = 1e-4

    # Data paths
    data_dir = 'data/Sagittal_T1_FLAIR'
    xlsx_path = 'data/metadata.xlsx'

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        xlsx_path=xlsx_path,
        batch_size=batch_size
    )

    # Create model (3-channel input because you stacked 3 slices)
    model = SimpleCNN(in_ch=3).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    print("\nStart training...")
    print("=" * 60)

    for epoch in range(epochs):
        # ---- Training phase ----
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)   # [B,3,H,W]
            labels = labels.to(device)   # [B]

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)  # [B]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Update LR
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            save_marker = " *"
        else:
            save_marker = ""

        print(f"Epoch [{epoch+1:3d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}{save_marker}")

    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")

    # Test phase
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    evaluate(model, test_loader, device)


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)

            predictions.extend(outputs.cpu().numpy().tolist())
            actuals.extend(labels.numpy().tolist())

    predictions = np.array(predictions, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)

    # Metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print("\n" + "=" * 40)
    print("          Test Results")
    print("=" * 40)
    print(f"MSE  (Mean Squared Error):      {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"R2   (Coefficient of Determination): {r2:.4f}")
    print("=" * 40)

    # Show prediction examples
    print("\nPrediction examples (first 10):")
    print("-" * 30)
    print(f"{'Actual':^12} | {'Predicted':^12}")
    print("-" * 30)
    for i in range(min(10, len(actuals))):
        print(f"{actuals[i]:^12.3f} | {predictions[i]:^12.3f}")
    print("-" * 30)


if __name__ == '__main__':
    train()
