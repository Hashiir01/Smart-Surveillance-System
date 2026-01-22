import torch
import torch.nn as nn
import torch.optim as optim
from model import ViolenceClassifier
from data_preprocessing import get_dataloaders

def train_initial_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ViolenceClassifier().to(device)
    train_loader, test_loader = get_dataloaders("C:\\Users\\User\\Desktop\\sss\\Dataset\\frames")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        print(f"\nEpoch {epoch + 1}/{epochs}")

        progress_interval = max(total_batches // 100, 1)
        next_progress = 1

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) >= (next_progress * progress_interval) and next_progress <= 100:
                print(f"{next_progress}% complete")
                next_progress += 1

        print(f"Loss: {running_loss / len(train_loader):.4f}")

    # Save model and optimizer state for future updates
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "C:\\Users\\User\\Desktop\\sss\\Scripts\\Saved Model\\model_checkpoint.pth")

    print("Initial model saved.")

if __name__ == "__main__":
    train_initial_model()
