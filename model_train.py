import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

# Parameters
data_dir = 'dataset'  # Path to your dataset folder
batch_size = 16
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 with weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Replace final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

# Use recommended transforms for the pretrained weights
transform = weights.transforms()

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "smoking_model_weights.pt")
print("✅ Model saved as smoking_model.pt")
 