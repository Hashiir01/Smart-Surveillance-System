import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from model import ViolenceClassifier
from data_preprocessing import get_dataloaders

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ViolenceClassifier().to(device)

    # Load checkpoint (saved using torch.save({'model_state_dict': ..., ...}))
    checkpoint_path = "C:\\Users\\User\\Desktop\\sss\\Scripts\\Saved Model\\model_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state dict only
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    _, test_loader = get_dataloaders("C:\\Users\\User\\Desktop\\sss\\Dataset\\frames")

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    evaluate_model()
