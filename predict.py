import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from model import ViolenceClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = ViolenceClassifier().to(device)

# Load checkpoint (make sure it includes 'model_state_dict')
checkpoint_path = "C:\\Users\\User\\Desktop\\sss\\Scripts\\Saved Model\\model_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_frame(frame):
    try:
        img = transform(frame).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Frame transformation error: {e}")
        return "Error"
    
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        return "Violence" if pred.item() == 1 else "Non-Violence"

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = predict_frame(frame)
        if label != "Error":
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if label == "Non-Violence" else (0, 0, 255), 2)

        cv2.imshow("Prediction", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_video("C:\\Users\\User\\Desktop\\sss\\test_video1.mp4")
