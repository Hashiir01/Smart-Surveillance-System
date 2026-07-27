import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import tempfile
import numpy as np
from torchvision import transforms

# ================== Model Definitions ==================

class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        self.base_model = resnet18(pretrained=False)  # Changed to match checkpoint
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Violence/Non-violence
        
    def forward(self, x):
        return self.base_model(x)

# ================== Model Loading ==================

@st.cache_resource
def load_smoking_model():
    """Load smoking detection model with pretrained weights"""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    
    # Load custom trained weights
    model.load_state_dict(
        torch.load(
            "C:\\Users\\User\\Desktop\\sss\\Scripts\\Saved Model\\smoking_model_weights.pt", 
            map_location=torch.device("cpu")
        )
    )
    model.eval()
    smoking_transform = weights.transforms()
    return model, smoking_transform

@st.cache_resource
def load_violence_model():
    """Load violence detection model with checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceClassifier().to(device)
    checkpoint_path = "C:\\Users\\User\\Desktop\\sss\\Scripts\\Saved Model\\model_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Fix key mismatches in state dict
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)  # strict=False to ignore non-matching keys
    model.eval()
    
    violence_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    return model, violence_transform, device

# Load both models
smoking_model, smoking_transform = load_smoking_model()
violence_model, violence_transform, device = load_violence_model()

# ================== Prediction Functions ==================

def predict_smoking(image):
    """Predict smoking from PIL Image"""
    input_tensor = smoking_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = smoking_model(input_tensor)
        prediction = torch.sigmoid(output).item()
    return "Smoking" if prediction > 0.5 else "No Smoking"

def predict_violence(frame):
    """Predict violence from OpenCV frame"""
    try:
        img = violence_transform(frame).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return "Error"
    
    with torch.no_grad():
        output = violence_model(img)
        _, pred = torch.max(output, 1)
        return "Violence" if pred.item() == 1 else "No Violence"

# ================== Video Processing ==================

def process_video(video_path, detection_type):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if detection_type == "Smoking Detection":
            pil_image = Image.fromarray(display_frame)
            label = predict_smoking(pil_image)
        else:
            label = predict_violence(frame)
        
        # Color coding for results
        color = (255, 0, 0) if ("Smoker" in label or "Violence" in label) else (0, 255, 0)
        
        # Add label to frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# ================== Webcam Processing ==================

def start_webcam(detection_type):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Camera")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if detection_type == "Smoking Detection":
            pil_image = Image.fromarray(display_frame)
            label = predict_smoking(pil_image)
        else:
            label = predict_violence(frame)
        
        color = (255, 0, 0) if ("Smoker" in label or "Violence" in label) else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

# ================== Streamlit UI ==================

# ---- Custom CSS for beautification ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    }
    .main {
        background: rgba(255,255,255,0.85);
        border-radius: 18px;
        padding: 2rem 2rem 1rem 2rem;
        box-shadow: 0 4px 32px 0 rgba(31,38,135,0.15);
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5em 2em;
    }
    .stRadio>div>label, .stSelectbox>div>div>div>div {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stSidebar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .result-box {
        border-radius: 12px;
        padding: 1em;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 1em;
        text-align: center;
    }
    .result-smoker { background: #ffe0e0; color: #d7263d; }
    .result-nonsmoker { background: #e0ffe0; color: #1b5e20; }
    .result-violence { background: #fff3cd; color: #856404; }
    .result-nonviolence { background: #e0f7fa; color: #006064; }
    .dev-list {
        margin-top: 1em;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.7em;
    }
    .dev-card {
        display: flex;
        align-items: center;
        gap: 0.7em;
        background: rgba(255,255,255,0.13);
        border-radius: 18px;
        padding: 0.5em 1.2em;
        box-shadow: 0 2px 8px 0 rgba(31,38,135,0.10);
        font-size: 1.08rem;
        font-weight: 600;
        color: #fff;
        transition: transform 0.18s, box-shadow 0.18s, background 0.18s;
        cursor: pointer;
    }
    .dev-card:hover {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        color: #222;
        transform: scale(1.04) translateY(-2px);
        box-shadow: 0 4px 16px 0 rgba(31,38,135,0.18);
    }
    .dev-emoji {
        font-size: 1.7em;
        margin-right: 0.3em;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.10));
    }
    .dev-title {
        font-size: 1.18rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.2em;
        color: #ffd200;
        text-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<div style='text-align:center;'>
    <h1 style='font-size:2.8rem; margin-bottom:0.2em;'>🚨 Smart Surveillance System</h1>
    <p style='font-size:1.2rem; color:#555;'>Detect <b>smoking</b> or <b>violent behavior</b> in images, videos, or live camera feed.<br>Powered by Deep Learning & Computer Vision</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for detection type selection
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1em;'>
        <h2 style='color:white;'>🛡️ Detection Type</h2>
    </div>
    """, unsafe_allow_html=True)
    detection_type = st.selectbox(
        "Select Detection Type:",
        ("Smoking Detection", "Violence Detection")
    )
    st.markdown("""
    <hr style='border:1px solid #fff; margin:1em 0;'>
    <div style='text-align:center;'>
        <div class='dev-title'>Developed By</div>
        <div class='dev-list'>
            <div class='dev-card'><span class='dev-emoji'>👩‍💻</span>Adan Akbar</div>
            <div class='dev-card'><span class='dev-emoji'>👨‍💻</span>Bilal Qaisar</div>
            <div class='dev-card'><span class='dev-emoji'>👨‍💻</span>Hashir Nasir</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main input options
option = st.radio(
    "Choose Input Type:",
    ("Upload Image", "Upload Video", "Live Camera"),
    horizontal=True
)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing image..."):
            if detection_type == "Smoking Detection":
                label = predict_smoking(image)
                if "Smoker" in label:
                    st.markdown(f"<div class='result-box result-smoker'>{label}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box result-nonsmoker'>{label}</div>", unsafe_allow_html=True)
            else:
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                label = predict_violence(frame)
                if "Violence" in label:
                    st.markdown(f"<div class='result-box result-violence'>{label}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box result-nonviolence'>{label}</div>", unsafe_allow_html=True)

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi","mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.info("Processing video. Please wait...")
        with st.spinner("Analyzing video frames..."):
            process_video(tfile.name, detection_type)
        st.success("Video analysis complete!")

elif option == "Live Camera":
    st.warning("Press the 'Stop Camera' button to end the webcam stream.")
    with st.spinner("Starting webcam and analyzing frames..."):
        start_webcam(detection_type)
    st.success("Webcam session ended.")
