import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Resolve model path relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Saved Model" / "best.pt"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model weights at:\n{MODEL_PATH}\n"
            "Place best.pt in the 'Saved Model' folder."
        )
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
    MODEL_LOAD_ERROR = None
except FileNotFoundError as e:
    model = None
    MODEL_LOAD_ERROR = str(e)


def predict_image(image_bgr, conf=0.25):
    """Run YOLO on a BGR image; return annotated RGB image and label summary."""
    results = model.predict(image_bgr, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()  # BGR with boxes
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    names = result.names or {}
    detections = []
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            label = names.get(cls_id, str(cls_id))
            detections.append(f"{label} ({score:.0%})")

    if detections:
        summary = "Detected: " + ", ".join(detections)
    else:
        summary = "No Smoking Detected"

    return annotated_rgb, summary, bool(detections)


def process_video(video_path, conf=0.25):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    status = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_rgb, summary, found = predict_image(frame, conf=conf)
        status.markdown(
            f"<div class='result-box {'result-smoker' if found else 'result-nonsmoker'}'>{summary}</div>",
            unsafe_allow_html=True,
        )
        stframe.image(annotated_rgb, channels="RGB")

    cap.release()


def start_webcam(conf=0.25):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status = st.empty()
    stop_button = st.button("Stop Camera")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        annotated_rgb, summary, found = predict_image(frame, conf=conf)
        status.markdown(
            f"<div class='result-box {'result-smoker' if found else 'result-nonsmoker'}'>{summary}</div>",
            unsafe_allow_html=True,
        )
        stframe.image(annotated_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()


# ================== Streamlit UI ==================

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
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0,0,0,0.08);
        text-align: center;
        color: #555;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align:center;'>
    <h1 style='font-size:2.8rem; margin-bottom:0.2em;'>🚨 Smart Surveillance System</h1>
    <p style='font-size:1.2rem; color:#555;'>Detect <b>smoking / cigarettes</b> in images, videos, or live camera feed.<br>Powered by YOLO Deep Learning</p>
</div>
""",
    unsafe_allow_html=True,
)

if MODEL_LOAD_ERROR:
    st.error(MODEL_LOAD_ERROR)
    st.stop()

with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; margin-bottom:1em;'>
        <h2 style='color:white;'>🛡️ Detection</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    st.caption(f"Model: `{MODEL_PATH.name}`")

st.markdown("---")

option = st.radio(
    "Choose Input Type:",
    ("Upload Image", "Upload Video", "Live Camera"),
    horizontal=True,
)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Analyzing image..."):
            annotated_rgb, summary, found = predict_image(frame, conf=conf)
        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
        css = "result-smoker" if found else "result-nonsmoker"
        st.markdown(f"<div class='result-box {css}'>{summary}</div>", unsafe_allow_html=True)

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        st.info("Processing video. Please wait...")
        with st.spinner("Analyzing video frames..."):
            process_video(tfile.name, conf=conf)
        st.success("Video analysis complete!")

elif option == "Live Camera":
    st.warning("Press the 'Stop Camera' button to end the webcam stream.")
    with st.spinner("Starting webcam and analyzing frames..."):
        start_webcam(conf=conf)
    st.success("Webcam session ended.")

st.markdown(
    "<div class='app-footer'>Developed by Muhammad Hahsir</div>",
    unsafe_allow_html=True,
)
