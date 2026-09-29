"""
RoadSafe - Interactive Web Application
======================================
A clean, interactive Streamlit app for real-time road damage detection
and severity assessment. Supports image uploads, pre-loaded samples,
video processing, and live camera input.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import time
import tempfile

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

from severity_estimator import SeverityEstimator
from inference import detect_frame, draw_detections

# Page configuration
st.set_page_config(
    page_title="RoadSafe - Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .badge-high { color: #DC2626; font-weight: bold; }
    .badge-med { color: #D97706; font-weight: bold; }
    .badge-low { color: #16A34A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(weights_path: str = "weights/best.pt"):
    """Load and cache the YOLO model."""
    path = Path(weights_path)
    if not path.exists():
        path = Path("runs/detect/road_damage/weights/best.pt")
    if not path.exists():
        st.error(f"Model weights not found at {weights_path}! Please ensure weights/best.pt exists.")
        st.stop()
    return YOLO(str(path))


def process_image(img_bgr, model, severity_est, conf, iou, device):
    """Run detection and return results + annotated image."""
    result = detect_frame(
        frame=img_bgr,
        model=model,
        severity_estimator=severity_est,
        conf_threshold=conf,
        iou_threshold=iou,
        device=device,
    )
    annotated_bgr = draw_detections(img_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return result, annotated_rgb


def main():
    st.markdown('<div class="main-title">🛣️ RoadSafe: Road Damage & Severity Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated AI detection of potholes and road cracks with real-time severity rating</div>', unsafe_allow_html=True)

    # ── Sidebar Controls ─────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", min_value=0.20, max_value=0.80, value=0.45, step=0.05)

    device_options = ["cpu", "0"] if torch.cuda.is_available() else ["cpu"]
    selected_device = st.sidebar.selectbox("Compute Device", device_options, index=0)

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Select Mode", ["📸 Image Detection", "🎥 Video Detection", "📷 Live Camera"])

    # Load Model & Severity Estimator
    model = load_model()
    severity_est = SeverityEstimator()

    # ── 1. Image Mode ─────────────────────────────────────────
    if mode == "📸 Image Detection":
        st.subheader("Image Analysis")
        input_choice = st.radio("Choose Input Method", ["Pick a Sample Image", "Upload Your Own Image"], horizontal=True)

        img_bgr = None
        sample_dir = Path("sample_images")

        if input_choice == "Pick a Sample Image":
            sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
            if sample_files:
                sample_names = [f.name for f in sample_files]
                selected_sample = st.selectbox("Select a sample road image:", sample_names, index=sample_names.index("India_009605.jpg") if "India_009605.jpg" in sample_names else 0)
                sample_path = sample_dir / selected_sample
                img_bgr = cv2.imread(str(sample_path))
            else:
                st.warning("No sample images found in sample_images/ directory.")
        else:
            uploaded_file = st.file_uploader("Upload a road photo (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            with st.spinner("Analyzing road surface..."):
                start_time = time.perf_counter()
                result, annotated_rgb = process_image(img_bgr, model, severity_est, conf_threshold, iou_threshold, selected_device)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Metric Cards
            num_dets = result["num_detections"]
            potholes = sum(1 for d in result["detections"] if d["class"] == "Pothole")
            cracks = num_dets - potholes
            high_sev = sum(1 for d in result["detections"] if d["severity"] == "High")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Defects", num_dets)
            m2.metric("Potholes", potholes)
            m3.metric("Cracks", cracks)
            m4.metric("High Severity", high_sev)
            m5.metric("Speed", f"{elapsed_ms:.1f} ms")

            st.markdown("---")

            # Image View (Side by side)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**Detected Road Damage & Severity**")
                st.image(annotated_rgb, use_container_width=True)

            # Detailed Detection Table
            if num_dets > 0:
                st.markdown("### 📋 Defect Breakdown")
                table_data = []
                for idx, det in enumerate(result["detections"], 1):
                    sev = det["severity"]
                    badge = "🔴 High" if sev == "High" else ("🟠 Medium" if sev == "Medium" else "🟢 Low")
                    table_data.append({
                        "#": idx,
                        "Defect Type": det["class"],
                        "Confidence": f"{det['confidence']:.1%}",
                        "Severity": badge,
                        "Bounding Box [x1, y1, x2, y2]": str(det["bbox"])
                    })
                st.dataframe(table_data, use_container_width=True)

                # Download Annotated Image
                out_img = Image.fromarray(annotated_rgb)
                temp_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                out_img.save(temp_buf.name)
                with open(temp_buf.name, "rb") as f:
                    st.download_button(
                        label="💾 Download Annotated Image",
                        data=f.read(),
                        file_name="roadsafe_detected.jpg",
                        mime="image/jpeg",
                    )
            else:
                st.success("✅ No road damage detected at the current confidence threshold!")

    # ── 2. Video Mode ─────────────────────────────────────────
    elif mode == "🎥 Video Detection":
        st.subheader("Dashcam Video Analysis")
        video_choice = st.radio("Choose Video Source", ["Sample Pothole Video", "Upload Video (.mp4)"], horizontal=True)

        video_path = None
        if video_choice == "Sample Pothole Video":
            sample_vid = Path("demo video/mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4")
            if sample_vid.exists():
                video_path = str(sample_vid)
                st.info(f"Using built-in sample: `{sample_vid.name}`")
            else:
                st.warning("Sample video not found in demo video/ directory.")
        else:
            uploaded_vid = st.file_uploader("Upload dashcam video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
            if uploaded_vid is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_vid.read())
                video_path = tfile.name

        if video_path is not None:
            max_frames = st.slider("Max Frames to Process (CPU speed limit)", min_value=30, max_value=300, value=75, step=15)
            
            if st.button("▶️ Start Video Detection"):
                cap = cv2.VideoCapture(video_path)
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                
                limit = min(max_frames, total_video_frames) if total_video_frames > 0 else max_frames

                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()

                frame_idx = 0
                total_detected = 0

                while cap.isOpened() and frame_idx < limit:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    res, ann = process_image(frame, model, severity_est, conf_threshold, iou_threshold, selected_device)
                    total_detected += res["num_detections"]

                    frame_placeholder.image(ann, caption=f"Frame {frame_idx}/{limit} | Current Detections: {res['num_detections']}", use_container_width=True)
                    progress_bar.progress(frame_idx / limit)
                    status_text.text(f"Processing frame {frame_idx} of {limit}...")

                cap.release()
                status_text.success(f"🎉 Done! Analyzed {frame_idx} frames. Total damage instances logged: {total_detected}")

    # ── 3. Live Camera Mode ───────────────────────────────────
    elif mode == "📷 Live Camera":
        st.subheader("Live Camera Snapshot")
        st.write("Take a picture of the road or pavement using your webcam:")
        camera_img = st.camera_input("Capture Road Photo")

        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Analyzing camera snapshot..."):
                result, annotated_rgb = process_image(img_bgr, model, severity_est, conf_threshold, iou_threshold, selected_device)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your Camera Shot**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**RoadSafe Detection**")
                st.image(annotated_rgb, use_container_width=True)

            st.write(f"**Found {result['num_detections']} damage defect(s)**")


if __name__ == "__main__":
    main()
