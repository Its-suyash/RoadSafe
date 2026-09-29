"""
RoadSafe - Minimalist Web Application
======================================
Vercel/Apple-inspired clean, minimalist interface for automated road surface
defect detection and geometric severity assessment.
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
    page_title="RoadSafe - AI Road Surface Audit",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist Vercel/Apple Design CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #09090B;
    }

    /* Minimalist App Header */
    .hero-badge {
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
        color: #52525B;
        background-color: #F4F4F5;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #E4E4E7;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #09090B;
        margin: 0;
        line-height: 1.15;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #71717A;
        margin-top: 0.35rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Metric Card */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #FAFAFA;
        border: 1px solid #E4E4E7;
        border-radius: 10px;
        padding: 12px 14px;
        transition: border-color 0.2s ease;
    }
    .metric-box:hover {
        border-color: #D4D4D8;
    }
    .metric-lbl {
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #71717A;
        margin-bottom: 4px;
    }
    .metric-num {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #09090B;
        line-height: 1.1;
    }

    /* Badges */
    .badge-high {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-med {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-low {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Streamlit overrides for minimalist feel */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E4E4E7;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.9rem;
        color: #71717A;
        padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F4F4F5 !important;
        color: #09090B !important;
        font-weight: 600 !important;
    }
    button[kind="primary"], .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }
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
    # Hero Title Bar
    st.markdown("""
    <div>
        <span class="hero-badge">Autonomous Road Vision</span>
        <h1 class="hero-title">RoadSafe</h1>
        <p class="hero-desc">Real-time road surface defect localization & geometric severity assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Fixed calibrated thresholds (cleaner, distraction-free interface)
    conf_threshold = 0.25
    iou_threshold = 0.45

    # ── Sidebar Controls ─────────────────────────────────────
    st.sidebar.markdown("**System Profile**")
    device_options = ["cpu", "0"] if torch.cuda.is_available() else ["cpu"]
    selected_device = st.sidebar.selectbox("Compute Hardware", device_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #71717A; line-height: 1.6;">
        <strong>Model:</strong> YOLOv8-nano<br>
        <strong>Parameters:</strong> 3.0M<br>
        <strong>Footprint:</strong> 5.97 MB<br>
        <strong>Benchmark:</strong> RDD2022 India
    </div>
    """, unsafe_allow_html=True)

    # Load Model & Severity Estimator
    model = load_model()
    severity_est = SeverityEstimator()

    # Minimalist Navigation Tabs
    tab_img, tab_vid, tab_cam = st.tabs(["📸 Image Audit", "🎥 Video Stream", "📷 Camera Capture"])

    # ── Tab 1: Image Audit ────────────────────────────────────
    with tab_img:
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            input_source = st.selectbox("Select Input", ["Curated Benchmark Samples", "Upload Custom Image"])

        img_bgr = None
        sample_dir = Path("sample_images")

        if input_source == "Curated Benchmark Samples":
            sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
            if sample_files:
                sample_names = [f.name for f in sample_files]
                with col_ctrl2:
                    selected_sample = st.selectbox("Select Road Scene", sample_names, index=sample_names.index("India_009605.jpg") if "India_009605.jpg" in sample_names else 0)
                sample_path = sample_dir / selected_sample
                img_bgr = cv2.imread(str(sample_path))
            else:
                st.warning("No sample images found in sample_images/ directory.")
        else:
            with col_ctrl2:
                uploaded_file = st.file_uploader("Upload road image (.jpg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            start_time = time.perf_counter()
            result, annotated_rgb = process_image(img_bgr, model, severity_est, conf_threshold, iou_threshold, selected_device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Minimalist Metrics Strip
            num_dets = result["num_detections"]
            potholes = sum(1 for d in result["detections"] if d["class"] == "Pothole")
            cracks = num_dets - potholes
            high_sev = sum(1 for d in result["detections"] if d["severity"] == "High")

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-lbl">Total Defects</div>
                    <div class="metric-num">{num_dets}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">Potholes</div>
                    <div class="metric-num">{potholes}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">Cracks</div>
                    <div class="metric-num">{cracks}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">High Risk</div>
                    <div class="metric-num" style="color: {'#DC2626' if high_sev > 0 else '#09090B'};">{high_sev}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">Latency</div>
                    <div class="metric-num">{elapsed_ms:.0f}<span style="font-size: 0.9rem; font-weight: 500; color: #71717A;">ms</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Visual Comparison
            col_orig, col_pred = st.columns(2)
            with col_orig:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #71717A; margin-bottom: 6px;'>ORIGINAL FOOTAGE</div>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_pred:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #71717A; margin-bottom: 6px;'>DETECTION & SEVERITY OVERLAY</div>", unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)

            # Defect Audit Table
            if num_dets > 0:
                st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Defect Inspection Log</div>", unsafe_allow_html=True)
                table_data = []
                for idx, det in enumerate(result["detections"], 1):
                    sev = det["severity"]
                    badge = "🔴 High" if sev == "High" else ("🟠 Medium" if sev == "Medium" else "🟢 Low")
                    table_data.append({
                        "ID": f"#{idx}",
                        "Category": det["class"],
                        "Confidence": f"{det['confidence']:.1%}",
                        "Severity": badge,
                        "Bounding Box [x1, y1, x2, y2]": str(det["bbox"])
                    })
                st.dataframe(table_data, use_container_width=True, hide_index=True)

                # Export Button
                out_img = Image.fromarray(annotated_rgb)
                temp_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                out_img.save(temp_buf.name)
                with open(temp_buf.name, "rb") as f:
                    st.download_button(
                        label="Download Annotated Evidence (.jpg)",
                        data=f.read(),
                        file_name="roadsafe_audit.jpg",
                        mime="image/jpeg",
                    )
            else:
                st.success("Clean Roadway: No structural defects detected at current confidence.")

    # ── Tab 2: Video Stream ───────────────────────────────────
    with tab_vid:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>Dashcam Video Processing</div>", unsafe_allow_html=True)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            video_mode = st.selectbox("Video Source", ["Built-in Pothole Dashcam Sample", "Upload Video File (.mp4)"])

        video_path = None
        if video_mode == "Built-in Pothole Dashcam Sample":
            sample_vid = Path("demo video/mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4")
            if sample_vid.exists():
                video_path = str(sample_vid)
                with col_v2:
                    st.caption(f"Loaded: `{sample_vid.name}`")
        else:
            with col_v2:
                uploaded_vid = st.file_uploader("Upload video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
            if uploaded_vid is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_vid.read())
                video_path = tfile.name

        if video_path is not None:
            max_frames = st.slider("Frame Limit (Processing Budget)", min_value=30, max_value=300, value=75, step=15)
            if st.button("Start Dashcam Analysis", type="primary"):
                cap = cv2.VideoCapture(video_path)
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

                    frame_placeholder.image(ann, caption=f"Frame {frame_idx}/{limit} | Active Detections: {res['num_detections']}", use_container_width=True)
                    progress_bar.progress(frame_idx / limit)
                    status_text.text(f"Auditing frame {frame_idx}/{limit}...")

                cap.release()
                status_text.success(f"Audit Complete: Processed {frame_idx} frames. Total damage instances flagged: {total_detected}")

    # ── Tab 3: Camera Capture ─────────────────────────────────
    with tab_cam:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>Live Camera Inspection</div>", unsafe_allow_html=True)
        st.caption("Capture pavement or road surface directly from your device camera:")
        camera_img = st.camera_input("Capture Roadway Frame")

        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Analyzing snapshot..."):
                result, annotated_rgb = process_image(img_bgr, model, severity_est, conf_threshold, iou_threshold, selected_device)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #71717A; margin-bottom: 6px;'>RAW CAMERA FRAME</div>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_c2:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #71717A; margin-bottom: 6px;'>AI DAMAGE OVERLAY</div>", unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)

            st.info(f"Analysis: {result['num_detections']} defect(s) detected in camera frame.")


if __name__ == "__main__":
    main()
