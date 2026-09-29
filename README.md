# 🛣️ RoadSafe: Deep Learning-Based Road Damage Detection & Severity Assessment System

> **A Lightweight, Real-Time Edge Vision Pipeline for Automated Road Surface Auditing**  
> *Developed with YOLOv8-nano, OpenCV, and Domain-Specific Geometric Severity Heuristics on the RDD2022 India Dataset*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Device: CPU & CUDA](https://img.shields.io/badge/Hardware-CPU%20%7C%20CUDA%20Edge-success)](https://developer.nvidia.com/cuda-toolkit)

---

## 🖥️ Minimalist Web Application Interface

![RoadSafe Minimalist Interface](assets/app_interface.png)

> **Figure 1**: RoadSafe Vercel/Apple-inspired minimalist interface running on local hardware. Features real-time defect tallying (Total Defects, Potholes, Cracks, High-Risk Hazards), sub-170ms CPU inference latency telemetry, side-by-side original vs. detection overlay, and 1-click evidence export.

---

## 📌 Table of Contents
- [1. Abstract & Motivation](#-1-abstract--motivation)
- [2. Problem Formulation & Defect Taxonomy](#-2-problem-formulation--defect-taxonomy)
- [3. End-to-End System Architecture](#-3-end-to-end-system-architecture)
- [4. Dataset Engineering & Preprocessing](#-4-dataset-engineering--preprocessing)
- [5. Model Architecture & Training Methodology](#-5-model-architecture--training-methodology)
- [6. Mathematical Severity Assessment Framework](#-6-mathematical-severity-assessment-framework)
- [7. Experimental Results & Performance Benchmarks](#-7-experimental-results--performance-benchmarks)
- [8. Real-Time Video & Live Webcam Inference](#-8-real-time-video--live-webcam-inference)
- [9. Repository Structure](#-9-repository-structure)
- [10. Quickstart & Reproduction Guide](#-10-quickstart--reproduction-guide)
- [11. Key Engineering Insights & Lessons Learned](#-11-key-engineering-insights--lessons-learned)
- [12. Future Scope & Extensions](#-12-future-scope--extensions)
- [13. References & Citation](#-13-references--citation)

---

## 📖 1. Abstract & Motivation

Municipal roadway networks form the backbone of national commerce and daily transit. However, timely maintenance of asphalt pavements remains a global engineering bottleneck. Traditional survey methods rely primarily on manual foot inspections or specialized inspection vehicles equipped with expensive LiDAR rigs (costing upwards of **$150,000 per unit**). These approaches are:
1. **Labor-Intensive & Cost-Prohibitive**: Impractical for continuous, large-scale municipal monitoring.
2. **Subjective**: High inter-observer variance in defect reporting.
3. **Slow & Reactive**: Minor fissures frequently deteriorate into structural potholes before repair tickets are issued.

**RoadSafe** addresses this challenge by delivering an automated, edge-deployable computer vision pipeline. Using a fine-tuned **YOLOv8-nano** model combined with a **domain-aware geometric severity estimator**, RoadSafe turns ordinary commercial dashcam footage or smartphone camera streams into continuous, objective road health inspection data.

### Key Engineering Goals
- **Edge Deployment**: Target an ultra-lightweight footprint (< 6 MB model) runnable on commodity laptop CPUs, embedded systems (e.g., Raspberry Pi 5, NVIDIA Jetson), or mobile processors without requiring cloud offloading.
- **Explainable Severity Ranking**: Move beyond pure bounding box coordinates by translating geometric damage proportions into actionable maintenance priority classes (`Low`, `Medium`, `High`).
- **Resilient Batch & Stream Processing**: Support single images, multi-thousand image directories, dashcam video files, and real-time webcam streams with fault-tolerant error boundaries.

---

## 🔍 2. Problem Formulation & Defect Taxonomy

The objective is formulated as a simultaneous **multi-class object detection** and **heuristic severity regression** problem. Given an input RGB frame $I \in \mathbb{R}^{H \times W \times 3}$, the model predicts a set of $N$ defect instances:

$$\mathcal{D} = \{ (c_i, p_i, \mathbf{b}_i, s_i) \}_{i=1}^N$$

where:
- $c_i \in \{0, 1, 2, 3\}$ is the predicted damage category.
- $p_i \in [0, 1]$ is the prediction confidence score.
- $\mathbf{b}_i = [x_1, y_1, x_2, y_2]$ represents the spatial bounding box coordinates.
- $s_i \in \{\text{Low}, \text{Medium}, \text{High}\}$ is the deterministic severity level.

### Defect Classes (Based on RDD2022 Benchmark)

| Class Index | Defect Category | RDD2022 Code | Structural Risk & Real-World Mechanism |
|:---:|:---|:---:|:---|
| **0** | **Pothole** | `D40`, `D43`, `D44` | High immediate safety risk; causes vehicle suspension damage, blowouts, and accidents. |
| **1** | **Longitudinal Crack** | `D00`, `D01` | Runs parallel to road centerline; caused by poor lane joint construction or heavy wheel-path stress. |
| **2** | **Alligator Crack** | `D20` | Interconnected hexagonal fatigue cracks resembling reptile skin; indicates subgrade structural failure. |
| **3** | **Transverse Crack** | `D10`, `D11` | Runs perpendicular to travel direction; caused by thermal shrinkage of asphalt during severe weather cycles. |

---

## 🏗️ 3. End-to-End System Architecture

```
                       ┌────────────────────────────────────────┐
                       │  Input Source: Image / Video / Webcam  │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │       Input Preprocessing Pipeline     │
                       │  • Letterbox Resizing to 640x640       │
                       │  • Normalization & BGR -> RGB Convert  │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │        YOLOv8-nano Deep Backbone       │
                       │  • Modified CSPDarknet53 + C2f Blocks  │
                       │  • PANet Neck (Multi-Scale Feature FPN)│
                       │  • Decoupled Anchor-Free Detection Head│
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │   Post-Processing & Filtering (NMS)    │
                       │  • Confidence Filtering (conf >= 0.25) │
                       │  • Non-Maximum Suppression (IoU = 0.45)│
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │   Geometric Severity Engine (Heuristic)│
                       │  • Potholes: Normalized Area Ratio     │
                       │  • Longitudinal Cracks: Height Ratio   │
                       │  • Transverse Cracks: Width Ratio      │
                       │  • Alligator Cracks: Area Ratio        │
                       └───────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│          Visual Render Engine         │   │         Structured Audit Log          │
│ • Color-Coded Bounding Boxes          │   │ • Frame-by-frame JSON metadata        │
│ • Severity Badges & Confidence Tags   │   │ • Aggregate damage counts             │
│ • Real-time HUD (FPS, Damage Counter) │   │ • Ready for Municipal Work-Order ERPs │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
```

---

## 📊 4. Dataset Engineering & Preprocessing

The model is trained on the India subset of the **Crowdsensing-based Road Damage Detection Challenge (RDD2022)**. 

### Data Ingestion & Conversion Pipeline (`data_preparation.py`)
1. **Format Standardization**: The raw dataset contains XML annotations adhering to the Pascal VOC format. We parse `<bndbox>` coordinates and transform them to YOLO normalized center-based coordinates:
   $$x_{center} = \frac{x_{min} + x_{max}}{2 \cdot W}, \quad y_{center} = \frac{y_{min} + y_{max}}{2 \cdot H}, \quad w = \frac{x_{max} - x_{min}}{W}, \quad h = \frac{y_{max} - y_{min}}{H}$$
2. **Class Mapping & Filtering**: Consolidated sub-variants (e.g., `D00` and `D01` $\to$ `LongitudinalCrack`; `D40`, `D43`, and `D44` $\to$ `Pothole`). Rare non-damage classes (such as `D50`) were omitted.
3. **Stratified Splitting**: Divided into **80% Training (1,224 images)**, **10% Validation (153 images)**, and **10% Test (153 images)**, stratified on the dominant defect class per image to ensure uniform distribution across splits.

```bash
# To regenerate dataset from raw unzipped RDD2022 data:
python data_preparation.py --raw-dir /path/to/raw_rdd2022 --output-dir dataset
```

---

## 🧠 5. Model Architecture & Training Methodology

### Why YOLOv8-nano (`yolov8n`)?
For an edge road safety audit system, inference latency and memory footprint are as critical as raw mean Average Precision (mAP). YOLOv8-nano delivers an optimal Pareto frontier:
- **Parameter Count**: Only **3.01 Million parameters** (5.97 MB footprint).
- **Decoupled Head**: Decouples classification and bounding box regression tasks, accelerating convergence.
- **Anchor-Free Architecture**: Predicts the center of objects directly, handling varied defect aspect ratios without hand-tuned anchor box priors.
- **Loss Formulation**: Complete IoU Loss ($\mathcal{L}_{CIoU}$) + Distribution Focal Loss ($\mathcal{L}_{DFL}$) for box regression, and Binary Cross-Entropy ($\mathcal{L}_{BCE}$) for multi-class classification.

### Domain-Specific Hyperparameters & Augmentations (`train.py`)

```python
# Key configurations in train.py:
optimizer       = "AdamW"          # Weight decay regularized optimizer
lr0             = 0.001            # Initial learning rate
lrf             = 0.01             # Cosine learning rate decay floor
weight_decay    = 0.0005           # L2 Regularization penalty
imgsz           = 640              # Standard evaluation resolution
mosaic          = 1.0              # 4-image mosaic composition
mixup           = 0.1              # Linear image blending probability
fliplr          = 0.5              # Horizontal flip (left-to-right invariant)
flipud          = 0.0              # DISABLED (Critical Domain Observation)
```

> 💡 **Key Computer Vision Insight: `flipud = 0.0`**  
> In generic object detection (e.g., COCO), vertical flip augmentation is frequently enabled. However, in vehicular dashcam and road inspection imagery, **roads are physically constrained to the ground plane**. Flipping an image vertically creates an impossible physical scenario (the sky on the bottom and the road on top), which actively confuses spatial feature representation in early convolutional layers. Setting `flipud=0.0` eliminates this orientation noise.

---

## 📐 6. Mathematical Severity Assessment Framework

Standard object detectors output bounding boxes without context on whether a defect constitutes an emergency. RoadSafe couples detection with a **domain-grounded geometric severity engine** (`severity_estimator.py`).

Each defect category is evaluated against a class-specific geometric metric normalized against frame dimensions:

### 1. Potholes & Alligator Cracks (Area Occupancy Ratio)
Potholes and alligator cracking represent surface area degradation. The metric measures the surface footprint:
$$R_{area} = \frac{(x_2 - x_1) \cdot (y_2 - y_1)}{W_{img} \cdot H_{img}}$$

### 2. Longitudinal Cracks (Vertical Propagation Ratio)
Longitudinal cracks propagate along the line of travel. The metric evaluates vertical extent:
$$R_{height} = \frac{y_2 - y_1}{H_{img}}$$

### 3. Transverse Cracks (Horizontal Lane Span Ratio)
Transverse cracks cut across the roadway. The metric assesses lane width penetration:
$$R_{width} = \frac{x_2 - x_1}{W_{img}}$$

### Piecewise Severity Threshold Matrix

| Defect Class | Governing Metric | Low Severity (🟢) | Medium Severity (🟠) | High Severity (🔴) |
|:---|:---:|:---:|:---:|:---:|
| **Pothole** | $R_{area}$ | $R_{area} < 2\%$ | $2\% \le R_{area} \le 5\%$ | $R_{area} > 5\%$ |
| **Alligator Crack** | $R_{area}$ | $R_{area} < 3\%$ | $3\% \le R_{area} \le 8\%$ | $R_{area} > 8\%$ |
| **Longitudinal Crack** | $R_{height}$ | $R_{height} < 10\%$ | $10\% \le R_{height} \le 25\%$ | $R_{height} > 25\%$ |
| **Transverse Crack** | $R_{width}$ | $R_{width} < 10\%$ | $10\% \le R_{width} \le 25\%$ | $R_{width} > 25\%$ |

---

## 📈 7. Experimental Results & Performance Benchmarks

### Quantitative Detection Metrics on Held-Out Test Set (153 Images)

Evaluated at IoU threshold = 0.50 and confidence threshold = 0.25:

| Metric | Measured Value | Analysis & Practical Implications |
|:---|:---:|:---|
| **mAP@50** | **28.73%** | Solid performance across heavily cluttered, dusty, and shadowed real-world Indian road scenes. |
| **mAP@50-95** | **10.42%** | High bounding box overlap stringency across multi-scale defect boundaries. |
| **Precision** | **42.00%** | Minimizes false-alarm dispatches for road maintenance crews. |
| **Recall** | **36.30%** | Detects over one-third of subtle or distant fissures from single frames. |
| **Model Size** | **5.97 MB** | Easily fits inside device L3 cache or embedded flash storage. |

### Per-Class Performance Breakdown

| Defect Class | AP@50 | Precision | Recall | Support (Instances) |
|:---|:---:|:---:|:---:|:---:|
| **Alligator Crack** | **40.24%** | 44.3% | 50.0% | 54 |
| **Pothole** | **38.71%** | 52.4% | 48.7% | 337 |
| **Longitudinal Crack** | **7.24%** | 29.4% | 10.2% | 49 |
| **Transverse Crack** | *0.00%\** | 0.0% | 0.0% | 0\* |

*\*Note: High natural class imbalance in the original RDD2022 India dataset (zero transverse crack annotations present in the random test split).*

### Inference Latency & Hardware Benchmarks

| Hardware Platform | Backend | Precision | Mean Latency | Throughput | Edge Viability |
|:---|:---|:---:|:---:|:---:|:---:|
| **NVIDIA RTX 4050 Laptop GPU** | PyTorch / CUDA 12.4 | FP32 | **5.9 ms** (model) / **16.3 ms** (E2E) | **~61.5 FPS** | 🟢 Ultra-fluid 60 FPS real-time |
| **Intel Core i5 / AMD Ryzen CPU** | PyTorch / OpenMP | FP32 | **80.4 ms** | **~12.4 FPS** | 🟢 Real-time dashcam stream |
| **ONNX Runtime (CPU)** *(est.)* | ONNX / INT8 Quantized | INT8 | **~25–35 ms** | **~30–40 FPS** | 🟢 Full 30 FPS on embedded CPU |

### Visual Artifacts & Analytics (`plots/`)

| Metric Progression & Loss Curves | Confusion Matrix | Class Distribution |
|:---:|:---:|:---:|
| ![Results](runs/detect/road_damage/results.png) | ![Confusion Matrix](runs/detect/road_damage/confusion_matrix.png) | ![Class Distribution](plots/class_distribution.png) |

---

## 🎥 8. Real-Time Video & Live Webcam Inference

RoadSafe supports video files (`.mp4`, `.avi`, `.mov`) and live camera streams with a dynamic HUD overlay:

```bash
# 1. Process sample pothole video with live preview window
python inference.py --video "demo video/mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4" --output results/

# 2. Run in headless mode (no GUI window, for servers or batch scripts)
python inference.py --video "path/to/dashcam.mp4" --output results/ --no-show

# 3. Live webcam feed (press 'q' in preview window to exit)
python inference.py --webcam --output results/
```

### Video HUD Features
- **Real-Time FPS Monitor**: Dynamically calculates running frames per second.
- **Active Defect Counter**: Instant readout of defects detected in the current frame.
- **Severity Footer**: Live summary of the highest-severity defects in the scene.
- **Audit Export**: Outputs both the annotated `.mp4` file and a machine-readable `video_log_<name>.json` audit log with frame counts, average FPS, and class/severity totals.

---

## 📁 9. Repository Structure

```
RoadSafe/
├── weights/
│   ├── best.pt                     # Production-ready trained weights (5.97 MB)
│   └── yolo26n.pt                  # Checkpoint weights
├── sample_images/                  # Curated sample test images for immediate demonstration
│   ├── India_009605.jpg            # Benchmark pothole + crack sample
│   ├── India_000017.jpg
│   ├── India_000147.jpg
│   ├── India_000268.jpg
│   └── India_000785.jpg
├── demo video/                     # Sample dashcam footage for video testing
│   └── mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4
├── docs/                           # Academic papers & deployment guides
│   ├── 3. Road Damage Detection and Severity Assessment.docx
│   └── HOW_TO_RUN_ON_ANOTHER_PC.txt
├── plots/                          # Analysis charts & detection collages
│   ├── class_distribution.png
│   ├── severity_distribution.png
│   └── detection_grid.png
├── runs/detect/road_damage/        # YOLOv8 training telemetry & evaluation curves
│   ├── results.png                 # Loss curves and mAP progression
│   ├── confusion_matrix.png        # Confusion matrix
│   ├── BoxPR_curve.png             # Precision-Recall curve
│   └── BoxF1_curve.png             # F1-Confidence curve
├── assets/                         # Application screenshots & interface UI assets
│   ├── app_interface.png           # Showcase screenshot of minimalist web app
│   └── app_full.png                # Full-page high-resolution audit view
├── RUN_APP.bat                     # 1-Click launcher: Start Minimalist Web Application
├── 1_RUN_DEMO.bat                  # 1-Click quickstart demo (Windows)
├── 2_RUN_INFERENCE.bat             # 1-Click batch inference on sample/test images
├── 3_INSTALL_DEPENDENCIES.bat      # 1-Click pip installer
├── 4_RUN_VIDEO.bat                 # 1-Click video & webcam interactive runner
├── app.py                          # Minimalist Streamlit web application
├── demo.py                         # Clean quickstart demonstration script
├── inference.py                    # Multi-source inference engine (Image, Dir, Video, Webcam)
├── severity_estimator.py           # Geometric severity heuristic calculation engine
├── evaluate.py                     # Validation and inference speed benchmark script
├── visualize.py                    # Analytical plot & grid generation utility
├── train.py                        # YOLOv8 fine-tuning script with early stopping
├── data_preparation.py             # VOC XML -> YOLO format converter and stratified splitter
├── dataset.yaml                    # YOLOv8 4-class dataset configuration
├── evaluation_report.json          # Machine-readable benchmark report
├── requirements.txt                # Exact dependency versions
└── .gitignore                      # Clean Git exclusions (omits heavy raw datasets)
```

---

## 🚀 10. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- CUDA-enabled GPU (Optional; CPU execution is fully supported out of the box)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/RoadSafe.git
cd RoadSafe

# Create and activate a virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Windows 1-Click Launchers (Zero-CLI)
If using Windows, you can double-click the `.bat` files directly:
- **`RUN_APP.bat`** *(Recommended)*: Launches the interactive minimalist web application in your default browser.
- **`1_RUN_DEMO.bat`**: Runs instant CLI detection on sample image (`sample_images/India_009605.jpg`).
- **`2_RUN_INFERENCE.bat`**: Runs batch detection and exports results to `results/`.
- **`3_INSTALL_DEPENDENCIES.bat`**: Automatically checks Python and installs `requirements.txt`.
- **`4_RUN_VIDEO.bat`**: Interactive menu to run on sample video, custom video, or webcam.

### Command-Line Usage

```bash
# 1. Launch the interactive web app:
streamlit run app.py

# 2. Run quickstart CLI demo:
python demo.py

# 2. Run inference on a specific image:
python inference.py --image sample_images/India_000017.jpg --output results/

# 3. Run inference on an entire directory:
python inference.py --image-dir sample_images/ --output results/

# 4. Run inference on a video:
python inference.py --video "demo video/mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4" --output results/

# 5. Run live webcam inference:
python inference.py --webcam 0 --output results/

# 6. Run benchmark evaluation:
python evaluate.py --split test --benchmark
```

---

## 💡 11. Key Engineering Insights & Lessons Learned

1. **Orientation Matters in Physical Domains**: Disabling vertical flip (`flipud=0.0`) is non-negotiable for dashcam vision. Treating aerial or satellite data differently from forward-facing vehicular perspectives is a foundational domain adaptation.
2. **Defensive Pipeline Engineering**: Production vision systems must never terminate abruptly due to a single malformed, corrupt, or unreadable frame in a multi-thousand image batch. Wrapping frame processing in individual try-except handlers with skipped-frame audit telemetry ensures high availability.
3. **Decoupled Severity Engine**: Decoupling the severity estimation from the neural network weights into an interpretable heuristic module (`severity_estimator.py`) allows municipal civil engineers to adjust severity thresholds on the fly without retraining or re-annotating the model.
4. **Graceful Fallbacks for Portable Code**: Scripts dynamically look for local sample data if the 96 MB training dataset is not unzipped, enabling seamless repository cloning and immediate verification by recruiters and collaborators.

---

## 🔮 12. Future Scope & Extensions

- **Temporal Tracking & Multi-Object Deduplication**: Integrate **ByteTrack** or **BoT-SORT** to track defects across consecutive video frames so that a single pothole captured over 20 frames is registered as one defect instance in the database.
- **GPS Telemetry Integration & GIS Heatmaps**: Extract NMEA or Exif GPS coordinate metadata from dashcam files to project defect clusters onto OpenStreetMap / Mapbox layers.
- **Edge Quantization**: Export to **TensorRT (FP16)** and **OpenVINO (INT8)** to achieve >30 FPS real-time throughput on low-power devices like the Raspberry Pi 5.

---

## 📚 13. References & Citation

1. **RDD2022 Benchmark**:
   ```bibtex
   @article{arya2022crowdsensing,
     title={Crowdsensing-based road damage detection challenge 2022},
     author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
     journal={IEEE Big Data},
     year={2022}
   }
   ```
2. **Ultralytics YOLOv8**:
   ```bibtex
   @software{yolov8_ultralytics,
     author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
     title = {Ultralytics YOLOv8},
     version = {8.0.0},
     year = {2023},
     url = {https://github.com/ultralytics/ultralytics}
   }
   ```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
