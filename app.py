"""
Streamlit Inference Dashboard
Multimodal: EfficientNet-B3 + YOLOv8n-pose → Fusion Classifier
"""

import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ═════════════════════════════════════════════════════════════════
#  ✏️  CONFIGURATION — edit these two lines to match your filenames
# ═════════════════════════════════════════════════════════════════
CLASSIFIER_WEIGHTS = Path(__file__).parent / "best_model.pth"   # ← rename to your .pth file
YOLO_WEIGHTS       = Path(__file__).parent / "yolov8n-pose.pt"  # ← rename to your YOLO .pt file

# Class names (driver distraction — 10 classes)
CLASS_NAMES = [
    "Normal Driving",
    "Texting - Right",
    "Talking on Phone - Right",
    "Texting - Left",
    "Talking on Phone - Left",
    "Operating Radio",
    "Drinking",
    "Reaching Behind",
    "Hair and Makeup",
    "Talking to Passenger",
]
NUM_CLASSES   = len(CLASS_NAMES)   # 10
YOLO_FEAT_DIM = 51                 # 17 keypoints × 3 (x, y, conf)

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Distraction Classifier",
    page_icon="🚗",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.6rem 2rem; border-radius: 12px;
        margin-bottom: 1.8rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.95rem; }

    .section-card {
        background: #f8f9fb; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem;
    }
    .compare-label {
        text-align: center; font-weight: 600; font-size: 0.85rem;
        color: #64748b; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.5rem;
    }
    .pred-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 8px; padding: 0.8rem 1rem; margin-top: 0.6rem;
        text-align: center;
    }
    .pred-box .pred-class { font-size: 1.25rem; font-weight: 700; color: #1e40af; }
    .pred-box .pred-conf  { font-size: 0.88rem; color: #64748b; margin-top: 2px; }
    .soft-divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }
    .warn-box {
        background: #fffbeb; border: 1px solid #fcd34d;
        border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.88rem; color: #92400e;
    }
    .model-status {
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.85rem; color: #166534;
        margin-bottom: 0.8rem;
    }
    .model-status-err {
        background: #fef2f2; border: 1px solid #fecaca;
        border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.85rem; color: #991b1b;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚗 Driver Distraction Classifier</h1>
    <p>EfficientNet-B3 visual features fused with YOLOv8n-pose keypoints → multimodal classification.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Model Definition (mirrors training code exactly)
# ─────────────────────────────────────────────────────────────────
class EfficientNetMultimodal(nn.Module):
    def __init__(self, num_classes=10, yolo_feat_dim=51):
        super().__init__()

        # 1. EfficientNet-B3 Backbone
        self.backbone = models.efficientnet_b3(weights=None)
        cnn_out_dim = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Identity()

        # 2. YOLO MLP Branch
        self.yolo_mlp = nn.Sequential(
            nn.Linear(yolo_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # 3. Fusion Classifier  (1536 + 64 = 1600)
        fused_dim = cnn_out_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img, x_yolo):
        if x_img.size(1) == 1:
            x_img = x_img.repeat(1, 3, 1, 1)
        x_cnn      = self.backbone(x_img)
        x_yolo_out = self.yolo_mlp(x_yolo)
        fused      = torch.cat([x_cnn, x_yolo_out], dim=1)
        return self.classifier(fused)


# ─────────────────────────────────────────────────────────────────
# Image transform (same as training)
# ─────────────────────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ─────────────────────────────────────────────────────────────────
# Sidebar — runtime settings only (no file uploaders)
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Runtime Settings")

    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

    device_choice = st.radio("Device", ["Auto (GPU if available)", "CPU only"])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_choice == "Auto (GPU if available)"
        else torch.device("cpu")
    )
    st.caption(f"Running on: **{str(device).upper()}**")




# ─────────────────────────────────────────────────────────────────
# Load classifier from disk
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier(weights_path: str, device_str: str):
    dev = torch.device(device_str)
    net = EfficientNetMultimodal(num_classes=NUM_CLASSES, yolo_feat_dim=YOLO_FEAT_DIM)

    if not Path(weights_path).exists():
        return net.to(dev).eval(), False   # demo mode with random weights

    state = torch.load(weights_path, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state, strict=False)
    net.to(dev)
    net.eval()
    return net, True


# ─────────────────────────────────────────────────────────────────
# Load YOLOv8 pose from disk
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics not installed. Run: pip install ultralytics"

    path = Path(weights_path)
    if path.exists():
        try:
            return YOLO(str(path)), None
        except Exception as e:
            return None, str(e)

    # File not found → try auto-download of the public yolov8n-pose
    try:
        return YOLO("yolov8n-pose.pt"), None
    except Exception as e:
        return None, f"File not found and auto-download failed: {e}"


# ─────────────────────────────────────────────────────────────────
# Load both models at startup
# ─────────────────────────────────────────────────────────────────
with st.spinner("Loading EfficientNet-B3 multimodal classifier…"):
    net, classifier_loaded = load_classifier(str(CLASSIFIER_WEIGHTS), str(device))

with st.spinner("Loading YOLOv8n-pose…"):
    yolo_model, yolo_err = load_yolo(str(YOLO_WEIGHTS))

# Status banners
if classifier_loaded:
    st.markdown(f'<div class="model-status">✅ Classifier loaded — <code>{CLASSIFIER_WEIGHTS.name}</code></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="model-status-err">⚠️ <code>{CLASSIFIER_WEIGHTS.name}</code> not found — running with random weights (demo mode).</div>', unsafe_allow_html=True)

if yolo_err:
    st.markdown(f'<div class="model-status-err">❌ YOLO error: {yolo_err}</div>', unsafe_allow_html=True)
    st.stop()
else:
    yolo_label = YOLO_WEIGHTS.name if YOLO_WEIGHTS.exists() else "yolov8n-pose.pt (auto-downloaded)"
    st.markdown(f'<div class="model-status">✅ YOLOv8n-pose loaded — <code>{yolo_label}</code></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Extract 51-dim keypoint vector from a BGR frame
# ─────────────────────────────────────────────────────────────────
def extract_yolo_features(frame_bgr):
    results        = yolo_model(frame_bgr, verbose=False)
    keypoints_data = results[0].keypoints

    if keypoints_data is None or len(keypoints_data.data) == 0:
        return np.zeros(YOLO_FEAT_DIM, dtype=np.float32)

    kpts = keypoints_data.data[0].cpu().numpy()   # (17, 3)
    flat = kpts.flatten()                          # (51,)

    if flat.size < YOLO_FEAT_DIM:
        flat = np.pad(flat, (0, YOLO_FEAT_DIM - flat.size))
    else:
        flat = flat[:YOLO_FEAT_DIM]

    return flat.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Single-frame prediction
# ─────────────────────────────────────────────────────────────────
def predict_frame(frame_bgr):
    """Returns (class_idx, class_name, confidence, probs_array, pose_detected)."""
    pil_img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(device)

    yolo_feat     = extract_yolo_features(frame_bgr)
    pose_detected = bool(yolo_feat.any())
    yolo_tensor   = torch.tensor(yolo_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(img_tensor, yolo_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx  = int(np.argmax(probs))
    name = CLASS_NAMES[idx] if idx < NUM_CLASSES else f"Class_{idx}"
    return idx, name, float(probs[idx]), probs, pose_detected


# ─────────────────────────────────────────────────────────────────
# Overlay annotation on frame
# ─────────────────────────────────────────────────────────────────
def overlay_label(frame_bgr, label, conf, above_threshold, pose_detected):
    out   = frame_bgr.copy()
    color = (0, 200, 80) if above_threshold else (0, 140, 255)
    text  = f"{label}  {conf * 100:.1f}%"
    if not pose_detected:
        text += "  [no pose]"
    cv2.rectangle(out, (0, 0), (frame_bgr.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, color, 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_video, tab_image, tab_results = st.tabs(
    ["🎥  Video", "🖼️  Image", "📊  Results / Data"]
)


# ───────────────────────────────────────────────────────────────────
# TAB 1 — VIDEO
# ───────────────────────────────────────────────────────────────────
with tab_video:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload a Video")
    video_file = st.file_uploader(
        "Supported: MP4, AVI, MOV, MKV",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    run_video = st.button("▶  Run on Video", use_container_width=True, key="run_video")
    st.markdown("</div>", unsafe_allow_html=True)

    if video_file and run_video:
        suffix = Path(video_file.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            in_path = tmp.name

        out_path = in_path.replace(suffix, "_output.mp4")

        cap          = cv2.VideoCapture(in_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress      = st.progress(0, text="Processing frames…")
        frame_records = []
        frame_idx     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            idx, cls, conf, probs, pose_det = predict_frame(frame)
            above = conf >= conf_threshold

            writer.write(overlay_label(frame, cls, conf, above, pose_det))

            frame_records.append({
                "Frame":           frame_idx,
                "Predicted Class": cls,
                "Confidence":      round(conf, 4),
                "Above Threshold": above,
                "Pose Detected":   pose_det,
                **{f"P({CLASS_NAMES[i]})": round(float(p), 4) for i, p in enumerate(probs)},
            })
            frame_idx += 1
            progress.progress(
                min(frame_idx / max(total_frames, 1), 1.0),
                text=f"Frame {frame_idx} / {total_frames}",
            )

        cap.release()
        writer.release()
        progress.empty()

        df = pd.DataFrame(frame_records)
        st.session_state.results_df = df

        pose_pct  = df["Pose Detected"].mean() * 100
        st.success(
            f"✅ Done! Processed **{frame_idx} frames** — "
            f"pose detected in **{pose_pct:.1f}%** of frames."
        )

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        col_orig, col_out = st.columns(2, gap="medium")
        with col_orig:
            st.markdown('<p class="compare-label">Original</p>', unsafe_allow_html=True)
            st.video(in_path)
        with col_out:
            st.markdown('<p class="compare-label">Model Output</p>', unsafe_allow_html=True)
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇  Download Output Video", data=f,
                    file_name=f"output_{video_file.name}",
                    mime="video/mp4", use_container_width=True,
                )

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.markdown("**Frame-level summary**")
        st.dataframe(
            df[["Frame", "Predicted Class", "Confidence", "Above Threshold", "Pose Detected"]],
            use_container_width=True, height=240,
        )
        st.caption("Full probability columns available in the **Results / Data** tab.")

        try:
            os.unlink(in_path)
        except OSError:
            pass

    elif video_file:
        st.info("Press **▶ Run on Video** to start.", icon="ℹ️")


# ───────────────────────────────────────────────────────────────────
# TAB 2 — IMAGE
# ───────────────────────────────────────────────────────────────────
with tab_image:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload an Image")
    image_file = st.file_uploader(
        "Supported: JPG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )
    run_image = st.button("▶  Run on Image", use_container_width=True, key="run_image")
    st.markdown("</div>", unsafe_allow_html=True)

    if image_file and run_image:
        pil_orig  = Image.open(image_file).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLOv8-pose + EfficientNet-B3…"):
            idx, cls, conf, probs, pose_det = predict_frame(frame_bgr)

        above = conf >= conf_threshold

        annotated_bgr = overlay_label(frame_bgr, cls, conf, above, pose_det)
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        col_orig, col_out = st.columns(2, gap="medium")
        with col_orig:
            st.markdown('<p class="compare-label">Original</p>', unsafe_allow_html=True)
            st.image(pil_orig, use_container_width=True)
        with col_out:
            st.markdown('<p class="compare-label">Model Output</p>', unsafe_allow_html=True)
            st.image(annotated_pil, use_container_width=True)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                "⬇  Download Output Image",
                data=buf.getvalue(),
                file_name=f"output_{image_file.name}",
                mime="image/png",
                use_container_width=True,
            )

        emoji       = "🟢" if above else "🟡"
        pose_badge  = "✓ pose detected" if pose_det else "⚠ no pose — zeros used"
        thresh_note = "✓ above threshold" if above else "⚠ below threshold"
        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-class">{emoji} {cls}</div>
            <div class="pred-conf">
                Confidence: {conf * 100:.2f}% &nbsp;·&nbsp; {thresh_note}<br>
                <small>Keypoints: {pose_badge}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not pose_det:
            st.markdown(
                '<div class="warn-box">⚠️ YOLOv8 did not detect a person in this image. '
                'The YOLO MLP branch received a zero vector — prediction may be less reliable.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.markdown("**Class probability distribution**")
        prob_df = pd.DataFrame({
            "Class":       CLASS_NAMES,
            "Probability": [round(float(p), 4) for p in probs],
        }).set_index("Class")
        st.bar_chart(prob_df)

        record = {
            "Source":          image_file.name,
            "Predicted Class": cls,
            "Confidence":      round(conf, 4),
            "Above Threshold": above,
            "Pose Detected":   pose_det,
            **{f"P({CLASS_NAMES[i]})": round(float(p), 4) for i, p in enumerate(probs)},
        }
        st.session_state.results_df = pd.concat(
            [st.session_state.results_df, pd.DataFrame([record])],
            ignore_index=True,
        )

    elif image_file:
        st.info("Press **▶ Run on Image** to start.", icon="ℹ️")


# ───────────────────────────────────────────────────────────────────
# TAB 3 — RESULTS
# ───────────────────────────────────────────────────────────────────
with tab_results:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Inference Results")
    st.caption("Accumulated across all video and image runs in this session.")
    st.markdown("</div>", unsafe_allow_html=True)

    df = st.session_state.results_df

    if df.empty:
        st.info("No results yet. Run inference on a video or image first.", icon="ℹ️")
    else:
        search_col, _ = st.columns([2, 3])
        with search_col:
            search = st.text_input("🔍 Filter rows", placeholder="e.g. Drinking")

        display_df = df
        if search:
            mask = df.apply(
                lambda c: c.astype(str).str.contains(search, case=False, na=False)
            ).any(axis=1)
            display_df = df[mask]

        st.dataframe(display_df, use_container_width=True, height=320)
        st.caption(f"Showing **{len(display_df)}** of **{len(df)}** rows.")

        if "Pose Detected" in df.columns:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total rows",      len(df))
            m2.metric("Pose detected",   f"{df['Pose Detected'].mean() * 100:.1f}%")
            m3.metric("Above threshold", f"{df['Above Threshold'].mean() * 100:.1f}%")

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.markdown("**Export full results:**")
        col_csv, col_xml, col_clear, _ = st.columns([1, 1, 1, 2])

        with col_csv:
            st.download_button(
                "⬇  CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_xml:
            def df_to_xml(dataframe: pd.DataFrame) -> bytes:
                lines = ['<?xml version="1.0" encoding="utf-8"?>', "<results>"]
                for _, row in dataframe.iterrows():
                    lines.append("  <record>")
                    for col, val in row.items():
                        tag = (
                            str(col)
                            .replace(" ", "_")
                            .replace("(", "")
                            .replace(")", "")
                            .replace("-", "_")
                        )
                        lines.append(f"    <{tag}>{val}</{tag}>")
                    lines.append("  </record>")
                lines.append("</results>")
                return "\n".join(lines).encode("utf-8")

            st.download_button(
                "⬇  XML",
                data=df_to_xml(df),
                file_name="results.xml",
                mime="application/xml",
                use_container_width=True,
            )

        with col_clear:
            if st.button("🗑  Clear Results", use_container_width=True):
                st.session_state.results_df = pd.DataFrame()
                st.rerun()