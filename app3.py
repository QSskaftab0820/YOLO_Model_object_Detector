# # app.py
# import streamlit as st
# import cv2
# import torch
# from ultralytics import YOLO
# from collections import Counter
# from PIL import Image
# import numpy as np
# import tempfile
# import os

# # ----------------------------
# # Page Config
# # ----------------------------
# st.set_page_config(page_title="YOLO Object Detection & Counting", layout="wide")

# st.title("🔍 YOLO Multi-Model Object Detection & Counting")

# # ----------------------------
# # Device Selection
# # ----------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# st.sidebar.info(f"Using device: **{device}**")

# # ----------------------------
# # Model Selection
# # ----------------------------
# st.sidebar.header("⚙️ Select YOLO Model")
# model_options = {
#     "YOLOv8 Nano": "yolov8n.pt",
#     "YOLOv8 Small": "yolov8s.pt",
#     "YOLOv8 Medium": "yolov8m.pt",
#     "YOLOv8 Large": "yolov8l.pt",
#     "YOLOv8 XLarge": "yolov8x.pt",
#     "YOLOv9 (if available)": "yolov9c.pt",  # optional
#     "YOLOv12 Nano (HuggingFace)": "yolov12n.pt"  # custom download
# }
# selected_model = st.sidebar.selectbox("Choose YOLO model:", list(model_options.keys()))

# # Load model
# @st.cache_resource
# def load_model(weight_path):
#     return YOLO(weight_path)

# weights = model_options[selected_model]
# model = load_model(weights)

# # ----------------------------
# # File Uploader
# # ----------------------------
# st.sidebar.header("📂 Upload File")
# source_type = st.sidebar.radio("Select source:", ["Image", "Video"])

# if source_type == "Image":
#     uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# else:
#     uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

# # ----------------------------
# # Detection & Counting
# # ----------------------------
# def detect_and_count(img, conf=0.5):
#     results = model(img, conf=conf, device=device)
#     annotated = results[0].plot()  # BGR image with boxes
#     boxes = results[0].boxes
#     labels = results[0].names

#     # Count detected objects
#     detected_labels = [labels[int(cls)] for cls in boxes.cls]
#     counts = Counter(detected_labels)

#     return annotated, counts

# # ----------------------------
# # Run on Image
# # ----------------------------
# if source_type == "Image" and uploaded_file:
#     img = Image.open(uploaded_file)
#     img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#     st.subheader("📸 Uploaded Image")
#     st.image(img, caption="Original", use_column_width=True)

#     # Detection
#     annotated, counts = detect_and_count(img_bgr)

#     st.subheader("🟢 Detected Objects")
#     st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected & Labeled", use_column_width=True)

#     st.subheader("📊 Object Counts")
#     st.write(dict(counts))

# # ----------------------------
# # Run on Video
# # ----------------------------
# elif source_type == "Video" and uploaded_file:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
#     video_path = tfile.name

#     cap = cv2.VideoCapture(video_path)
#     stframe = st.empty()
#     all_counts = Counter()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated, counts = detect_and_count(frame)
#         all_counts.update(counts)

#         # Display frame
#         stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

#     cap.release()

#     st.subheader("📊 Final Object Counts (Video)")
#     st.write(dict(all_counts))

import cv2
import sys
print("✅ OpenCV version:", cv2.__version__, " (from:", cv2.__file__, ")")
print("✅ Python version:", sys.version)

# app.py
import streamlit as st
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from collections import OrderedDict, deque
import os

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="YOLO Multi-model + OpenCV Detector", layout="wide")
st.title("🦾 YOLO + OpenCV — Multi-object Detection, Tracking & Counting")

# -------------------------
# Helper utils
# -------------------------
def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"

def normalize_names(model):
    if model is None:
        return {}
    try:
        names = model.names
    except Exception:
        try:
            names = model.model.names
        except Exception:
            names = {}
    if isinstance(names, (list, tuple)):
        return {i: n for i, n in enumerate(names)}
    return dict(names)

def safe_box_extract(box):
    try:
        xy = box.xyxy.cpu().numpy().ravel()[:4].astype(int)
    except Exception:
        xy = np.array(box.xyxy).ravel()[:4].astype(int)
    x1,y1,x2,y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
    try:
        conf = float(box.conf.cpu().item())
    except Exception:
        conf = float(box.conf)
    try:
        cls = int(box.cls.cpu().item())
    except Exception:
        cls = int(box.cls)
    return x1,y1,x2,y2,cls,conf

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=80):
        self.nextID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, oid):
        if oid in self.objects: del self.objects[oid]
        if oid in self.disappeared: del self.disappeared[oid]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1,y1,x2,y2) in enumerate(rects):
            input_centroids[i] = (int((x1+x2)/2), int((y1+y2)/2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (r, c) in zip(rows, cols):
                if r in usedRows or c in usedCols: continue
                if D[r, c] > self.max_distance: continue
                oid = objectIDs[r]
                self.objects[oid] = tuple(input_centroids[c])
                self.disappeared[oid] = 0
                usedRows.add(r); usedCols.add(c)

            unusedRows = set(range(D.shape[0])) - usedRows
            for r in unusedRows:
                oid = objectIDs[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            unusedCols = set(range(D.shape[1])) - usedCols
            for c in unusedCols:
                self.register(tuple(input_centroids[c]))

        return self.objects

class TrackableObject:
    def __init__(self, oid, centroid, cls=None):
        self.oid = oid
        self.centroids = deque([centroid], maxlen=30)
        self.counted = False
        self.cls = cls

# -------------------------
# Model options
# -------------------------
MODEL_OPTIONS = {
    "yolov8n.pt": "YOLOv8-n (fast, small, general purpose)",
    "yolov8s.pt": "YOLOv8-s (balanced, better accuracy)",
    "yolo11n.pt": "YOLOv11-n (fast, rural field use, 480p–1080p)",
    "yolo11s.pt": "YOLOv11-s (better accuracy for insects/snakes)",
    "yolov12n": "YOLOv12-n (HuggingFace: very fast, experimental)",
    "custom": "Custom model — upload or provide path"
}

model_choice = st.sidebar.selectbox("Choose YOLO Model", list(MODEL_OPTIONS.keys()), format_func=lambda x: f"{x} — {MODEL_OPTIONS[x]}")
custom_model_path = ""
if model_choice == "custom":
    custom_model_path = st.sidebar.text_input("Path to custom .pt", "")

# -------------------------
# Input options
# -------------------------
st.sidebar.header("Input Source")
use_webcam = st.sidebar.checkbox("Use webcam (local only)", value=False)
uploaded = st.sidebar.file_uploader("Upload video", type=["mp4","mov","avi"])
stream_url = st.sidebar.text_input("RTSP/HTTP stream URL (optional)", "")

# Detection params
st.sidebar.header("Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

# -------------------------
# Model loader
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(path, hf=False):
    try:
        if hf:
            return YOLO.from_pretrained(path)
        return YOLO(path)
    except Exception:
        return None

if model_choice == "custom":
    model_path = custom_model_path.strip()
    model = load_model(model_path)
elif model_choice == "yolov12n":
    model = load_model("nielsr/yolov12n", hf=True)
else:
    model = load_model(model_choice)

if model is None:
    st.error("⚠️ Failed to load model. Check path or Hugging Face link.")
else:
    st.sidebar.success(f"✅ Model loaded: {model_choice}")

names = normalize_names(model)
device = device_name()

# -------------------------
# Session state
# -------------------------
if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker()
if "tracks" not in st.session_state:
    st.session_state.tracks = {}
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "processing" not in st.session_state:
    st.session_state.processing = False

# -------------------------
# Sidebar live detection
# -------------------------
detected_area = st.sidebar.empty()

# -------------------------
# Processing loop
# -------------------------
def process_capture(cap):
    fps_time = time.time()
    frame_idx = 0
    st.session_state.counts = {}
    st.session_state.tracks = {}
    st.session_state.tracker = CentroidTracker()

    frame_area = st.empty()

    while cap.isOpened() and st.session_state.processing:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # inference
        results = model.predict(rgb, imgsz=640, conf=conf_thresh, iou=iou_thresh, device=device, verbose=False)
        boxes, cls_ids, confs = [], [], []
        if len(results) > 0:
            r = results[0]
            for box in getattr(r, "boxes", []):
                x1,y1,x2,y2,cls,conf = safe_box_extract(box)
                if conf < conf_thresh: continue
                boxes.append((x1,y1,x2,y2))
                cls_ids.append(cls)
                confs.append(conf)

        objects = st.session_state.tracker.update(boxes)

        # Track & count unique IDs
        for oid, centroid in objects.items():
            assigned_idx = None
            min_dist = float("inf")
            for i, b in enumerate(boxes):
                bx = int((b[0]+b[2]) / 2); by = int((b[1]+b[3]) / 2)
                d = (bx - centroid[0])**2 + (by - centroid[1])**2
                if d < min_dist:
                    min_dist = d; assigned_idx = i

            to = st.session_state.tracks.get(oid, None)
            if to is None:
                cls_for_id = cls_ids[assigned_idx] if assigned_idx is not None else None
                to = TrackableObject(oid, centroid, cls_for_id)
                st.session_state.tracks[oid] = to
            else:
                to.centroids.append(centroid)
                if to.cls is None and assigned_idx is not None:
                    to.cls = cls_ids[assigned_idx]

            if not to.counted and to.cls is not None:
                cls_name = names.get(to.cls, str(to.cls))
                st.session_state.counts[cls_name] = st.session_state.counts.get(cls_name, 0) + 1
                to.counted = True

        # Draw detections
        out = frame.copy()
        for (b, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1,y1,x2,y2 = b
            name = names.get(cls_id, str(cls_id))
            lbl = f"{name} {conf:.2f}"
            cv2.rectangle(out, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(out, lbl, (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Draw counts
        y0 = 30
        for i, (cls_name, count) in enumerate(st.session_state.counts.items()):
            cv2.putText(out, f"{cls_name}: {count}", (10, y0 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS
        fps = 1.0 / (time.time() - fps_time) if (time.time() - fps_time) > 0 else 0.0
        fps_time = time.time()
        cv2.putText(out, f"FPS: {fps:.1f}", (10, out.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Show in app
        frame_area.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Live sidebar update
        if st.session_state.counts:
            sidebar_text = "### Live Detected Objects\n"
            for k,v in st.session_state.counts.items():
                sidebar_text += f"- {k}: {v}\n"
            detected_area.markdown(sidebar_text)

    cap.release()
    st.success("✅ Video finished. Final counts:")
    st.write(st.session_state.counts if st.session_state.counts else "No objects detected.")

# -------------------------
# Start/Stop buttons
# -------------------------
col1, col2 = st.sidebar.columns(2)
if col1.button("▶ Start"):
    if model is None:
        st.error("Load a model first.")
    else:
        st.session_state.processing = True
        if use_webcam:
            cap = cv2.VideoCapture(0)
            process_capture(cap)
        elif uploaded is not None:
            tmp_path = f"temp_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f: f.write(uploaded.read())
            cap = cv2.VideoCapture(tmp_path)
            process_capture(cap)
            try: os.remove(tmp_path)
            except: pass
        elif stream_url:
            cap = cv2.VideoCapture(stream_url)
            process_capture(cap)
        else:
            st.info("Please provide a video source.")
if col2.button("⏹ Stop"):
    st.session_state.processing = False



