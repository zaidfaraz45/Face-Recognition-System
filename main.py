import cv2
import torch
import torch.nn.functional as F
import os
import csv
from model import CNN
from datetime import datetime
from torchvision import transforms, datasets
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.85   # top-1 confidence must exceed this
MARGIN_THRESHOLD     = 0.20   # top-1 must beat top-2 by at least this margin
STABLE_FRAMES_NEEDED = 8      # consecutive agreeing frames before marking
ATTENDANCE_FILE      = "attendance/attendance.csv"
DATA_PATH            = "faces"
# ──────────────────────────────────────────────────────────────────────────────

# Device agnostic code
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

# Load class names
dataset     = datasets.ImageFolder(root=DATA_PATH)
class_names = dataset.classes
num_classes = len(class_names)
print("\nClasses:", class_names, end='\n\n')

# Load trained model
model = CNN(num_classes).to(device)
model.load_state_dict(torch.load("face_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

os.makedirs("attendance", exist_ok=True)


# ── Attendance helpers ────────────────────────────────────────────────────────

def mark_attendance(name: str) -> None:
    """Write one attendance row for `name` if not already marked today."""
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip header
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == date_str:
                print(f"[!] {name} already marked today ({date_str})")
                return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, date_str, time_str])
    print(f"[✔] Attendance marked for {name} | {date_str} {time_str}")


# ── Face prediction with ambiguity check ─────────────────────────────────────

def predict_face(face_bgr) -> tuple[str, float, bool]:
    """
    Returns (name, confidence, is_certain).

    is_certain is False when:
      - top-1 confidence < CONFIDENCE_THRESHOLD, OR
      - the margin between top-1 and top-2 < MARGIN_THRESHOLD  ← ambiguity fix
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor   = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]   # shape: (num_classes,)

    top2_conf, top2_idx = torch.topk(probs, k=min(2, num_classes))
    top1_conf  = top2_conf[0].item()
    top1_name  = class_names[top2_idx[0].item()]

    # Ambiguity: two classes are too close in probability
    if num_classes > 1:
        margin = (top2_conf[0] - top2_conf[1]).item()
    else:
        margin = 1.0   # only one class → no ambiguity possible

    is_certain = (top1_conf >= CONFIDENCE_THRESHOLD) and (margin >= MARGIN_THRESHOLD)

    return top1_name, top1_conf, is_certain


# ── Camera loop ───────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("Camera started. Press E or ESC to quit.\n")

marked_today     : set[str]        = set()
# stable_detections: name → consecutive frames that agree on this name
stable_detections: dict[str, int]  = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(100, 100)
    )

    # Names detected in THIS frame — used to decay counters for absent faces
    names_this_frame: set[str] = set()

    for (x, y, w, h) in faces:
        # Filter non-face aspect ratios
        if not (0.75 < w / float(h) < 1.3):
            continue

        # Crop with padding
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        y1 = max(0, y - pad_h);          y2 = min(frame.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w);          x2 = min(frame.shape[1], x + w + pad_w)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        pred_name, conf, is_certain = predict_face(face_crop)

        if is_certain:
            # ── Stable-frame accumulation (FIX: actually update the set) ──
            names_this_frame.add(pred_name)
            stable_detections[pred_name] = stable_detections.get(pred_name, 0) + 1

            label     = f"{pred_name}  {conf*100:.1f}%"
            box_color = (0, 200, 80)   # green

            # Only mark after STABLE_FRAMES_NEEDED consecutive certain detections
            if (stable_detections[pred_name] >= STABLE_FRAMES_NEEDED
                    and pred_name not in marked_today):
                mark_attendance(pred_name)
                marked_today.add(pred_name)
        else:
            # Ambiguous or low-confidence → show "Unknown" and reset counter
            stable_detections.pop(pred_name, None)
            label     = f"Unknown  {conf*100:.1f}%"
            box_color = (0, 80, 220)   # red/blue

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), box_color, -1)

        # Label text
        cv2.putText(frame, label, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Decay stable counters for faces no longer on screen (FIX: was broken before)
    for tracked_name in list(stable_detections.keys()):
        if tracked_name not in names_this_frame:
            stable_detections[tracked_name] = max(0, stable_detections[tracked_name] - 1)
            if stable_detections[tracked_name] == 0:
                del stable_detections[tracked_name]

    # HUD: face count
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    cv2.imshow("Face Recognition - Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('e'), ord('E'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nSession ended. Attendance saved to: {ATTENDANCE_FILE}")