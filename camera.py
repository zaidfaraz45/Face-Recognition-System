import cv2
import torch
import torch.nn.functional as F
import os
import csv
from model import CNN
from datetime import datetime
from torchvision import transforms, datasets
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

# Load class names 
data_path = 'faces'
dataset = datasets.ImageFolder(root=data_path)
class_names = dataset.classes
num_classes = len(dataset.classes)
print("\nClasses:", class_names, end='\n\n')

# Load trained model 
model = CNN(num_classes).to(device)
model.load_state_dict(torch.load("face_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

ATTENDANCE_FILE = "attendance/attendance.csv"
os.makedirs("attendance", exist_ok=True)

def mark_attendance(name):
    # Get current date and time
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Create CSV with header if it does not exist
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])

    # Read existing rows and skip the header row
    already_marked = False
    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip header row
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == date_str:
                already_marked = True
                break

    # Write new row if not already marked today
    if not already_marked:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        print(f"[✔] Attendance marked for {name} | Date: {date_str} | Time: {time_str}")
    else:
        print(f"[!] {name} already marked today ({date_str})")

# Predict face 
def predict_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(face_rgb)
    tensor   = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return class_names[idx.item()], conf.item()

# Camera loop 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("Camera started. Press Q or ESC to quit.\n")

marked_today = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(120, 120)
    )

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        name, conf = predict_face(face_crop)

        if conf >= 0.95:
            label     = f"{name}  {conf*100:.1f}%"
            box_color = (0, 200, 80)     # green

            if name not in marked_today:
                mark_attendance(name)
                marked_today.add(name)
        else:
            label     = f"Unknown  {conf*100:.1f}%"
            box_color = (0, 80, 220)     # red

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), box_color, -1)

        # Label text
        cv2.putText(
            frame, label,
            (x + 4, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Face count on screen
    cv2.putText(
        frame, f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (220, 220, 220), 2, cv2.LINE_AA
    )

    cv2.imshow("Face Recognition - Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('e'), ord('E'), 27):    # E or ESC to exit window
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nSession ended. Attendance saved to: {ATTENDANCE_FILE}")