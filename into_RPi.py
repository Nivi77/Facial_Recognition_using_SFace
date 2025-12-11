import cv2
from deepface import DeepFace
import numpy as np
import os
import time
import onnxruntime as ort
from numpy.linalg import norm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------
MODEL = "SFace"
THRESHOLD = 0.45
KNOWN_FACES_DIR = "knownfaces"
SKIP_FRAMES = 5
FACE_CHANGE_THRESHOLD = 0.50
FRAME_RESIZE_WIDTH = 320


# -----------------------------------------------------------
# CHECK GPU OR CPU
# -----------------------------------------------------------
print("\n====================================")
print("     ONNX Runtime Execution")
print("====================================")

model = DeepFace.build_model(MODEL)
try:
    session = ort.InferenceSession(
        model.onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    providers = session.get_providers()
except:
    providers = ["CPUExecutionProvider"]

if "CUDAExecutionProvider" in providers:
    print("Using GPU (CUDA)\n")
else:
    print("Using CPU\n")


# -----------------------------------------------------------
# LOAD KNOWN FACES
# -----------------------------------------------------------
known_names = []
known_embeddings = []

print("[INFO] Loading known faces...")

for fname in os.listdir(KNOWN_FACES_DIR):
    if fname.lower().endswith((".jpg", "jpeg", "png")):
        path = os.path.join(KNOWN_FACES_DIR, fname)
        emb = DeepFace.represent(path, model_name=MODEL, enforce_detection=False)[0]["embedding"]
        known_embeddings.append(np.array(emb))
        known_names.append(os.path.splitext(fname)[0])

print(f"[INFO] Loaded {len(known_names)} known faces.\n")


# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


# -----------------------------------------------------------
# SEND EMAIL NOTIFICATION
# -----------------------------------------------------------
def send_email(person_name, confidence, dist):
    sender_email = "sender_mail@gmail.com"
    app_password = "app_password"
    receiver_email = "receiver_mail.com"

    subject = f"[ALERT] {person_name} Identified"
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""Person Identified: {person_name}
      Time Detected: {time_now}
      Confidence: {confidence:.2f}%
      """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"[EMAIL SENT] Alert sent for: {person_name}")
    except Exception as e:
        print("[EMAIL ERROR]:", e)


# -----------------------------------------------------------
# CAMERA
# -----------------------------------------------------------
VIDEO_PATH = r"video_path"   # <-- change to your video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Could not open video file!")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cached_emb = None
frame_count = 0

prev_name = "No Face"
prev_color = (0, 255, 0)
prev_conf = 0
prev_dist = None


# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    scale = FRAME_RESIZE_WIDTH / w
    small = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Process only every SKIP_FRAMES frames
    if len(faces) > 0 and frame_count % SKIP_FRAMES == 0:
        (x, y, fw, fh) = faces[0]

        # Map coords back to original frame
        x0 = int(x / scale)
        y0 = int(y / scale)
        x1 = int((x + fw) / scale)
        y1 = int((y + fh) / scale)

        face_img = frame[y0:y1, x0:x1]

        # Embedding
        emb = np.array(
            DeepFace.represent(face_img, model_name=MODEL, enforce_detection=False)[0]["embedding"]
        )

        # Check face drift
        if cached_emb is None or cosine_distance(emb, cached_emb) > FACE_CHANGE_THRESHOLD:
            cached_emb = emb

            # -------------- constant threshold -------------------
            dynamic_th = THRESHOLD
            # -----------------------------------------------------

            # Distances
            dists = [cosine_distance(emb, e) for e in known_embeddings]
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            confidence = (1 - best_dist) * 100

            # MATCH
            if best_dist <= dynamic_th:
                prev_name = known_names[best_idx]
                prev_color = (0, 0, 255)   # RED = recognized person
                
                print(f"[IDENTIFIED] {prev_name}  |  Confidence: {confidence:.2f}%  |  Distance: {best_dist:.4f}")

                # send mail
                send_email(prev_name, confidence, best_dist)
            else:
                prev_name = "Unknown"
                prev_color = (0, 255, 0)   # GREEN = stranger

            prev_conf = confidence
            prev_dist = best_dist

    frame_count += 1

    # -------------------------------------------------------
    # DRAW RESULTS
    # -------------------------------------------------------
    if prev_dist is not None:
        label = f"{prev_name} | {prev_conf:.1f}% | Dist: {prev_dist:.3f}"
    else:
        label = f"{prev_name}"

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, prev_color, 2)

    if len(faces) > 0:
        (x, y, fw, fh) = faces[0]
        x0 = int(x / scale)
        y0 = int(y / scale)
        x1 = int((x + fw) / scale)
        y1 = int((y + fh) / scale)
        cv2.rectangle(frame, (x0, y0), (x1, y1), prev_color, 2)

    cv2.imshow("Fast Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
