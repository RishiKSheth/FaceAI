#This AI was made by Rishi Sheth and Atharva Kadam. It is able to identify and save faces into a database. It can be implemented in cameras such as doorbells, Security Cams and Airport Security Cams.

import time
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import json
import tkinter as tk

# ===============[ SETUP ]=================
cap = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh()

conn = sqlite3.connect("Data/cameraDataBase.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS CameraData (
        user_id TEXT,
        landmarks TEXT
    )
''')

# Load button and overlay images
button_image = cv2.imread("images/cameraButtonImage.png", cv2.IMREAD_UNCHANGED)
face_image = cv2.imread("images/FaceOutline.png", cv2.IMREAD_UNCHANGED)

button_resized = cv2.resize(button_image, (140, 140))
button_pos = (10, 10)
button_clicked = False

face_resized = cv2.resize(face_image, (300, 400))  # H, W

# ===============[ FUNCTIONS ]=================

stable_indices = [33, 263, 1, 4, 78, 308, 152, 10, 234, 454]
key_pairs = [(0, 1), (2, 3), (3, 6), (4, 5), (7, 8)]

def get_stable_landmarks(landmarks):
    return [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in stable_indices]

def compute_signature(landmarks):
    signature = []
    for i, j in key_pairs:
        xi, yi, zi = landmarks[i]
        xj, yj, zj = landmarks[j]
        dist = np.linalg.norm([xi - xj, yi - yj, zi - zj])
        signature.append(dist)
    normalizer = np.mean(signature) if np.mean(signature) != 0 else 1
    return [d / normalizer for d in signature]

def compare_signatures(current, saved, threshold=0.05):
    current_np = np.array(current)
    saved_np = np.array(saved)
    if current_np.shape != saved_np.shape:
        return False
    dist = np.linalg.norm(current_np - saved_np)
    print(f"Signature distance: {dist:.4f}")
    return dist < threshold

def find_matching_user(current_signature):
    cursor.execute("SELECT user_id, landmarks FROM CameraData")
    for user_id, saved_json in cursor.fetchall():
        saved_signature = json.loads(saved_json)
        if compare_signatures(current_signature, saved_signature):
            return user_id
    return None

def show_popup(title, message):
    def on_submit():
        nonlocal user_input
        user_input = entry.get().strip()
        popup.destroy()

    def validate_entry(*args):
        text = entry_var.get().strip()
        submit_btn.config(state='normal' if text else 'disabled')

    user_input = None
    popup = tk.Tk()
    popup.title(title)
    popup.geometry("300x120")
    tk.Label(popup, text=message).pack(padx=20, pady=10)

    entry_var = tk.StringVar()
    entry_var.trace_add("write", validate_entry)
    entry = tk.Entry(popup, width=30, textvariable=entry_var)
    entry.pack(pady=5)
    entry.focus()

    submit_btn = tk.Button(popup, text="Submit", state='disabled', command=on_submit)
    submit_btn.pack(pady=10)
    popup.mainloop()
    return user_input

def overlay_image(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Clip if overlay goes out of bounds
    if x < 0:
        overlay = overlay[:, -x:]
        ow = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        oh = overlay.shape[0]
        y = 0
    if x + ow > bw:
        overlay = overlay[:, :bw - x]
        ow = overlay.shape[1]
    if y + oh > bh:
        overlay = overlay[:bh - y, :]
        oh = overlay.shape[0]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+oh, x:x+ow, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * background[y:y+oh, x:x+ow, c]
            ).astype(np.uint8)
    else:
        background[y:y+oh, x:x+ow] = overlay
    return background

def mouse_click(event, x, y, flags, param):
    global button_clicked
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    mirrored_x = int(frame_width - x)
    bx, by = button_pos
    bw, bh = 140, 140
    if event == cv2.EVENT_LBUTTONDOWN:
        if bx <= mirrored_x <= bx + bw and by <= y <= by + bh:
            button_clicked = True

cv2.namedWindow("MediaPipe")
cv2.setMouseCallback("MediaPipe", mouse_click)

# ===============[ MAIN LOOP ]=================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)
    results_mesh = face_mesh.process(frame_rgb)

    if results_face.detections and results_mesh.multi_face_landmarks:
        for detection, face_landmarks in zip(results_face.detections, results_mesh.multi_face_landmarks):
            landmark_data = get_stable_landmarks(face_landmarks.landmark)
            signature = compute_signature(landmark_data)
            matched_user = find_matching_user(signature)

            mp_drawing.draw_detection(frame, detection)

            for i in stable_indices:
                x = int(face_landmarks.landmark[i].x * frame.shape[1])
                y = int(face_landmarks.landmark[i].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if matched_user:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                box_width = int(bbox.width * w)
                box_height = int(bbox.height * h)

                cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 0, 0), 2)
                cv2.putText(frame, matched_user, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Center overlay image (face outline)
    frame_h, frame_w = frame.shape[:2]
    face_h, face_w = face_resized.shape[:2]
    face_x = (frame_w - face_w) // 2
    face_y = (frame_h - face_h) // 2
    frame = overlay_image(frame, face_resized, face_x, face_y)

    # Camera button
    frame = overlay_image(frame, button_resized, *button_pos)

    # On button click → Save face
    if button_clicked:
        button_clicked = False
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                landmark_data = get_stable_landmarks(face_landmarks.landmark)
                signature = compute_signature(landmark_data)
                matched_user = find_matching_user(signature)

                if matched_user:
                    print(f"Already registered as: {matched_user}")
                else:
                    user_name = show_popup("Save Face", "Enter your name:")
                    if user_name:
                        cursor.execute('''
                            INSERT INTO CameraData (user_id, landmarks)
                            VALUES (?, ?)
                        ''', (user_name, json.dumps(signature)))
                        conn.commit()
                        print(f"Saved signature for '{user_name}'")
                    else:
                        print("No name entered. Skipped saving.")
                break

    cv2.imshow('MediaPipe', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============[ CLEANUP ]=================
cap.release()
cv2.destroyAllWindows()
