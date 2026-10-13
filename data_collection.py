"""
data_collection.py
-------------------
STEP 1 of the pipeline: build your own dataset.

Usage:
    python data_collection.py

How it works:
- Opens your webcam.
- You choose which letter/sign you're currently recording by pressing
  its key on the keyboard (a-z), or a custom label.
- While a label is "armed", every frame where a hand is detected gets
  its hybrid feature vector (63 MediaPipe landmark features + 13
  classical CV shape features from preprocessing.py — skin-segmented
  contour solidity, extent, aspect ratio, convexity-defect count, and
  Hu moments) saved as one row in data/landmarks.csv, tagged with that
  label.
- Press SPACE to arm/disarm recording for the currently selected label.
- Press ENTER to type a custom label (e.g. "hello", "thanks") in the console.
- Press ESC to quit. (Letters q and n are normal labels like any other.)

Tips for good data:
- Record ~150-300 samples per sign.
- Vary hand angle/position/distance slightly between samples so the
  model generalizes instead of memorizing one exact pose.
- Keep lighting reasonably consistent and your hand fully in frame.
"""

import os
import csv
import cv2
from utils import get_hands_detector, extract_combined_features, draw_landmarks
from preprocessing import NUM_CLASSICAL_FEATURES

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
NUM_LANDMARK_FEATURES = 63  # 21 landmarks x 3 coords
NUM_FEATURES = NUM_LANDMARK_FEATURES + NUM_CLASSICAL_FEATURES  # hybrid vector


def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]
            writer.writerow(header)


def append_sample(label, features):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + list(features))


def main():
    ensure_csv()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    detector = get_hands_detector(static_image_mode=False, max_num_hands=1)

    current_label = None
    recording = False
    sample_count = 0

    print("Controls:")
    print("  a-z         : select that letter as the current label")
    print("  ENTER       : type a custom label in the terminal")
    print("  SPACE       : start/stop recording samples for current label")
    print("  ESC         : quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            if recording and current_label is not None:
                features = extract_combined_features(frame, hand_landmarks)
                append_sample(current_label, features)
                sample_count += 1

        status = f"Label: {current_label or '-'} | Recording: {recording} | Samples this session: {sample_count}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.imshow("Data Collection - press ESC to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (13, 10):  # ENTER
            label = input("Enter custom label: ").strip()
            if label:
                current_label = label
                recording = False
        elif key == ord(" "):
            recording = not recording
        elif ord("a") <= key <= ord("z"):
            current_label = chr(key).upper()
            recording = False

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total samples saved this session: {sample_count}")
    print(f"Data file: {CSV_PATH}")


if __name__ == "__main__":
    main()
