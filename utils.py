"""
utils.py
Shared helper functions for hand landmark extraction using MediaPipe.

MediaPipe Hands gives us 21 landmarks per hand, each with (x, y, z)
coordinates normalized to [0, 1] relative to the image frame.

For classification we don't want to use raw (x, y) coordinates directly,
because that would make the model sensitive to WHERE the hand is in the
frame and how big it is. Instead we normalize each landmark relative to
the wrist (landmark 0) and scale by the hand's bounding box size. This
makes the features translation- and scale-invariant, so the model learns
the SHAPE of the sign, not its position on screen.
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_hands_detector(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5):
    """Create and return a configured MediaPipe Hands detector."""
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmark_features(hand_landmarks):
    """
    Convert a MediaPipe hand_landmarks object into a flat, normalized
    feature vector of length 63 (21 landmarks x 3 coordinates).

    Steps:
    1. Pull out raw (x, y, z) for each of the 21 landmarks.
    2. Subtract the wrist position (landmark 0) from every point, so the
       feature set no longer depends on where the hand is in the frame.
    3. Scale by the max distance from the wrist, so the feature set
       doesn't depend on how close/far the hand is from the camera.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    wrist = coords[0].copy()
    coords -= wrist  # translation invariance

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 1e-6:
        coords /= max_dist  # scale invariance

    return coords.flatten()  # shape (63,)


def extract_combined_features(frame_bgr, hand_landmarks):
    """
    Build the hybrid feature vector used throughout this project:
    63 MediaPipe landmark features + 13 classical CV shape features
    (from preprocessing.py), extracted from a skin-segmented contour
    inside the hand's bounding box. Total length: 76.

    Returns the combined feature vector (numpy array), or the
    landmark-only vector padded with zeros if no valid classical
    contour was found (so the feature vector length stays constant).
    """
    from preprocessing import extract_classical_features, landmarks_to_bbox, NUM_CLASSICAL_FEATURES

    landmark_feats = extract_landmark_features(hand_landmarks)

    bbox = landmarks_to_bbox(hand_landmarks, frame_bgr.shape)
    classical_feats, _, _ = extract_classical_features(frame_bgr, roi_box=bbox)

    if classical_feats is None:
        classical_feats = np.zeros(NUM_CLASSICAL_FEATURES, dtype=np.float32)

    return np.concatenate([landmark_feats, classical_feats])


def draw_landmarks(image, hand_landmarks):
    """Draw hand skeleton overlay on the given BGR image (in-place)."""
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
