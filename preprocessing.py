
"""
preprocessing.py
-----------------
Classical computer vision preprocessing for the Sign Language to Speech
project.

Pipeline:
1. HSV skin-color segmentation
2. Morphological noise removal
3. Largest-contour extraction
4. Shape descriptors:
   - Solidity
   - Extent
   - Aspect ratio
   - Convexity defects
   - Hu moments
   - Equivalent diameter ratio
   - Circularity

Note: Skin-color segmentation is sensitive to lighting and background
colors. MediaPipe landmarks are recommended for reliable hand recognition.
"""

import cv2
import numpy as np


# Default HSV range for skin-color segmentation.
# OpenCV HSV ranges: H = 0-179, S = 0-255, V = 0-255.
DEFAULT_LOWER_HSV = np.array([0, 30, 60], dtype=np.uint8)
DEFAULT_UPPER_HSV = np.array([25, 150, 255], dtype=np.uint8)

NUM_CLASSICAL_FEATURES = 13


def segment_skin(frame_bgr, lower_hsv=None, upper_hsv=None):
    """
    Convert a BGR image into a cleaned binary skin-color mask.
    White pixels represent detected skin-color regions.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Input frame is empty.")

    if lower_hsv is None:
        lower_hsv = DEFAULT_LOWER_HSV
    if upper_hsv is None:
        upper_hsv = DEFAULT_UPPER_HSV

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8)
    )

    # Morphological opening removes small noise.
    # Closing fills small gaps in detected regions.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, mask = cv2.threshold(
        mask, 127, 255, cv2.THRESH_BINARY
    )

    return mask


def largest_contour(mask, min_area=1500):
    """
    Find and return the largest external contour.
    Return None if no contour is large enough.
    """
    if mask is None or mask.size == 0:
        return None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < min_area:
        return None

    return contour


def find_convexity_defects(contour, depth_ratio=0.08):
    """
    Detect deep convexity defects, which can roughly represent gaps
    between fingers.

    Returns:
        List of (start_point, end_point, far_point, depth_pixels).
    """
    if contour is None or len(contour) < 4:
        return []

    hull_indices = cv2.convexHull(
        contour, returnPoints=False
    )

    if hull_indices is None or len(hull_indices) < 3:
        return []

    try:
        defects = cv2.convexityDefects(
            contour, hull_indices
        )
    except cv2.error:
        return []

    if defects is None:
        return []

    _, _, width, height = cv2.boundingRect(contour)
    minimum_depth = depth_ratio * max(width, height)

    found = []

    for item in defects[:, 0, :]:
        start_idx, end_idx, far_idx, depth_fixed = map(
            int, item
        )

        # OpenCV stores depth in fixed-point format (x256).
        depth_pixels = depth_fixed / 256.0

        if depth_pixels > minimum_depth:
            start_point = tuple(
                map(int, contour[start_idx][0])
            )
            end_point = tuple(
                map(int, contour[end_idx][0])
            )
            far_point = tuple(
                map(int, contour[far_idx][0])
            )

            found.append(
                (
                    start_point,
                    end_point,
                    far_point,
                    float(depth_pixels)
                )
            )

    return found


def count_convexity_defects(contour, depth_ratio=0.08):
    """Return the number of prominent convexity defects."""
    return len(
        find_convexity_defects(contour, depth_ratio)
    )


def shape_descriptors(contour):
    """
    Calculate 13 classical shape features:

    [solidity, extent, aspect_ratio, defect_count,
     Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7,
     equivalent_diameter_ratio, circularity]
    """
    if contour is None or len(contour) < 3:
        raise ValueError("A valid contour is required.")

    area = cv2.contourArea(contour)

    x, y, width, height = cv2.boundingRect(contour)

    rectangle_area = width * height

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    perimeter = cv2.arcLength(contour, True)

    solidity = (
        area / hull_area if hull_area > 0 else 0.0
    )

    extent = (
        area / rectangle_area
        if rectangle_area > 0 else 0.0
    )

    aspect_ratio = (
        width / height if height > 0 else 0.0
    )

    defect_count = count_convexity_defects(contour)

    # Hu moments describe the overall shape.
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    # Log scaling makes very small moment values easier to use.
    hu_log = (
        -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    )

    equivalent_diameter = (
        np.sqrt(4.0 * area / np.pi)
        if area > 0 else 0.0
    )

    equivalent_diameter_ratio = (
        equivalent_diameter / max(width, height, 1)
    )

    circularity = (
        4.0 * np.pi * area / (perimeter ** 2)
        if perimeter > 0 else 0.0
    )

    features = np.concatenate([
        np.array(
            [
                solidity,
                extent,
                aspect_ratio,
                float(defect_count)
            ],
            dtype=np.float64
        ),
        hu_log,
        np.array(
            [
                equivalent_diameter_ratio,
                circularity
            ],
            dtype=np.float64
        )
    ]).astype(np.float32)

    assert len(features) == NUM_CLASSICAL_FEATURES

    return features


def extract_classical_features(frame_bgr, roi_box=None):
    """
    Run segmentation, contour extraction, and shape analysis.

    roi_box:
        Optional (x1, y1, x2, y2) pixel coordinates around the hand.

    Returns:
        features, mask, contour

    Features and contour are None when no sufficiently large contour
    is detected. When using roi_box, contour coordinates are relative
    to the cropped region.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None, None

    if roi_box is not None:
        x1, y1, x2, y2 = map(int, roi_box)

        frame_height, frame_width = frame_bgr.shape[:2]

        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))

        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None, None, None
    else:
        crop = frame_bgr

    mask = segment_skin(crop)
    contour = largest_contour(mask)

    if contour is None:
        return None, mask, None

    features = shape_descriptors(contour)

    return features, mask, contour


def landmarks_to_bbox(hand_landmarks, frame_shape, margin=0.25):
    """
    Convert MediaPipe normalized hand landmarks into a pixel bounding
    box (x1, y1, x2, y2), expanded by a margin around the hand.
    """
    frame_height, frame_width = frame_shape[:2]

    landmarks = hand_landmarks.landmark

    xs = [float(point.x) for point in landmarks]
    ys = [float(point.y) for point in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_width = x_max - x_min
    box_height = y_max - y_min

    x_min -= box_width * margin
    x_max += box_width * margin
    y_min -= box_height * margin
    y_max += box_height * margin

    return (
        max(0, int(x_min * frame_width)),
        max(0, int(y_min * frame_height)),
        min(frame_width, int(x_max * frame_width)),
        min(frame_height, int(y_max * frame_height))
    )


def calibrate_skin_range(frame_bgr, box):
    """
    Calibrate HSV skin thresholds from the central portion of a palm
    patch. Percentiles reduce the influence of background pixels.

    Args:
        frame_bgr: Camera frame in BGR format.
        box: (x1, y1, x2, y2) calibration-box coordinates.

    Returns:
        lower_hsv, upper_hsv as uint8 arrays.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Cannot calibrate from an empty frame.")

    frame_height, frame_width = frame_bgr.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    # Clamp coordinates to the camera frame.
    x1 = max(0, min(x1, frame_width - 1))
    x2 = max(x1 + 1, min(x2, frame_width))

    y1 = max(0, min(y1, frame_height - 1))
    y2 = max(y1 + 1, min(y2, frame_height))

    patch = frame_bgr[y1:y2, x1:x2]

    if patch.size == 0:
        raise ValueError("Calibration box is empty.")

    # Sample the central 60% of the patch to reduce edge contamination.
    patch_height, patch_width = patch.shape[:2]

    margin_x = int(patch_width * 0.20)
    margin_y = int(patch_height * 0.20)

    if (
        patch_width - 2 * margin_x >= 1
        and patch_height - 2 * margin_y >= 1
    ):
        patch = patch[
            margin_y:patch_height - margin_y,
            margin_x:patch_width - margin_x
        ]

    hsv = cv2.cvtColor(
        patch, cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # Exclude very dark and nearly gray pixels.
    usable = hsv[
        (hsv[:, 2] > 40) &
        (hsv[:, 1] > 15)
    ]

    if len(usable) < 100:
        raise ValueError(
            "Not enough usable pixels for calibration. "
            "Fill the box with your palm, use even lighting, "
            "and press SPACE again."
        )

    # Percentile thresholds are more robust to outliers than mean +/- 3 SD.
    lower = np.percentile(
        usable, 5, axis=0
    ).astype(np.float32)

    upper = np.percentile(
        usable, 95, axis=0
    ).astype(np.float32)

    # Allow a little variation in color and brightness.
    lower -= np.array([3, 20, 25], dtype=np.float32)
    upper += np.array([3, 20, 25], dtype=np.float32)

    # Clamp thresholds to valid OpenCV HSV limits.
    lower = np.maximum(
        lower, np.array([0, 20, 40], dtype=np.float32)
    )

    upper = np.minimum(
        upper, np.array([179, 255, 255], dtype=np.float32)
    )

    return (
        lower.astype(np.uint8),
        upper.astype(np.uint8)
    )
