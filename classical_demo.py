"""
classical_demo.py
------------------
Pure classical image-processing demo — NO MediaPipe involved. Shows
every stage of the pipeline side by side so you can screenshot it for
your report and explain the image-processing fundamentals your course
is likely asking for:

    Raw frame -> HSV skin mask -> morphological cleanup ->
    largest contour -> convex hull + convexity defects

Usage:
    python classical_demo.py

Controls:
    SPACE : calibrate skin-color range from the patch inside the
            on-screen box (hold your palm flat over the box)
    r     : reset to default skin-color range
    q     : quit
"""

import cv2
import numpy as np
from preprocessing import (
    segment_skin, largest_contour, shape_descriptors, find_convexity_defects,
    calibrate_skin_range, DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV,
)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    lower_hsv, upper_hsv = DEFAULT_LOWER_HSV.copy(), DEFAULT_UPPER_HSV.copy()

    # Calibration box: user holds their palm flat over this region
    calib_box = None

    print("Controls: SPACE=calibrate skin color | r=reset | q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if calib_box is None:
            bw, bh = 80, 80
            cx, cy = w // 2, h // 2
            calib_box = (cx - bw, cy - bh, cx + bw, cy + bh)

        # --- Stage 1+2: skin segmentation + morphological cleanup ---
        mask = segment_skin(frame, lower_hsv, upper_hsv)

        # --- Stage 3: largest contour ---
        contour = largest_contour(mask)

        overlay = frame.copy()
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if contour is not None:
            # --- Stage 4: hull + convexity defects + shape descriptors ---
            hull = cv2.convexHull(contour)

            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(overlay, [hull], -1, (255, 0, 0), 2)

            for _, _, far, _ in find_convexity_defects(contour):
                cv2.circle(overlay, far, 5, (0, 0, 255), -1)

            features = shape_descriptors(contour)
            solidity, extent, aspect_ratio, defect_count = features[:4]
            info = (f"solidity={solidity:.2f} extent={extent:.2f} "
                    f"aspect={aspect_ratio:.2f} defects={int(defect_count)}")
            cv2.putText(overlay, info, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        # draw calibration box for reference
        x1, y1, x2, y2 = calib_box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(overlay, "Hold palm here + SPACE to calibrate", (x1 - 40, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- side-by-side display: original+overlay | mask ---
        combined = np.hstack([overlay, mask_bgr])
        cv2.imshow("Classical CV Pipeline: overlay | skin mask", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            lower_hsv, upper_hsv = calibrate_skin_range(frame, calib_box)
            print(f"Calibrated HSV range: lower={lower_hsv}, upper={upper_hsv}")
        elif key == ord("r"):
            lower_hsv, upper_hsv = DEFAULT_LOWER_HSV.copy(), DEFAULT_UPPER_HSV.copy()
            print("Reset to default HSV range.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
