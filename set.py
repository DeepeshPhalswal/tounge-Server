import cv2
import numpy as np
import mediapipe as mp
import sys
import os

mp_face_mesh = mp.solutions.face_mesh

def mouth_bbox_from_landmarks(landmarks, img_w, img_h, pad=0.2):
    # Use lips landmarks indices from MediaPipe face mesh
    # We'll take a range around the mouth area: use lower/upper lip landmarks
    # Indices based on MediaPipe mesh (commonly used ranges).
    mouth_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 308, 324, 318, 402, 317, 14
    ]
    xs = [landmarks[i].x * img_w for i in mouth_indices]
    ys = [landmarks[i].y * img_h for i in mouth_indices]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    w = x_max - x_min
    h = y_max - y_min
    # pad relative to size
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x_min - px)
    y1 = max(0, y_min - py)
    x2 = min(img_w, x_max + px)
    y2 = min(img_h, y_max + py)
    return x1, y1, x2, y2

def segment_tongue_in_mouth(mouth_roi_bgr):
    # Convert to HSV for color-based segmentation (tongue ~ pink/red)
    hsv = cv2.cvtColor(mouth_roi_bgr, cv2.COLOR_BGR2HSV)

    # Two ranges: low-red and high-red in HSV (wrap-around)
    # These ranges are adjustable depending on light and skin tone.
    lower1 = np.array([0, 40, 80])     # e.g., light red/pink lower bound
    upper1 = np.array([12, 255, 255])  # small hue for red/pink

    lower2 = np.array([160, 30, 80])   # upper-red wrap-around
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Additional heuristic: the tongue is often more saturated and brighter than mouth interior
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: keep the largest contour (assume tongue is largest red region inside mouth)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask  # empty
    largest = max(contours, key=cv2.contourArea)
    # create new mask with only largest contour if it's reasonably sized
    mask2 = np.zeros_like(mask)
    area = cv2.contourArea(largest)
    if area > 50:  # threshold in pixels, adjust if needed
        cv2.drawContours(mask2, [largest], -1, 255, thickness=-1)
        return mask2
    return mask

def keep_tongue_and_blacken(img_bgr, detect_confidence=0.5):
    img_out = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4,
                               refine_landmarks=True, min_detection_confidence=detect_confidence) as fm:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = fm.process(img_rgb)

        full_tongue_mask = np.zeros((h, w), dtype=np.uint8)

        if not result.multi_face_landmarks:
            # no face found: fallback attempt - try whole image color segmentation (less reliable)
            print("Warning: no face detected. Attempting global color segmentation.")
            global_mask = segment_tongue_in_mouth(img_bgr)
            full_tongue_mask = global_mask
        else:
            for face_landmarks in result.multi_face_landmarks:
                x1, y1, x2, y2 = mouth_bbox_from_landmarks(face_landmarks.landmark, w, h, pad=0.4)
                # ensure rect valid
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                mouth_roi = img_bgr[y1:y2, x1:x2]
                mouth_mask = segment_tongue_in_mouth(mouth_roi)

                # place mouth_mask into full image mask
                # mouth_mask is single-channel; ensure same size
                if mouth_mask.shape[:2] != (y2-y1, x2-x1):
                    mouth_mask = cv2.resize(mouth_mask, (x2-x1, y2-y1))
                full_tongue_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_tongue_mask[y1:y2, x1:x2], mouth_mask)

        # refine full_tongue_mask with morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        full_tongue_mask = cv2.morphologyEx(full_tongue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        full_tongue_mask = cv2.medianBlur(full_tongue_mask, 7)

        # create three-channel mask
        mask3 = cv2.merge([full_tongue_mask, full_tongue_mask, full_tongue_mask])
        # Create output: black everywhere
        black_bg = np.zeros_like(img_bgr)
        # Copy only tongue pixels
        tongue_pixels = cv2.bitwise_and(img_bgr, mask3)
        # Combine: tongue over black
        out = tongue_pixels  # everything else already zero
        # But to preserve transparent smooth edge, we can blend a bit (optional)
        # return out and mask for debugging
        return out, full_tongue_mask

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python black_everything_except_tongue.py input.jpg output.jpg")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.exists(in_path):
        print("Input file not found:", in_path)
        sys.exit(1)

    img = cv2.imread(in_path)
    out_img, mask = keep_tongue_and_blacken(img)

    # Optional: overlay a faint contour for debugging (uncomment)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(out_img, contours, -1, (0,255,0), 1)

    cv2.imwrite(out_path, out_img)
    print("Saved:", out_path)
