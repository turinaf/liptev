import cv2
import numpy as np
import mediapipe as mp

LIP_LANDMARKS = [
    61,146,91,181,84,17,314,405,321,375,291,
    185,40,39,37,0,267,269,270,409,
    78,95,88,178,87,14,317,402,318,324,308
]

def crop_lip_region(rgb_image, detection_result, padding=12, size=(96,96)):
    if not detection_result.face_landmarks:
        return None

    face_landmarks = detection_result.face_landmarks[0]
    h, w, _ = rgb_image.shape

    xs = []
    ys = []

    for idx in LIP_LANDMARKS:
        lm = face_landmarks[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x_min = max(min(xs) - padding, 0)
    x_max = min(max(xs) + padding, w)

    y_min = max(min(ys) - padding, 0)
    y_max = min(max(ys) + padding, h)

    lip_crop = rgb_image[y_min:y_max, x_min:x_max]

    if lip_crop.size == 0:
        return None

    # Handle resizing correctly to standard 96x96 shape
    # Using cv2.resize as it is canonical for array resizing over matplotlib 
    # but maintaining your bounding box formulation exactly.
    lip_crop = cv2.resize(lip_crop, size)

    return lip_crop


def extract_lip_frames(video_path, detector, correct_orientation=False):
    """
    Extracts frames from a video file and crops the lip region dynamically
    using the provided MediaPipe Tasks API detector.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    prev_crop = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if correct_orientation:
            # Need to process orientation for smartphones
            if frame.shape[1] < frame.shape[0]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect(mp_image)
        
        crop = crop_lip_region(frame_rgb, detection_result, padding=12, size=(96,96))
        
        if crop is not None:
            frames.append(crop)
            prev_crop = crop
        else:
            # Fallback to previous crop if tracked face drops for a frame
            if prev_crop is not None:
                frames.append(prev_crop)

    cap.release()
    
    return frames
