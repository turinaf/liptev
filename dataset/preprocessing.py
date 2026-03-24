import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


def extract_lip_frames(video_path, max_frames=75, model_path="face_landmarker.task"):
    """
    Extracts frames from a video file and crops the lip region dynamically
    using the provided MediaPipe Tasks API.
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    prev_crop = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
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
    
    # Pad or truncate to max_frames
    if len(frames) == 0:
        # Fallback if video completely fails or no face ever detected
        frames = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(max_frames)]
    elif len(frames) > max_frames:
        frames = frames[:max_frames]
    else:
        pad_len = max_frames - len(frames)
        last_frame = frames[-1]
        frames.extend([last_frame] * pad_len)
        
    return frames
