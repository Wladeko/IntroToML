import os
import cv2
import mediapipe as mp
import pyrootutils
import streamlit as st
from PIL import Image

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

from src.utils.functions import inference_picture
from src.utils.predict import Predict


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

file = input()

output_folder = "combined_images"
os.makedirs(output_folder, exist_ok=True)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    model = Predict()

    import numpy as np

    image = Image.open(file)

    image = np.array(image)

    image = inference_picture(
        image,
        mp_drawing,
        mp_face_detection,
        face_detection,
        model,
        SIZE=0.1,
    )

    output_path = os.path.join(output_folder, f"image_response.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, image)

