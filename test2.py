from roboflow import Roboflow
import cv2
import pandas as pd
import xlrd
from ultralytics import YOLO
import supervision as sv

rf = Roboflow(api_key="G1DqaWlFlqKj1fuE2crO")
project = rf.workspace().project("drijfvuil-detectie")
model = project.version(1).model

# infer on a local image
print(model.predict("Drijfvuil detectie.v1i.yolov8/train/images/VID_20230406_131902-2_000028_jpg.rf.158c4af36d910dbae3df0d7b912fd22e", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())