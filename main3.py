'''
Author: Jip Rasenberg
Date: 20-4-2023
Graduation project: Afval analyse automatisatie
Company: Noria

'''
#Libraries#-------------------------------------------------------
import cv2
from ultralytics import YOLO
import supervision as sv

#Variables to change----------------------------------------------
img1 = cv2.imread('Images\VID_20230406_141107~2_000062.jpg')
img2 = cv2.imread('Images\VID_20230406_131902~2_000027.jpg')
inputs = [img1, img2]  # list of numpy arrays
modelPath = "runs/detect/train12/weights/best.pt"


confidence = 0.6 #Object confidence threshold for detection
IoU = 0.7 #Intersection over union (IoU) threshold for NMS


def main():

    model = YOLO(modelPath) #Load YOLOv8 model
    model.names[0] = 'Drankblikjes'
    model.names[1] = 'Drankflessen'
    model.names[2] = 'Piepschuim'
    model.names[3] = 'Plastic folies'
    model.names[4] = 'Snoep verpakkingen'
    model.names[5] = 'Voedselverpakkingen'


    model.predict(source='Images\VID_20230406_131902~2_000027.jpg', show = True, save=True, conf=confidence, iou=IoU)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

