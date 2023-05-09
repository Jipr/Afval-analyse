'''
Author: Jip Rasenberg
Date: 20-4-2023
Graduation project: Afval analyse automatisatie
Company: Noria

'''
#Libraries#-------------------------------------------------------
import cv2
import pandas as pd
import xlrd
from ultralytics import YOLO
import supervision as sv
from datetime import date

#Variables to change----------------------------------------------
img1 = cv2.imread('/path_to_image/messi5.jpg',0)
img2 = cv2.imread('/path_to_image/messi5.jpg',0)
inputs = [img1, img2]  # list of numpy arrays
modelPath = "runs/detect/train12/weights/best.pt"

confidence = 0.6 #Object confidence threshold for detection
IoU = 0.7 #Intersection over union (IoU) threshold for NMS

counter = 1 

def create_labels(model, detections):
    labels = [
        f"# {tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    return labels



def main(counter):
        
    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5)

    model = YOLO(modelPath) #Load YOLOv8 model
    model.names[0] = 'Drankblikjes'
    model.names[1] = 'Drankflessen'
    model.names[2] = 'Piepschuim'
    model.names[3] = 'Plastic folies'
    model.names[4] = 'Snoep verpakkingen'
    model.names[5] = 'Voedselverpakkingen'

    results = model(inputs)
    
    for result in results:  
        #Detect al objects-----------------------------------------
        detections = sv.Detections.from_yolov8(result) #Get object detections from result
        if result.boxes.id is not None: #If nothing is detected in frame
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        labels = create_labels(model,detections) #Create labels

        #Show boxes and line counter--------------------------------
        frame = box_annotator.annotate(scene=result, detections=detections, labels=labels)
        # Filename
        filename = f'Sheet{counter}'
        cv2.imwrite(filename, frame)
        counter += 1

if __name__ == "__main__":
    main(counter)

