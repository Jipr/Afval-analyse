'''
Author: Jip Rasenberg
Date: 27-3-2023
Graduation project: Afval analyse automatisatie
Company: Noria

'''

#Libraries#---------------------------------------------
import pandas as pd
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy

#Global variables#--------------------------------------
START = sv.Point(320,0) # Start point counting line
END = sv.Point(320, 480) # End poinst counting line


#Functions on or off (True of False)#-------------------
video_show = True
detect = True


#Functions#---------------------------------------------
def release_cam():
    if video_show == True:
        # Destroy all the windows
        cv2.destroyAllWindows()


def commit_to_dataBase(object_class):
    a = 10
    b = 23
    c = 55
    f = pd.DataFrame([[a], [b], [c]], index=['Class 1', 'Class 2', 'Class 3'], columns=['Amount'])
    f.to_excel('OSPAR_Test.xlsx', sheet_name='First_try')

def create_labels(model, detections):
    labels = [
        f"# {tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections 
    ]
    return labels


#Main code#-------------------------------------------

while(detect):
    line_counter  = sv.LineZone(start=START, end=END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5)
    model = YOLO("yolov8l.pt") #Trained model / path to local file

    for result in model.track(source=0, show=video_show, stream=True, agnostic_nms=True ): # Detect frame by frame
        frame = result.orig_img 
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        #detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]
        labels = create_labels(model,detections)
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        line_counter.trigger(detections=detections) 
        frame = line_annotator.annotate(frame=frame, line_counter=line_counter)

        #if video_show == True: # Display the resulting frame
        #    cv2.imshow('Litter ditection', frame)  

        if cv2.waitKey(30) == 27: # Quit script with esc
            release_cam()
            detect = False
            break






