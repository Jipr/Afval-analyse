import pandas as pd
import openpyxl
import numpy
import cv2
from ultralytics import YOLO
import supervision as sv


video_show = True

def connect_cam():
    # define a video capture object
    vid = cv2.VideoCapture(1)
    vid.set(3, 640)
    vid.set(4, 480)
    return vid

def release_cam(vid):
    vid.release()
    if video_show == True:
        # Destroy all the windows
        cv2.destroyAllWindows()


def commit_to_dataBase(object_class):
    a = 10
    b = 23
    c = 55
    f = pd.DataFrame([[a], [b], [c]],
                    index=['Class 1', 'Class 2', 'Class 3'], columns=['Amount'])
    f.to_excel('OSPAR_Test.xlsx', sheet_name='First_try')

def create_box_annotator():
    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
    )
    return box_annotator
    

def create_labels(model, detections):
    labels = [
        f"# {tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections 
    ]
    return labels


while(True):
    vid = connect_cam # Connect to camera
    ret, frame = vid.read() # Capture the video frame
    
    model = YOLO("Test.pt") #Trained model / path to local file
    
    box_annotator = create_box_annotator 

    for result in model.track(source=1, show=True, stream=True): # Detect frame by frame
        frame = result.orig_img 
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            
        labels = create_labels(model,detections)
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)



        if video_show == True: # Display the resulting frame
            cv2.imshow('Litter ditection', frame)     
        if cv2.waitKey(1) & 0xFF == ord('q'): # Quit script button
            release_cam(vid)
            break






