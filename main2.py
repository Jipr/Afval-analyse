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
video_file_name = "VID_20230406_143256~3.mp4"
modelPath = "runs/detect/train12/weights/best.pt"
confidence = 0.6 #Object confidence threshold for detection
IoU = 0.7 #Intersection over union (IoU) threshold for NMS

#Global variables#------------------------------------------------
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True
workbook = xlrd.open_workbook('OSPAR_Test.xlsx')
worksheet = workbook.sheet_by_index(-1)
cell = worksheet.cell(1, 2)
value = cell.value
print("Value is:", value)
sheet_number = int(value + 1)

LINE_START = sv.Point(0, 675)
LINE_END = sv.Point(1080, 657)

today = date.today()


#Objects ID's
object1_id = 0 # Object ID of: Drankblikjes
object1_name = 'Drankblikjes'
object2_id = 1 # Object ID of: Drankflessen < 1/2 liter
object2_name = 'Drankflessen < 1/2 liter'
object3_id = 3 # Object ID of: Plastic folies of stukken daarvan 2.5 - 50 cm
object3_name = 'Plastic folies of stukken daarvan 2.5 - 50 cm'
object4_id = 4 # Object ID of: Snoep snack en chips verpakkingen
object4_name = 'Snoep snack en chips verpakkingen'
object5_id = 5 # Object ID of: Voedselverpakkingen
object5_name = 'Voedselverpakkingen'
object6_id = 2 # Object ID of: Ondefinieerbare stukjes piepschuim 2.5- 50 cm
object6_name = 'Ondefinieerbare stukjes piepschuim 2.5- 50 cm'

global object1_in
global object2_in
global object3_in
global object4_in
global object5_in
global object6_in

#Save video--------------------------------------------------------
video = cv2.VideoCapture(video_file_name)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
# Below VideoWriter object will create a frame of above defined The output is stored in BatchNumber'x' file.
video_result = cv2.VideoWriter(f'BatchNumber{sheet_number}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)



#Functions#-------------------------------------------------------
def commit_to_dataBase(object1_count,object2_count,object3_count,object4_count,object5_count,object6_count):
    f = pd.DataFrame([[object1_count], [object2_count], [object3_count], [object4_count], [object5_count], [object6_count]], index=[object1_name, object2_name, object3_name, object4_name, object5_name, object6_name], columns=['Amount'])
    with pd.ExcelWriter('OSPAR_Test.xlsx', engine="openpyxl", mode='a') as writer:  
        f.at[object1_name, 'Sheet number'] = sheet_number
        f.at[object1_name, 'Video name'] = video_file_name
        f.at[object1_name, 'Date'] = today
        f.to_excel(writer, sheet_name= f'Sheet{sheet_number}')
        

def create_labels(model, detections):
    labels = [
        f"# {tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    return labels


#Class#------------------------------------------------------------
class object_line_counter():
    def __init__(self,class_id_number,line_counter):
        self.class_id_number = class_id_number
        self.line_counter = line_counter

    def detections(self,result):
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == self.class_id_number]

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)  
            
        self.line_counter.trigger(detections=detections)
        count_in = self.line_counter.in_count
        count_out = self.line_counter.out_count

        return count_in,count_out


#Main function#----------------------------------------------------
def main():
    try:
        #Initialisation -----------------------------------------------
        line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
        line_counter1 = sv.LineZone(start=LINE_START, end=LINE_END)
        line_counter2 = sv.LineZone(start=LINE_START, end=LINE_END)
        line_counter3 = sv.LineZone(start=LINE_START, end=LINE_END)
        line_counter4 = sv.LineZone(start=LINE_START, end=LINE_END)
        line_counter5 = sv.LineZone(start=LINE_START, end=LINE_END)
        line_counter6 = sv.LineZone(start=LINE_START, end=LINE_END)

        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5)

        # Create class objects ----------------------------------------
        object1 = object_line_counter(object1_id, line_counter1)
        object2 = object_line_counter(object2_id, line_counter2)
        object3 = object_line_counter(object3_id, line_counter3)
        object4 = object_line_counter(object4_id, line_counter4)
        object5 = object_line_counter(object5_id, line_counter5)
        object6 = object_line_counter(object6_id, line_counter6)

        model = YOLO(modelPath) #Load YOLOv8 model
        model.names[0] = 'Drankblikjes'
        model.names[1] = 'Drankflessen'
        model.names[2] = 'Piepschuim'
        model.names[3] = 'Plastic folies'
        model.names[4] = 'Snoep verpakkingen'
        model.names[5] = 'Voedselverpakkingen'

        for result in model.track(source=video_file_name, show=True, stream=True, agnostic_nms=True, conf=confidence, iou=IoU):

            frame = result.orig_img

            #Detect al objects-----------------------------------------
            detections = sv.Detections.from_yolov8(result) #Get object detections from result
            if result.boxes.id is not None: #If nothing is detected in frame
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = create_labels(model,detections) #Create labels

            #Detect specific objects-----------------------------------
            object1_in,object1_out = object1.detections(result)
            object2_in,object2_out = object2.detections(result)
            object3_in,object3_out = object3.detections(result)
            object4_in,object4_out = object4.detections(result)
            object5_in,object5_out = object5.detections(result)
            object6_in,object6_out = object6.detections(result)

            #Show boxes and line counter--------------------------------
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            line_counter.trigger(detections=detections)
            line_annotator.annotate(frame=frame, line_counter=line_counter)

            video_result.write(frame) #Write the frame into the file BatchNumber'x' file.
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("yolov8", small)

            if (cv2.waitKey(30) == 27):
                cv2.destroyAllWindows()
                break
    finally: #If video is finished, or ESC is pressed, the counters are uploaded to the excel file with sheet name Sheet{sheet_number}
        commit_to_dataBase(object1_in, object2_in, object3_in, object4_in, object5_in, object6_in)
        video_result.release()
        print("The video, and counters are successfully saved")

if __name__ == "__main__":
    main()

