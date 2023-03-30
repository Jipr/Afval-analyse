'''
Author: Jip Rasenberg
Date: 30-3-2023
Graduation project: Afval analyse automatisatie
Company: Noria

'''

#Libraries#---------------------------------------------
import cv2
import pandas as pd
import xlrd
from ultralytics import YOLO
import supervision as sv

#Global variables#--------------------------------------
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True
workbook = xlrd.open_workbook('OSPAR_Test.xlsx')
worksheet = workbook.sheet_by_index(-1)
cell = worksheet.cell(1, 2)
value = cell.value
print("Value is:", value)
sheet_number = int(value + 1)

LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)

#Objects ID's 
object1_id = 41 # Object ID of: cup
object1_name = 'Cup'
object2_id = 39 # Object ID of: bottle
object2_name = 'Bottle'
object3_id = 3 # Object ID of:
object3_name = 'Object 3'
object4_id = 4 # Object ID of:
object4_name = 'Object 4'
object5_id = 5 # Object ID of:
object5_name = 'Object 5'


#Functions#---------------------------------------------
def commit_to_dataBase(object1_count,object2_count,object3_count,object4_count,object5_count):
    f = pd.DataFrame([[object1_count], [object2_count], [object3_count], [object4_count], [object5_count]], index=[object1_name, object2_name, object3_name, object4_name, object5_name], columns=['Amount'])
    
  
    with pd.ExcelWriter('OSPAR_Test.xlsx', engine="openpyxl", mode='a') as writer:  
        f.at[object1_name, 'Sheet number'] = sheet_number
        f.to_excel(writer, sheet_name= f'Sheet{sheet_number}')
        

def create_labels(model, detections):
    labels = [
        f"# {class_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections 
    ]
    return labels
    

#Class#---------------------------------------------
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


#Main function#-------------------------------------------
def main():

    #Initialisation --------------------------------------
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter1 = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter2 = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter3 = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter4 = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter5 = sv.LineZone(start=LINE_START, end=LINE_END)
    
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5)

    # Create class objects ------------------------------
    object1 = object_line_counter(object1_id, line_counter1)
    object2 = object_line_counter(object2_id, line_counter2)
    object3 = object_line_counter(object3_id, line_counter3)
    object4 = object_line_counter(object4_id, line_counter4)
    object5 = object_line_counter(object5_id, line_counter5)

    model = YOLO("yolov8l.pt")

    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
        
        frame = result.orig_img

        #Detect al objects
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id != 0]
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        labels = create_labels(model,detections)

        object1_in,object1_out = object1.detections(result)
        object2_in,object2_out = object2.detections(result)
        object3_in,object3_out = object3.detections(result)
        object4_in,object4_out = object4.detections(result)
        object5_in,object5_out = object5.detections(result)

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        print('object1_in',object1_in, 'object1_out',object1_out)
        print('object2_in',object2_in, 'object2_out',object2_out)
        print('object3_in',object3_in, 'object3_out',object3_out)
        print('object4_in',object4_in, 'object4_out',object4_out)
        print('object5_in',object5_in, 'object5_out',object5_out)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            commit_to_dataBase(object1_in,object2_in,object3_in,object4_in,object5_in)
            break


if __name__ == "__main__":
    main()