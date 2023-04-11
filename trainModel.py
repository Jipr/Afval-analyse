from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.yaml")  # build a new model from scratch


# Use the model
model.train(data="trainInfo1.yaml", epochs=1, patience=50, batch=-1, )  # train the model
