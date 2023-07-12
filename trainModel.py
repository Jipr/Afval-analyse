from ultralytics import YOLO
def main():
    # Load a model
    model = YOLO("yolov8l.yaml")  # build a new model from scratch


    # Use the model
    model.train(data="trainInfo1.yaml", epochs=300, patience=50)  # train the model


if __name__ == "__main__":
    main()