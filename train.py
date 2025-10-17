from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='data.yaml',
    epochs=100,   
    name='yolo_speedclimbing'
)
 