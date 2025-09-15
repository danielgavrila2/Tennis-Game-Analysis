from ultralytics import YOLO

model = YOLO("D:\PROJECTS\Tennis Game Analysis\models\yolo5_last.pt")

result = model.predict("input_videos/input_video.mp4",
                       save=True, conf=0.25, show=True)

print(result)

print("Boxes:")
for box in result[0].boxes:
    print(box)
