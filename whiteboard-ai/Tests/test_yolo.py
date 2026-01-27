from ultralytics import YOLO

model = YOLO('/home/irfana/Documents/repos/LLM-Trainings/runs/detect/runs/whiteboard_yolo/yolo_whiteboard_n/weights/best.pt')
results = model('/home/irfana/Documents/repos/LLM-Trainings/whiteboard-ai/datasets/pretest/sample_whiteboard_3.jpg')
results[0].show()
