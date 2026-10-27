from ultralytics import YOLO

modelo = YOLO("yolov8n.pt")

# Detecta exclusivamente personas con confianza mayor al 50%
modelo.predict(source=0, show=True, conf=0.5)