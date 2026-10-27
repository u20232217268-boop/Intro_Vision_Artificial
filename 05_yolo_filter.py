import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 nano
modelo = YOLO("yolov8n.pt")

# Inicializar la cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo recibir el frame.")
        break

    # Realizar la predicción en el frame actual (verbose=False oculta los logs en consola)
    resultados = modelo(frame, conf=0.5, verbose=False)

    # Dibuja automáticamente las cajas delimitadoras y etiquetas sobre el frame
    frame_anotado = resultados[0].plot()

    # Mostrar el frame procesado
    cv2.imshow("YOLO personalizado - presiona Q para salir", frame_anotado)

    # Romper el bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar los recursos
camara.release()
cv2.destroyAllWindows()