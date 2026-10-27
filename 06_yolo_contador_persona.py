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

    # Realizar inferencia
    resultados = modelo(frame, conf=0.5, verbose=False, classes=[0])

    conteo_personas = 0

    # Contar únicamente las detecciones cuyo nombre sea 'person'
    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre = modelo.names[clase_id]
        if nombre == "person":
            conteo_personas += 1

    # Generar la imagen con las cajas dibuajadas por YOLO
    frame_anotado = resultados[0].plot()

    # Superponer el texto del contador en la esquina superior izquierda
    cv2.putText(
        frame_anotado,
        f"Personas: {conteo_personas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Mostrar ventana
    cv2.imshow("Contador de personas - Q para salir", frame_anotado)

    # Tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la cámara y cerrar ventanas
camara.release()
cv2.destroyAllWindows()