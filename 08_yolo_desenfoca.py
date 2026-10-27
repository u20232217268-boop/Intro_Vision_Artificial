import cv2

# Inicializar la cámara
camara = cv2.VideoCapture(0)

# Cargar el clasificador de rostros
ruta_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(ruta_cascade)

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo recibir el frame.")
        break

    # Convertir a escala de grises para realizar la detección
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Buscar rostros
    rostros = detector.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Aplicar el desenfoque en la región del rostro
    for (x, y, w, h) in rostros:
        # Extraer únicamente el área del rostro (ROI)
        rostro = frame[y:y+h, x:x+w]
        
        # Aplicar filtro gaussiano
        rostro_blur = cv2.GaussianBlur(rostro, (99, 99), 30)
        
        # Reemplazar la región original por la imagen desenfocada
        frame[y:y+h, x:x+w] = rostro_blur

    # Mostrar la ventana
    cv2.imshow("Rostros desenfocados - Q para salir", frame)

    # Tecla 'q' para finalizar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la cámara y cerrar ventanas
camara.release()
cv2.destroyAllWindows()