
import cv2
from cv2 import aruco

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Crear el diccionario predefinido (6x6 con 250 marcadores)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Inicializar el detector de ArUco
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    # Capturar un frame desde la cámara
    ret, frame = cap.read()

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los marcadores ArUco en la imagen
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Dibujar los marcadores detectados en la imagen
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Calcular el centro de cada marcador detectado
        for corner in corners:
            # Obtener los puntos de las esquinas del marcador
            corners_array = corner.reshape(4, 2)
            top_left = corners_array[0]
            bottom_right = corners_array[2]
            
            # Calcular el centro del marcador
            center_x = int((top_left[0] + bottom_right[0]) / 2)
            center_y = int((top_left[1] + bottom_right[1]) / 2)

            # Dibujar un círculo en el centro del marcador
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # Mostrar las coordenadas del centro
            print(f"Marcador ID: {ids}, Centro: ({center_x}, {center_y})")

    # Mostrar el frame resultante
    cv2.imshow('Aruco Detection', frame)

    # Romper el loop si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
