import cv2
import numpy as np
import time
from cv2 import aruco

posicion_x=300
posicion_y=300
# Definir las frecuencias para los 9 estímulos (en Hz)
frequencies = [6, 7, 8, 9, 10, 11, 12, 13, 14]

# Crear una ventana de tamaño adecuado
window_size = 1000
stimulus_size = 50  # tamaño de sepraci+ón
stimulus=40 # Tamaño reducido para los estímulos

cv2.namedWindow("SSVEP Stimuli", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("SSVEP Stimuli", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Especificar posiciones manualmente
# Cada tupla representa la posición (x, y) para un estímulo
positions = [
    (posicion_x-stimulus_size, posicion_y-stimulus_size),  # Posición para el estímulo 1
    (posicion_x, posicion_y-stimulus_size),  # Posición para el estímulo 2
    (posicion_x+stimulus_size, posicion_y-stimulus_size),  # Posición para el estímulo 3
    (posicion_x-stimulus_size, posicion_y),  # Posición para el estímulo 4
    (posicion_x,posicion_y),  # Posición para el estímulo 5
    (posicion_x+stimulus_size, posicion_y),  # Posición para el estímulo 6
    (posicion_x-stimulus_size, posicion_y+stimulus_size),  # Posición para el estímulo 7
    (posicion_x, posicion_y+stimulus_size),  # Posición para el estímulo 8
    (posicion_x+stimulus_size, posicion_y+stimulus_size),  # Posición para el estímulo 9
]

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Verificar que la cámara se haya abierto correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Tiempo inicial
start_time = time.time()


# Crear el diccionario predefinido (6x6 con 250 marcadores)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Inicializar el detector de ArUco
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    
    ####################################################################
     # Capturar un frame desde la cámara
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (window_size, window_size))
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
            posicion_x = int((top_left[0] + bottom_right[0]) / 2)
            posicion_y = int((top_left[1] + bottom_right[1]) / 2)

            # Dibujar un círculo en el centro del marcador
            #cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # Mostrar las coordenadas del centro
            #print(f"Marcador ID: {ids}, Centro: ({center_x}, {center_y})")

    positions = [
    (posicion_x-stimulus_size, posicion_y-stimulus_size),  # Posición para el estímulo 1
    (posicion_x, posicion_y-stimulus_size),  # Posición para el estímulo 2
    (posicion_x+stimulus_size, posicion_y-stimulus_size),  # Posición para el estímulo 3
    (posicion_x-stimulus_size, posicion_y),  # Posición para el estímulo 4
    (posicion_x,posicion_y),  # Posición para el estímulo 5
    (posicion_x+stimulus_size, posicion_y),  # Posición para el estímulo 6
    (posicion_x-stimulus_size, posicion_y+stimulus_size),  # Posición para el estímulo 7
    (posicion_x, posicion_y+stimulus_size),  # Posición para el estímulo 8
    (posicion_x+stimulus_size, posicion_y+stimulus_size),  # Posición para el estímulo 9
]
        
    # Capturar el frame de la cámara
    #ret, frame = cap.read()    
    
    # Redimensionar el frame capturado para mostrarlo junto con los estímulos
    #frame = cv2.resize(frame, (window_size, window_size))

    # Crear una imagen negra para los estímulos
    stimuli_img = np.zeros((window_size, window_size, 3), dtype=np.uint8)

    # Iterar sobre cada estímulo
    for idx, freq in enumerate(frequencies):
        # Calcular el periodo de la frecuencia
        period = 1 / freq
        
        # Alternar color cada medio periodo
        elapsed_time = time.time() - start_time
        if int(elapsed_time * 1000) % int(period * 1000) < (period * 500):
            color = (255, 255, 255)  # Blanco
        else:
            color = (0, 0, 0)  # Negro
        
        # Determinar la posición y dibujar el rectángulo correspondiente
        x, y = positions[idx]
        stimuli_img[y:y + stimulus, x:x + stimulus] = color
    
    # Combinar el frame de la cámara con los estímulos visuales
    combined_img = cv2.addWeighted(frame, 0.5, stimuli_img, 1, 0)

    # Mostrar la imagen combinada
    cv2.imshow("SSVEP Stimuli", combined_img)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
