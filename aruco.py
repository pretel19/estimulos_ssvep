#generacion aruco
import cv2
from cv2 import aruco

# Crear el diccionario predefinido (en este caso, un diccionario 6x6 con 250 marcadores)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Especificar el ID del marcador que quieres generar
marker_id = 33  # Puedes elegir cualquier ID entre 0 y 249 en este diccionario
marker_size = 200  # Tamaño del marcador en píxeles

# Generar la imagen del marcador
marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Guardar la imagen generada como un archivo PNG
cv2.imwrite(f'aruco_marker_{marker_id}.png', marker_image)

# Mostrar la imagen generada en una ventana (opcional)
cv2.imshow('Generated ArUco Marker', marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()