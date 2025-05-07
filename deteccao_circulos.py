import cv2
import numpy as np

# Carregar e pré-processar a imagem
img = cv2.imread('circulos_1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# Detecção de círculos usando Hough
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=10, maxRadius=100)

# Desenhar os círculos detectados
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

cv2.imwrite('resultado_circulos.jpg', img)