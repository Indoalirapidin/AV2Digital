import cv2
import numpy as np

# Carregar a imagem
img = cv2.imread('img_folha_1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Máscaras para folhas saudáveis (verde) e danificadas (marrom/amarelo)
saudavel = cv2.inRange(hsv, (36, 40, 40), (85, 255, 255))
danificada = cv2.inRange(hsv, (10, 100, 20), (25, 255, 255))

# Operações morfológicas para refinar
kernel = np.ones((5, 5), np.uint8)
saudavel = cv2.morphologyEx(saudavel, cv2.MORPH_OPEN, kernel)
danificada = cv2.morphologyEx(danificada, cv2.MORPH_OPEN, kernel)

# Criar imagem de saída colorida
resultado = img.copy()
resultado[saudavel > 0] = [0, 255, 0]     # Verde para saudável
resultado[danificada > 0] = [0, 0, 255]   # Vermelho para danificada

cv2.imwrite('resultado_folha.jpg', resultado)