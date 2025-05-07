import cv2
import numpy as np

# Carregar imagens
img = cv2.imread('img_fundo_verde_1.jpg')
fundo = cv2.imread('novo_background.jpg')
fundo = cv2.resize(fundo, (img.shape[1], img.shape[0]))

# Converter para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir intervalo para cor verde
green_min = np.array([35, 40, 40])
green_max = np.array([85, 255, 255])
mask = cv2.inRange(hsv, green_min, green_max)
mask_inv = cv2.bitwise_not(mask)

# Separar as partes da imagem
fg = cv2.bitwise_and(img, img, mask=mask_inv)
bg = cv2.bitwise_and(fundo, fundo, mask=mask)

# Combinar as imagens
resultado = cv2.add(fg, bg)
cv2.imwrite('resultado_chroma.jpg', resultado)