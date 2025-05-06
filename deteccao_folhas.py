import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregamento da imagem da folha
folha = cv2.imread('folhas/img_folha_1.JPG')

# Conversão para RGB para exibição e HSV para segmentação
folha_rgb = cv2.cvtColor(folha, cv2.COLOR_BGR2RGB)
folha_hsv = cv2.cvtColor(folha, cv2.COLOR_BGR2HSV)

# Definição de faixas de cor no HSV para folhas saudáveis e danificadas
limite_verde_inf = np.array([30, 40, 40])
limite_verde_sup = np.array([90, 255, 255])

limite_danif_inf = np.array([10, 30, 30])
limite_danif_sup = np.array([30, 255, 255])

# Criação das máscaras binárias
mascara_saudavel = cv2.inRange(folha_hsv, limite_verde_inf, limite_verde_sup)
mascara_danificada = cv2.inRange(folha_hsv, limite_danif_inf, limite_danif_sup)

# Redução de ruídos nas máscaras com abertura morfológica
estrutura = np.ones((3, 3), np.uint8)
mascara_saudavel = cv2.morphologyEx(mascara_saudavel, cv2.MORPH_OPEN, estrutura)
mascara_danificada = cv2.morphologyEx(mascara_danificada, cv2.MORPH_OPEN, estrutura)

# Aplicação das máscaras nas regiões da imagem original
folha_saudavel = cv2.bitwise_and(folha_rgb, folha_rgb, mask=mascara_saudavel)
folha_danificada = cv2.bitwise_and(folha_rgb, folha_rgb, mask=mascara_danificada)

# Visualização dos resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(folha_rgb)
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(folha_saudavel)
plt.title("Região Saudável")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(folha_danificada)
plt.title("Região Danificada")
plt.axis('off')

plt.tight_layout()
plt.show()
