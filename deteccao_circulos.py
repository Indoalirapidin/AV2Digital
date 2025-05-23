import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carrega a imagem contendo círculos
imagem = cv2.imread('circulos/circulos_1.png')
imagem_resultado = imagem.copy()

# Converte para tons de cinza
imagem_pb = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica a Transformada de Hough para detectar círculos
detectados = cv2.HoughCircles(
    imagem_pb,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=80,
    param1=100,
    param2=50,
    minRadius=40,
    maxRadius=60
)

# Se forem encontrados círculos, desenha-os
if detectados is not None:
    detectados = np.round(detectados[0, :]).astype("int")
    for (cx, cy, raio) in detectados:
        cv2.circle(imagem_resultado, (cx, cy), raio, (0, 255, 0), 4)
        cv2.rectangle(imagem_resultado, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)

# Mostra as imagens lado a lado
plt.figure(figsize=(10, 5))

# Original
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

# Resultado com círculos
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB))
total_circulos = len(detectados) if detectados is not None else 0
plt.title(f"Detectados: {total_circulos}")
plt.axis('off')

plt.tight_layout()
plt.show()
