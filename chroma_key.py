import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carrega a imagem com fundo verde
img_original = cv2.imread('chromakey/img_fundo_verde_1.jpg')

# Converte para o espaço de cor LAB
img_lab = cv2.cvtColor(img_original, cv2.COLOR_BGR2LAB)

# Isola o canal 'a' (diferencia tons de verde e magenta)
canal_a_lab = img_lab[:, :, 1]

# Aplica limiarização com método de Otsu
_, mask_otsu = cv2.threshold(canal_a_lab, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Aplica a máscara Otsu na imagem original
objeto_segmentado = cv2.bitwise_and(img_original, img_original, mask=mask_otsu)

# Define o fundo como branco onde a máscara não detectou o objeto
objeto_segmentado[mask_otsu == 0] = (255, 255, 255)

# Normaliza o canal 'a' da imagem segmentada
canal_normalizado = cv2.normalize(cv2.cvtColor(objeto_segmentado, cv2.COLOR_BGR2LAB)[:, :, 1],
                                   dst=None, alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Aplica limiar fixo para remoção de ruídos
_, mask_ruido = cv2.threshold(canal_normalizado, 100, 255, cv2.THRESH_BINARY_INV)

# Ajusta o canal 'a' para neutralizar ruído restante
img_lab_mod = cv2.cvtColor(objeto_segmentado, cv2.COLOR_BGR2LAB)
img_lab_mod[:, :, 1][mask_ruido == 255] = 127

# Reconverte para BGR e define fundo como branco nas áreas não segmentadas
img_processada = cv2.cvtColor(img_lab_mod, cv2.COLOR_LAB2BGR)
img_processada[mask_otsu == 0] = (255, 255, 255)

# Carrega e redimensiona a nova imagem de fundo
novo_fundo = cv2.imread('chromakey/background_1.png')
novo_fundo = cv2.resize(novo_fundo, (img_original.shape[1], img_original.shape[0]))

# Inverte a máscara para identificar o fundo original
mask_fundo = cv2.bitwise_not(mask_otsu)

# Separa regiões de fundo e primeiro plano
area_fundo = cv2.bitwise_and(novo_fundo, novo_fundo, mask=mask_fundo)
area_objeto = cv2.bitwise_and(img_processada, img_processada, mask=mask_otsu)

# Combina objeto com novo fundo
imagem_final = cv2.add(area_fundo, area_objeto)

# Exibição dos resultados
plt.figure(figsize=(16, 8))

imagens = [
    (img_original, "1. Imagem Original"),
    (canal_a_lab, "2. Canal 'a' (LAB)", 'gray'),
    (mask_otsu, "3. Máscara Otsu", 'gray'),
    (objeto_segmentado, "4. Objeto Segmentado"),
    (canal_normalizado, "5. Canal Normalizado", 'gray'),
    (mask_ruido, "6. Máscara de Ruído", 'gray'),
    (img_processada, "7. Sem Fundo (Branco)"),
    (imagem_final, "8. Resultado Final")
]

for i, item in enumerate(imagens, 1):
    plt.subplot(2, 4, i)
    if len(item) == 3 and item[2] == 'gray':
        plt.imshow(item[0], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(item[0], cv2.COLOR_BGR2RGB))
    plt.title(item[1])
    plt.axis('off')

plt.tight_layout()
plt.show()
