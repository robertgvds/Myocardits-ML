import os
import cv2
import numpy as np

# Caminho principal das imagens
caminho_imagens = "C:/Users/rober/Documents/MyocarditisML/LOCALIZACAO/dataset-myocardits-type01/images-and-annotations"
novo_caminho = "C:/Users/rober/Documents/MyocarditisML/LOCALIZACAO/dataset-myocardits-type01/cropped-images"

# Função para ler as anotações em YOLOv8
def ler_arquivo_txt(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        linhas = file.readlines()
    anotações = []
    for linha in linhas:
        valores = linha.strip().split()
        if len(valores) == 5:  # YOLOv8 usa [classe, x_center, y_center, largura, altura]
            anotações.append([float(valores[i]) for i in range(5)])
    return anotações

# Função para ajustar a imagem para formato quadrado
def ajustar_para_quadrado(imagem, largura, altura):
    # Calcular a dimensão quadrada (tamanho máximo)
    tamanho = max(largura, altura)
    # Criar uma nova imagem preta com o tamanho quadrado
    imagem_quadrada = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    # Calcular os offsets
    x_offset = (tamanho - largura) // 2
    y_offset = (tamanho - altura) // 2
    # Colocar a imagem original no centro da imagem quadrada
    imagem_quadrada[y_offset:y_offset+altura, x_offset:x_offset+largura] = imagem
    return imagem_quadrada

# Função para recortar a imagem
def recortar_imagem(imagem, anotações, largura_imagem, altura_imagem):
    for anotacao in anotações:
        # Convertendo as coordenadas normalizadas para coordenadas absolutas
        x_center, y_center, largura, altura = anotacao[1:]
        x1 = int((x_center - largura / 2) * largura_imagem)
        y1 = int((y_center - altura / 2) * altura_imagem)
        x2 = int((x_center + largura / 2) * largura_imagem)
        y2 = int((y_center + altura / 2) * altura_imagem)
        
        # Recorta a imagem com base nas coordenadas
        imagem_recortada = imagem[y1:y2, x1:x2]
        # Ajusta para quadrado
        imagem_quadrada = ajustar_para_quadrado(imagem_recortada, x2-x1, y2-y1)
        return imagem_quadrada
    return imagem

# Função para processar as imagens e as anotações
def processar_imagens_e_anotacoes(caminho_imagens, novo_caminho):
    for root, dirs, files in os.walk(caminho_imagens):
        for file in files:
            if file.endswith(".jpg"):  # Verifica se é uma imagem
                # Define os caminhos completos para imagem e anotação
                caminho_imagem = os.path.join(root, file)
                nome_base = os.path.splitext(file)[0]
                caminho_arquivo_txt = os.path.join(root, nome_base + ".txt")  # Arquivo .txt na mesma pasta

                # Verifica se existe a anotação para a imagem
                if os.path.exists(caminho_arquivo_txt):
                    # Carrega a imagem
                    imagem = cv2.imread(caminho_imagem)
                    altura_imagem, largura_imagem, _ = imagem.shape

                    # Lê as anotações
                    anotações = ler_arquivo_txt(caminho_arquivo_txt)

                    # Recorta a imagem com base nas anotações
                    imagem_recortada = recortar_imagem(imagem, anotações, largura_imagem, altura_imagem)

                    # Caminho para salvar a imagem recortada mantendo a estrutura de diretórios
                    caminho_destino_imagem = os.path.join(novo_caminho, os.path.relpath(root, caminho_imagens))
                    if not os.path.exists(caminho_destino_imagem):
                        os.makedirs(caminho_destino_imagem)
                    
                    # Salva a imagem recortada
                    cv2.imwrite(os.path.join(caminho_destino_imagem, nome_base + ".jpg"), imagem_recortada)
                    print(f"Imagem recortada salva: {os.path.join(caminho_destino_imagem, nome_base + '.jpg')}")
                else:
                    print(f"Anotação não encontrada para a imagem {file}.")

# Chama a função
processar_imagens_e_anotacoes(caminho_imagens, novo_caminho)
