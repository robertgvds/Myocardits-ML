# BIBLIOTECAS

diretorio_raiz = r"C:\Users\rober\Documents\MyocarditisML\DATASETS\dataset-myocarditis-cleaned-224"

'''
def criar_pastas(numero_pastas, diretorio):
    for i in range(1, numero_pastas + 1):
        nome_pasta = f"Fold_{i}"
        caminho_pasta = os.path.join(diretorio, nome_pasta)
        os.makedirs(caminho_pasta)
        
        for subpasta in ["train", "test", "valid"]:
            os.makedirs(os.path.join(caminho_pasta, subpasta))

# Exemplo: criar 5 pastas no diretório "C:\\MeuDiretorio"
criar_pastas(5, dir)
'''

import os
import sys
from PIL import Image

SIZE = 224
TARGET_SIZE = (SIZE, SIZE)

num_imagens = 0

def redimensionar_imagens(diretorio):
    global num_imagens
    for pasta_atual, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                caminho = os.path.join(pasta_atual, arquivo)
                imagem = Image.open(caminho)
                imagem_redimensionada = imagem.resize(TARGET_SIZE)
                imagem_redimensionada.save(caminho)
                num_imagens += 1
                sys.stdout.write("\rNumero de caminhos de imagens redimensionadas: %i" % num_imagens)
                sys.stdout.flush()

# Redimensiona as imagens em todas as subpastas
redimensionar_imagens(diretorio_raiz)

print("\nImagens redimensionadas com sucesso!")

num_imagens = 0

def verificar_imagens(diretorio):
    global num_imagens
    for pasta_atual, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                caminho = os.path.join(pasta_atual, arquivo)
            try:
                imagem = Image.open(caminho)
                largura, altura = imagem.size
                if largura != SIZE or altura != SIZE:
                    print(f"A imagem {arquivo} não está em {SIZE}x{SIZE} pixels.")
                elif imagem.mode != "L":
                    print(f"A imagem {arquivo} não está em escala de cinza.")
            except Exception as e:
                print(f"Erro ao processar a imagem {arquivo}: {str(e)}")
            num_imagens += 1
            sys.stdout.write("\rNumero de caminhos de imagens verificadas: %i" % num_imagens)
            sys.stdout.flush()

verificar_imagens(diretorio_raiz)

print("\nTodas as imagens verificadas!")
