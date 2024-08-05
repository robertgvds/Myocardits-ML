import os
import shutil

def contar_imagens(diretorio):
    return sum([len(arquivos) for _, _, arquivos in os.walk(diretorio)])

def renomear_pastas(diretorio_raiz):
    # Contar imagens em cada subdiretório
    contagem = {}
    for pasta in os.listdir(diretorio_raiz):
        caminho_pasta = os.path.join(diretorio_raiz, pasta)
        if os.path.isdir(caminho_pasta):
            contagem[pasta] = contar_imagens(caminho_pasta)

    # Ordenar pastas por contagem de imagens
    pastas_ordenadas = sorted(contagem.items(), key=lambda x: x[1], reverse=True)

    # Renomear pastas
    for i, (pasta, _) in enumerate(pastas_ordenadas, start=1):
        caminho_antigo = os.path.join(diretorio_raiz, pasta)
        caminho_novo = os.path.join(diretorio_raiz, f'Individuo_{str(i).zfill(2)}')
        shutil.move(caminho_antigo, caminho_novo)


DATASET_PATH = 'C:/Users/rober/Documents/MyocarditisML/DATASETS/dataset-myocarditis-resized_cleaned/Sick/'
renomear_pastas(DATASET_PATH)