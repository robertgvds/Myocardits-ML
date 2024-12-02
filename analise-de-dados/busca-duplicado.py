import os
from collections import defaultdict

num_imgs_duplicadas = 0


def busca_arquivo_duplicado(diretorio):
    global num_imgs_duplicadas
    
    # Dicionário para armazenar o nome do arquivo e os caminhos onde ele aparece
    image_files = defaultdict(list)

    # Percorre todos os arquivos e subpastas
    for pasta, subpasta, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(pasta, arquivo)
                image_files[arquivo].append(file_path)

    ehDuplicado = False
    for file_name, paths in image_files.items():
        if len(paths) > 1:
            ehDuplicado = True
            print(f"Arquivo duplicado: {file_name}")
            for path in paths:
                num_imgs_duplicadas += 1
                print(f" - {path}")
            print("\n")

    #if not ehDuplicado:
        #print("Nenhum arquivo duplicado encontrado.")

diretorio = 'C:/Users/rober/Documents/MyocarditisML/DATASETS/cleaned/dataset-myocarditis-cleaned-240'

busca_arquivo_duplicado(diretorio)
print(f"Número de imagns duplicadas: {num_imgs_duplicadas}")