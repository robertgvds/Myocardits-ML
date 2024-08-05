import os

LIMITACAO = 5

def manter_imagens(pasta):
    # Lista todos os arquivos na pasta
    arquivos = os.listdir(pasta)

    # Calcula os índices para manter os arquivos
    total_arquivos = len(arquivos)
    if LIMITACAO == 5:
        indices = [0, total_arquivos // 3, total_arquivos // 2, 2 * total_arquivos // 3, total_arquivos - 1]
    if LIMITACAO == 10:
        indices = [0, total_arquivos // 10, total_arquivos // 5, 3 * total_arquivos // 10, 9 * total_arquivos // 10]
    arquivos_manter = [arquivos[i] for i in indices]

    # Remove os arquivos que não estão na lista de arquivos a serem mantidos
    for arquivo in arquivos:
        if arquivo not in arquivos_manter:
            caminho_arquivo = os.path.join(pasta, arquivo)
            os.remove(caminho_arquivo)
            print(f"Arquivo {arquivo} removido da pasta {pasta[-18:]}")

# Diretório principal (PASTA_PRIMARIA)
pasta_primaria = "C:/Users/rober/Documents/MyocarditisML/DATASETS/dataset-myocarditis-limited-to-five/224/"

# Percorre todas as subpastas dentro da PASTA_PRIMARIA
for subpasta1 in os.listdir(pasta_primaria):
    caminho_subpasta1 = os.path.join(pasta_primaria, subpasta1)
    if os.path.isdir(caminho_subpasta1):
        for subpasta2 in os.listdir(caminho_subpasta1):
            caminho_subpasta2 = os.path.join(caminho_subpasta1, subpasta2)
            if os.path.isdir(caminho_subpasta2):
                for subpasta3 in os.listdir(caminho_subpasta2):
                    caminho_subpasta3 = os.path.join(caminho_subpasta2, subpasta3)
                    if os.path.isdir(caminho_subpasta3):
                        manter_imagens(caminho_subpasta3)

def contar_imagens(pasta):
    # Lista todos os arquivos na pasta
    arquivos = os.listdir(pasta)

    # Conta o número de arquivos
    numero_imagens = len(arquivos)

    # Verifica se o número de imagens é 5 ou menos
    if numero_imagens <= LIMITACAO:
        print(f"A pasta {pasta[-18:]} contém {numero_imagens} imagem(s).")
    else:
        print(f"A pasta {pasta[-18:]} contém mais de {LIMITACAO} imagens.")

# Percorre todas as subpastas dentro da estrutura
for subpasta1 in os.listdir(pasta_primaria):
    caminho_subpasta1 = os.path.join(pasta_primaria, subpasta1)
    if os.path.isdir(caminho_subpasta1):
        for subpasta2 in os.listdir(caminho_subpasta1):
            caminho_subpasta2 = os.path.join(caminho_subpasta1, subpasta2)
            if os.path.isdir(caminho_subpasta2):
                for subpasta3 in os.listdir(caminho_subpasta2):
                    caminho_subpasta3 = os.path.join(caminho_subpasta2, subpasta3)
                    if os.path.isdir(caminho_subpasta3):
                        contar_imagens(caminho_subpasta3)
