import os
import shutil

def contar_imagens(diretorio):
    # Corrigido para contar arquivos em todas as subpastas
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

    # Renomear pastas de forma segura com nomes temporários para evitar conflitos
    temp_pasta = os.path.join(diretorio_raiz, 'temp_rename')
    os.makedirs(temp_pasta, exist_ok=True)
    
    # Fase 1: Mover pastas para diretório temporário com nomes intermediários
    for i, (pasta, _) in enumerate(pastas_ordenadas, start=1):
        caminho_antigo = os.path.join(diretorio_raiz, pasta)
        caminho_temp = os.path.join(temp_pasta, f'temp_Individuo_{str(i).zfill(2)}')
        shutil.move(caminho_antigo, caminho_temp)

    # Fase 2: Mover as pastas temporárias de volta com o nome final
    for pasta_temp in os.listdir(temp_pasta):
        caminho_temp = os.path.join(temp_pasta, pasta_temp)
        caminho_final = os.path.join(diretorio_raiz, pasta_temp.replace('temp_', ''))
        shutil.move(caminho_temp, caminho_final)
    
    # Remover diretório temporário
    os.rmdir(temp_pasta)


# Caminho dos datasets
DATASET_PATH = 'C:/Users/rober/Documents/MyocarditisML/DATASETS/localized/cleaned-reordenado'
NORMAL_PATH = DATASET_PATH + '/Normal/'
SICK_PATH = DATASET_PATH + '/Sick/'

# Renomear as pastas dentro dos diretórios
renomear_pastas(NORMAL_PATH)
renomear_pastas(SICK_PATH)