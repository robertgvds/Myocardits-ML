import os
import numpy as np
import statistics

def conta_imagens(diretorio):
    total_imagens = 0
    for pasta, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith(('.png', '.jpg', '.jpeg')):
                total_imagens += 1
    return total_imagens

def conta_imagens_paciente(diretorio):
    extensoes_imagens = ['.jpg', '.jpeg', '.png', '.gif']
    contador = {}

    for pasta, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if any(arquivo.endswith(ext) for ext in extensoes_imagens):
                pasta_principal = pasta.split(os.sep)[0]
                contador[pasta_principal] = contador.get(pasta_principal, 0) + 1
    return contador

DATASET_TYPE = ['COMPLETE', 'CLEANED', 'SELECTED']

for dataset in ['CLEANED']:
    DATASET_PATH = 'C:/Users/rober/Downloads/heart-detection/cropped_images/'
    NORMAL_PATH = DATASET_PATH + '/Normal/'
    SICK_PATH = DATASET_PATH + '/Sick/'
    
    total = conta_imagens(DATASET_PATH)
    normals = conta_imagens(NORMAL_PATH)
    sicks = conta_imagens(SICK_PATH)
    
    porcentagem_normals = normals/total * 100
    porcentagem_sicks = sicks/total * 100
    
    print(f'\n\nDADOS DO DATASET {dataset}:')
    print(f'Quantidade total de imagens: {total:.2f}')
    print(f'Saudáveis: {normals} ou {porcentagem_normals:.2f}%')
    print(f'Doentes: {sicks} ou {porcentagem_sicks:.2f}%')

    pacientes_sicks = conta_imagens_paciente(SICK_PATH)
    pacientes_normais = conta_imagens_paciente(NORMAL_PATH)

    i = 1
    total_imagens = []
    
    print("Pacientes saudáveis:")
    for pasta, contagem in pacientes_normais.items():
        #print(f'O paciente saudável {i} contém {contagem} imagens.')
        print(f'Individuo {i}: {contagem}')
        total_imagens.append(contagem)
        i = i + 1
    
    print("Pacientes doentes:")
    for pasta, contagem in pacientes_sicks.items():
        #print(f'O paciente doente {i} contém {contagem} imagens.')
        print(f'Individuo {i}: {contagem}')
        total_imagens.append(contagem)
        i = i + 1
        
    mediaImagens = np.mean(total_imagens)
    medianImagens = np.median(total_imagens)
    modaImagens = statistics.mode(total_imagens)
    stdImagens = np.std(total_imagens)
    
    print(f'\nMédia de Imagens por paciente: {mediaImagens:.2f}')
    print(f'\nMediana de Imagens por paciente: {medianImagens:.2f}')
    print(f'\nModa de Imagens por paciente: {modaImagens:.2f}')
    print(f'\nDesvio Padrão de Imagens por paciente: {stdImagens:.2f}')
    
