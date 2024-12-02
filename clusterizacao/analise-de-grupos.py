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

def conta_imagens_por_grupo(diretorio):
    grupos = {}
    
    for grupo in os.listdir(diretorio):
        grupo_path = os.path.join(diretorio, grupo)
        if os.path.isdir(grupo_path):
            # Inicializa contagem do grupo
            grupos[grupo] = {'total': 0, 'normals': 0, 'sicks': 0}
            
            normal_path = os.path.join(grupo_path, 'Normal')
            sick_path = os.path.join(grupo_path, 'Sick')
            
            if os.path.exists(normal_path):
                normals = conta_imagens(normal_path)
                grupos[grupo]['normals'] = normals
                grupos[grupo]['total'] += normals
            
            if os.path.exists(sick_path):
                sicks = conta_imagens(sick_path)
                grupos[grupo]['sicks'] = sicks
                grupos[grupo]['total'] += sicks

    return grupos

DATASET_PATH = 'C:/Users/rober/Downloads/cluestering-results'

grupos = conta_imagens_por_grupo(DATASET_PATH)

for grupo, contagens in grupos.items():
    print(f'\n{grupo}:')
    print(f'  Total de imagens: {contagens["total"]}')
    print(f'  Saudáveis (Normal): {contagens["normals"]}')
    print(f'  Doentes (Sick): {contagens["sicks"]}')

total_imagens_por_grupo = [contagens['total'] for contagens in grupos.values()]
total_normals_por_grupo = [contagens['normals'] for contagens in grupos.values()]
total_sicks_por_grupo = [contagens['sicks'] for contagens in grupos.values()]

media_total = np.mean(total_imagens_por_grupo)
media_normals = np.mean(total_normals_por_grupo)
media_sicks = np.mean(total_sicks_por_grupo)

print(f'\nMédia de imagens por grupo: {media_total:.2f}')
print(f'Média de imagens saudáveis por grupo: {media_normals:.2f}')
print(f'Média de imagens doentes por grupo: {media_sicks:.2f}')
