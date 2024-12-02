import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input  # Correção aqui
from tqdm import tqdm
from multiprocessing import Pool

# Definir o caminho do dataset original e o caminho de saída
dataset_path = 'C:/Users/rober/Documents/MyocarditisML/DATASETS/selected/dataset-myocarditis-selected/100'
output_path = 'C:/Users/rober/Documents/MyocarditisML/MyocarditisML/reorganaze'

# Carregar o modelo pré-treinado EfficientNet
model = EfficientNetB7(weights='imagenet', include_top=False)

# Função para carregar e pré-processar uma única imagem
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_data = img_to_array(img)
    img_data = preprocess_input(img_data)
    return img_data

# Função para extrair características e aplicar KMeans
def extract_and_cluster(img_paths, n_clusters=5, batch_size=64):
    features = []
    
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Processando lotes"):
        batch_paths = img_paths[i:i+batch_size]
        
        # Usar multiprocessamento para carregar as imagens em paralelo
        with Pool() as pool:
            batch_images = pool.map(load_and_preprocess_image, batch_paths)
        
        # Converter a lista de imagens em um array NumPy
        batch_images = np.array(batch_images)
        
        # Extrair as características do lote
        batch_features = model.predict(batch_images)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)
        
        features.append(batch_features)
    
    features = np.vstack(features)

    # Aplicar KMeans
    print(f"Aplicando KMeans com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

# Listas de classes, indivíduos e séries
CLASS_TYPE = ['Normal', 'Sick']
EIXO = ['1-short', '2-long']
INDIVIDUOS_NORMAL = [f'Individuo_{i:02}' for i in range(1, 17)]
INDIVIDUOS_SICK = [f'Individuo_{i:02}' for i in range(17, 48)]
SERIES = [f'series{i:04}-Body' for i in range(1, 150)]

# Armazenar os caminhos das imagens
image_paths = []

# Percorrer as imagens e coletar os caminhos
for Class in CLASS_TYPE:
    if Class == 'Normal':
        INDIVIDUOS = INDIVIDUOS_NORMAL
    else:
        INDIVIDUOS = INDIVIDUOS_SICK
        
    for individuo in INDIVIDUOS:
        for eixo in EIXO:
            for serie in SERIES:
                img_dir = os.path.join(dataset_path, Class, individuo, eixo, serie)
                if os.path.exists(img_dir):
                    for img_name in os.listdir(img_dir):
                        img_path = os.path.join(img_dir, img_name)
                        image_paths.append(img_path)

# Extrair características e aplicar KMeans
n_clusters = 5  # Defina o número de grupos desejado
print(f"Extraindo características e agrupando imagens em {n_clusters} grupos...")
groups = extract_and_cluster(image_paths, n_clusters=n_clusters)

# Criar novos diretórios e mover as imagens
print("Movendo imagens para os novos grupos...")
for group_id, img_path in tqdm(zip(groups, image_paths), total=len(image_paths)):
    # Construir o caminho resultante
    original_parts = img_path.split(os.sep)
    Class, individuo, serie, img_name = original_parts[-4], original_parts[-3], original_parts[-2], original_parts[-1]
    
    new_dir = os.path.join(output_path, f'GRUPO_{group_id}', Class, individuo, serie)
    os.makedirs(new_dir, exist_ok=True)
    
    # Copiar a imagem para o novo diretório
    shutil.copy(img_path, os.path.join(new_dir, img_name))

print("Processo concluído!")
