# CODIGO PARA DATA AUGMENTATION

#------------------------------------------------------------------------------
# BIBLIOTECAS

import matplotlib.pyplot as plt

import os
import numpy as np
import random
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers

#------------------------------------------------------------------------------
# DIRETÓRIOS

DATASET_PATH = "C:/Users/rober/Documents/MyocarditisML/DATASETS/localized/dataset-myocarditis-localized-selected/cropped_images/"
RESULT_PATH = "C:/Users/rober/Documents/MyocarditisML/DATA AUGMENTATION/myocardits-datasets-selected-augmentation-two/"

CLASS_TYPE = ['Normal', 'Sick']
INDIVIDUOS_NORMAL = [f'Individuo_{i:02}' for i in range(1, 17)]
INDIVIDUOS_SICK = [f'Individuo_{i:02}' for i in range(17, 48)]
SHORT_LONG = ['1-short', '2-long']
SERIES = [f'series{i:04}-Body' for i in range(1, 126)]

#------------------------------------------------------------------------------
# CONSTANTES E VARIÁVEIS GLOBAIS

IMG_SIZE = 100
NUM = 2

SEED = 10
np.random.seed(SEED) # semente geradora dos numeros aleatorios
random.seed(SEED)
tf.random.set_seed(SEED)

img_por_serie = 0
num_imagens = 0
num_imagens_ind = []

#------------------------------------------------------------------------------
# DATA AUGMENTATION

def random_invert_img(x, factor):
    if tf.random.uniform([]) < factor:
        x = (1-x)
        x = (x-0.6)
        x = layers.RandomContrast(1.)(x)
    else:
        x
    return x

def random_invert(factor):    
    return layers.Lambda(lambda x: random_invert_img(x, factor))

def data_augmentation(img):
    
    resize_and_rescale = tf.keras.Sequential([
      layers.Resizing(100,100),
      layers.Rescaling(1./255)
    ])
    
    img = resize_and_rescale(img)
    img = tf.cast(tf.expand_dims(img, 0), tf.float32)    

    
    model = tf.keras.Sequential()
    
    model.add(layers.RandomFlip("horizontal_and_vertical"))
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomContrast(0.5))
    model.add(layers.RandomZoom(0.2, None))
    model.add(random_invert(0.5))
    #model.add(layers.RandomBrightness(0.2, value_range=(0, 1)))
    
    new_images = []
    
    # plt.figure(figsize=(100, 100))
    for i in range(NUM):
      augmented_image = model(img)
      
      img_array = np.array(augmented_image[0] * 255, dtype=np.uint8)
      new_images.append(img_array)
      
      print(f"Imagem criada: {i+1}")
      
      #plt.subplot(3, 3, i + 1)
      #plt.imshow(augmented_image[0])
      #plt.axis("off")
    
    return new_images

#------------------------------------------------------------------------------
# PROCESSAMENTO E SALVAMENTO DE IMAGENS

def processamento_imagens(caminho):
    
    img = Image.open(caminho)
    img = np.array(img)
        
    return img

def carregar_imagens(diretorio, result_dir):
    global img_por_serie
    global num_imagens
    
    if not os.path.exists(diretorio):
        #print(f"Diretório de imagens não encontrado: {diretorio}")
        return
    
    for filename in os.listdir(diretorio):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            caminho = os.path.join(diretorio, filename)
            
            print(f"Processando imagem: {filename[:-4]}")
            
            image = processamento_imagens(caminho)
            
            new_result_dir = os.path.join(result_dir, f"{filename[:-4]}")
            if not os.path.exists(new_result_dir):
                os.makedirs(new_result_dir, exist_ok=True)
            
            image = Image.fromarray(np.array(image))
            image_path = os.path.join(new_result_dir, f"{filename}")
            image.save(image_path)
            
            num_new_image = 1
            
            new_images = data_augmentation(image)
            for new_img in new_images:
                image = Image.fromarray(new_img)
                image_path = os.path.join(new_result_dir, f"{filename[:-4]}_{num_new_image:02}.jpg")
                image.save(image_path)
                
                num_new_image += 1
            
            num_imagens += num_new_image
            img_por_serie += 1

#------------------------------------------------------------------------------
# PERCORRENDO O DIRETÓRIO

for Class in CLASS_TYPE:
    if Class == 'Normal':
        INDIVIDUOS = INDIVIDUOS_NORMAL
    else:
        INDIVIDUOS = INDIVIDUOS_SICK
        
    for individuo in INDIVIDUOS:
        
        for eixo in SHORT_LONG:
        
            for serie in SERIES:
            
                image_directory = f'{DATASET_PATH}/{Class}/{individuo}/{eixo}/{serie}'
                
                # Diretório de saída para salvar as imagens recortadas
                output_directory = f"{RESULT_PATH}/{Class}/{individuo}/{eixo}/{serie}"

                carregar_imagens(image_directory, output_directory)

                img_por_serie = 0

        print(f'Número total de anotações do {individuo}: {num_imagens}')
        num_imagens_ind.append(num_imagens)
        num_imagens = 0

#------------------------------------------------------------------------------
# ANÁLISE DO NUM DE IMAGENS

print('Número de anotações por individuo:')
i = 1
for num in num_imagens_ind:
    print(f'Individuo {i:02}: {num} anotações')
    i+=1