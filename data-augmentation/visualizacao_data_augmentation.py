# CODIGO PARA DATA AUGMENTATION

#------------------------------------------------------------------------------
# BIBLIOTECAS

import matplotlib.pyplot as plt

import numpy as np
import random
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers

#------------------------------------------------------------------------------
# CONSTANTES E VARIÁVEIS GLOBAIS

IMG_SIZE = 100
NUM = 9

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
    model.add(layers.RandomZoom(0.2, None)) #
    model.add(random_invert(0.5))
    #model.add(layers.RandomBrightness(0.2, value_range=(0, 1)))
    
    new_images = []
    
    # plt.figure(figsize=(100, 100))
    for i in range(NUM):
      augmented_image = model(img)
      
      img_array = np.array(augmented_image[0] * 255, dtype=np.uint8)
      new_images.append(img_array)
      
      print(f"Imagem criada: {i+1}")
      
      plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_image[0])
      plt.axis("off")
    
    return new_images

#------------------------------------------------------------------------------
# PROCESSAMENTO E SALVAMENTO DE IMAGENS

def processamento_imagens(caminho):
    
    img = Image.open(caminho)
    img = np.array(img)
        
    return img

caminho = r"C:\Users\rober\Documents\MyocarditisML\DATASETS\localized\dataset-myocarditis-localized-selected\cropped_images\Normal\Individuo_01\1-short\series0027-Body\cropped_002.jpg"
img = processamento_imagens(caminho)
data_augmentation(img)

