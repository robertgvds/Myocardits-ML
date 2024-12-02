# TREINAMENTO USANDO A REDE INICIAL CATEGORICA E COM DADOS EM ARRAYS 

!pip install efficientnet

#------------------------------------------------------------------------------
# BIBLIOTECAS

import os
import numpy as np
from PIL import Image
import sys
import random

# BIBLIOTECAS DEEP LEARNING
import datetime
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from efficientnet.tfkeras import EfficientNetB7  # Arquitetura EfficientNet

#------------------------------------------------------------------------------
# CONSTANTES

DATASET_TYPE = ['cleaned', 'selected']
DATASET_TYPE = DATASET_TYPE[1] # Diretorio utilizado no treinamento

SEED = 10
np.random.seed(SEED) # semente geradora dos numeros aleatorios
random.seed(SEED)
tf.random.set_seed(SEED)

# CLASSES UTILIZADAS
NORMAL, SICK = [1, 0], [0, 1]

N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = (32 if DATASET_TYPE == 'selected' else 512)
TARGET_SIZE = (100, 100)

#------------------------------------------------------------------------------
# CONSTANTES

DATASET_TYPE = ['cleaned', 'selected']
DATASET_TYPE = DATASET_TYPE[1] # Diretorio utilizado no treinamento

SEED = 10
np.random.seed(SEED) # semente geradora dos numeros aleatorios
random.seed(SEED)
tf.random.set_seed(SEED)

# CLASSES UTILIZADAS
NORMAL, SICK = [1, 0], [0, 1]

N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = (32 if DATASET_TYPE == 'selected' else 512)
TARGET_SIZE = (100, 100)

#------------------------------------------------------------------------------
# DIRETÓRIOS DOS DATASETS

DATASET_PATH = f'C:/Users/rober/Documents/MyocarditisML/DATASETS/dataset-myocarditis-resized_{DATASET_TYPE}/'
RESULTS_PATH = f'C:/Users/rober/Documents/MyocarditisML/TREINAMENTOS-ARRAY/REDE-INICIAL-CONV2D/{DATASET_TYPE}/'

NORMAL_PATH = DATASET_PATH + '/Normal/'
SICK_PATH = DATASET_PATH + '/Sick/'


# Diretorios de cada Individuo:
normal_datasets = [f'{NORMAL_PATH}Individuo_{i:02}/' for i in range(1, 17)]
sick_datasets = [f'{SICK_PATH}Individuo_{i:02}/' for i in range(17, 48)]

print(normal_datasets)
print(sick_datasets)

NORMAL_SPLITS = [[9, 10, 12, 15],
                 [1,8,11],
                 [4,5,14],
                 [2,7,13],
                 [3,6,16]]

SICK_SPLITS = [[21,23,26,27,36,38],
               [18,37,39,40,41,45,46],
               [20,24,28,29,31,32],
               [19,22,30,33,42,47],
               [17,25,34,35,43,44]]

normal_splits = [[],[],[],[],[]]
sick_splits = [[],[],[],[],[]]

for split in range(N_FOLDS):
    normal_splits[split].extend(normal_datasets[i-1] for i in NORMAL_SPLITS[split])
    sick_splits[split].extend(sick_datasets[i-17] for i in SICK_SPLITS[split])
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

    for pasta, subpastas, arquivos in os.walk(diretorio + '/'):
        for arquivo in arquivos:
            if any(arquivo.endswith(ext) for ext in extensoes_imagens):
                pasta_principal = pasta.split(os.sep)[0]
                contador[pasta_principal] = contador.get(pasta_principal, 0) + 1
    return contador

for dataset in ['CLEANED']:
    
    total = conta_imagens(DATASET_PATH)
    normals = conta_imagens(NORMAL_PATH)
    sicks = conta_imagens(SICK_PATH)
    
    porcentagem_normals = normals/total * 100
    porcentagem_sicks = sicks/total * 100
    
    print(f'\n\nDADOS DO DATASET {dataset}:')
    print(f'Quantidade total de imagens: {total:.2f}')
    print(f'Saudáveis: {normals} ou {porcentagem_normals:.2f}%')
    print(f'Doentes: {sicks} ou {porcentagem_sicks:.2f}%')

    

    i = 1
    total_imagens = []
    
    print("Pacientes saudáveis:")
    for individuo in normal_datasets:
        contagem = conta_imagens(individuo)
        print(f'Individuo {i}: {contagem}\n{individuo}')
        total_imagens.append(contagem)
        i = i + 1
    
    print("Pacientes doentes:")
    for individuo in sick_datasets:
        contagem = conta_imagens(individuo)
        print(f'Individuo {i}: {contagem}\n{individuo}')
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
    
#------------------------------------------------------------------------------
# CARREGAMENTO DE DADOS

print('\nIniciando carregamento e processamento das imagens..............!')

def carregar_imagens(diretorio):
    global num_imagens  # Adicione esta linha
    imagens = []
    for pasta_atual, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                caminho = os.path.join(pasta_atual, arquivo)
                
                img = Image.open(caminho)  # Carrega a imagem
                img = img.resize(TARGET_SIZE)  # Redimensiona para 100x100
                img_array = np.array(img)  # Converte para array NumPy
                rgb_array = np.tile(img_array[:, :, np.newaxis], (1, 1, 3))
                imagens.append(rgb_array)  # Adiciona à lista de imagens
                
                num_imagens += 1
                sys.stdout.write("\rNumero de imagens carregados: %i" % num_imagens)
                sys.stdout.flush()
                
    return imagens

num_imagens = 0

print('\nPacientes normais:')
normal_groups = []
for diretorios in normal_splits:
    imagens = []
    for individuos in diretorios:
        imagens.extend(carregar_imagens(individuos + '/'))
    normal_groups.append(imagens)

num_imagens = 0

print('\n\nPacientes doentes:')
sick_groups = []
for diretorios in sick_splits:
    imagens = []
    for individuos in diretorios:
        imagens.extend(carregar_imagens(individuos + '/'))
    sick_groups.append(imagens)

# DATASETS SEPRADAOS EM 5 PARA VALIDAÇÃO CRUZADA    
x_data = [[],[],[],[],[]]
y_data = [[],[],[],[],[]]

print('\n\nNumero de imagens por fold:')
for i in range(N_FOLDS):
    x_data[i].extend(path for path in normal_groups[i])
    y_data[i].extend(NORMAL for path in normal_groups[i])
    x_data[i].extend(path for path in sick_groups[i])
    y_data[i].extend(SICK for path in sick_groups[i])
    print(f'Fold {i+1}: {len(x_data[i])} imagens ({len(normal_groups[i])} saudáveis e {len(sick_groups[i])} doentes).')
    
# Arquitetura CNN
def efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

#------------------------------------------------------------------------------
# SEPARAÇÃO DE DADOS E TREINAMENTO

lst_accuracy=[]
lst_acc=[]
lst_loss=[]
lst_reports=[]
lst_AUC=[]
lst_matrix=[]
lst_times=[]
lst_history=[]

#------------------------------------------------------------------------------
# TREINAMENTO POR FOLDS

for fold in range(1, N_FOLDS+1):

    print(f'\n\nFOLD {fold}:')

    #--------------------------------------------------------------------------
    # CARREGAMENTO DAS IMAGENS DE TREINAMENTO, VALIDACAO E TESTE

    print(f'\nCarregamento das imagens do fold {fold} para treinamento.............!')

    folds = [1, 2, 3, 4, 5]

    x_test = np.array(x_data[fold-1])
    y_test = np.array(y_data[fold-1])
    folds.remove(fold)

    x_valid = np.array(x_data[folds[0]])
    y_valid = np.array(y_data[folds[0]])
    folds.remove(folds[0])

    x_train, y_train = [], []
    for i in folds:
        x_train.extend(x_data[i-1])
        y_train.extend(y_data[i-1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print(f'Numero de imagens no treinamento: {len(x_train)} imagens.')
    print(f'Numero de imagens na validação: {len(x_valid)} imagens.')
    print(f'Numero de imagens no teste: {len(x_test)} imagens.')

    #--------------------------------------------------------------------------
    # ARQUITETURA E COMPILACAO
    
    
    input_shape = (100, 100, 3)
    num_classes = 2  # Número de classes no seu problema

    # Arquitetura CNN
    model = efficientnet_model(input_shape, num_classes)

    # Compilacao do modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    #--------------------------------------------------------------------------
    # TREINAMENTO
    print('\nIniciando o treinamento.........................................!\n')

    calback=CSVLogger(RESULTS_PATH + f'/logger_fold{fold}.log')

    # Treinando o modelo
    start=datetime.datetime.now()

    history=model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_valid, y_valid), callbacks=[calback])

    end=datetime.datetime.now()
    training_time=end-start

    # Salvamento do modelo
    model.save(RESULTS_PATH + f'/{DATASET_TYPE}-CNN-{fold}.h5')

    #--------------------------------------------------------------------------
    # TESTE

    # Testando o modelo
    print("\nTestando imagens................................................!\n")

    # Acuracia e Perda do Teste
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(model.metrics_names)

    #--------------------------------------------------------------------------
    # ARMAZENAMENTO DOS INFORMACOES

    # Fazendo previsões
    predicts = model.predict(x_test)
    predicts = predicts.argmax(axis=1)

    # Obtendo os rótulos verdadeiros
    actuals=y_test.argmax(axis=1)

    # Calculando a curva ROC
    fpr, tpr, _ = roc_curve(actuals, predicts, pos_label=1)
    a = auc(fpr, tpr)

    # Gerando o relatório de classificação
    r = classification_report(actuals, predicts, zero_division=1)

    # Calculando a matriz de confusão
    c = confusion_matrix(actuals, predicts)
    accuracy = np.trace(c)/np.sum(c)

    lst_history.append(history)
    lst_times.append(training_time)
    lst_accuracy.append(accuracy)
    lst_acc.append(test_acc)
    lst_loss.append(test_loss)
    lst_AUC.append(a)
    lst_reports.append(r)
    lst_matrix.append(c)

#--------------------------------------------------------------------------
# SALVAMENTO DOS DADOS

print('\nSalvando informações da rede......................................!')

path = RESULTS_PATH + f'/{DATASET_TYPE}-resultados_rede-incial-conv2d.txt'
    
matrix_total = np.sum(lst_matrix, axis=0)
accuracy_total = np.trace(matrix_total)/np.sum(matrix_total)
    
losses=[]
val_losses=[]
accuracies=[]
val_accuracies=[]

for item in lst_history:
    
    history=item.history
    loss=history['loss']
    accuracy=history['categorical_accuracy']
    
    val_loss=history['val_loss']
    val_accuracy=history['val_categorical_accuracy']
    
    losses.append(sum(loss)/len(loss))
    accuracies.append(sum(accuracy)/len(accuracy))
    
    val_losses.append(sum(val_loss)/len(val_loss))
    val_accuracies.append(sum(val_accuracy)/len(val_accuracy))

f1=open(path,'w')
f1.write(f'TREINAMENTO USANDO A REDE INICIAL CATEGORICA COM DADOS EM ARRAYS E DATASET {DATASET_TYPE}\n')

f1.write('\nTest Accuracias: '+str(lst_acc)+'\nTest Losses: '+str(lst_loss))
f1.write('\n\nTest Accuracies Mean: '+str(np.mean(lst_acc)))

f1.write('\n\n__________________________________________________________\n')

f1.write('\n\nValid Accuracies: '+str(val_accuracies)+'\nValid Losses: '+str(val_losses))
f1.write('\n\nValid Accuracies Mean: '+str(np.mean(val_accuracies)))

f1.write('\n\n__________________________________________________________\n')

f1.write('\nAccuracies from Confusion Matrix: '+str(lst_accuracy))

f1.write('\n\nTotal Confusion Matrix: \n'+str(matrix_total)+'\n\n')
f1.write('\nTotal Accuracie from Confusion Matrix: '+str(accuracy_total))

f1.write('\n\n__________________________________________________________\n')

f1.write('\n\nMetrics for all Folds: \n\n')
for i in range(len(lst_reports)):
    f1.write(str(lst_reports[i]))
    f1.write('\n\nTraining Time: '+str(lst_times[i])+'\nAUC: '+str(lst_AUC[i]))
    f1.write('\n\nAcurácia: ' + str(lst_accuracy[i]))
    f1.write('\n\nMatriz de Confusao: \n'+str(lst_matrix[i])+'\n\n__________________________________________________________\n')
f1.close()