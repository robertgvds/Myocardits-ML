import os
import pandas as pd

# Configurações iniciais
CLASS_TYPE = ['Normal', 'Sick']
INDIVIDUOS_NORMAL = [f'Individuo_{i:02}' for i in range(1, 17)]
INDIVIDUOS_SICK = [f'Individuo_{i:02}' for i in range(17, 48)]
SERIES = [f'series{i:04}-Body' for i in range(1, 160)]

dataset_path = 'C:/Users/rober/Documents/Myocardits-ML/datasets/dataset-myocarditis-cleaned/dataset-myocarditis-cleaned'
novo_dataset_path = 'C:/Users/rober/Documents/Myocardits-ML/datasets/dataset-myocarditis-cleaned/dataset-myocarditis-cleaned-renamed'

# Armazenar os caminhos para a planilha
caminhos_antigos = []
caminhos_novos = []

# Percorrer o dataset para renomear as imagens
for classe in CLASS_TYPE:
    INDIVIDUOS = INDIVIDUOS_NORMAL if classe == 'Normal' else INDIVIDUOS_SICK
    
    for individuo_idx, individuo in enumerate(INDIVIDUOS, start=1):
        for serie_idx, serie in enumerate(SERIES, start=1):
            img_dir = os.path.join(dataset_path, classe, individuo, serie)
            
            if os.path.exists(img_dir):
                for img_idx, img_name in enumerate(os.listdir(img_dir), start=1):
                    img_path_antigo = os.path.join(img_dir, img_name)
                    
                    # Criar o novo nome
                    novo_nome = f"IND{individuo_idx:02}-S{serie_idx:03}-{img_idx:02}.jpg"
                    novo_dir = os.path.join(novo_dataset_path, classe, individuo, serie)
                    os.makedirs(novo_dir, exist_ok=True)
                    
                    # Caminho completo para o novo nome
                    img_path_novo = os.path.join(novo_dir, novo_nome)
                    
                    # Mover/renomear a imagem
                    os.rename(img_path_antigo, img_path_novo)
                    
                    # Armazenar os caminhos para a planilha
                    caminhos_antigos.append(img_path_antigo)
                    caminhos_novos.append(img_path_novo)

# Criar a planilha com os caminhos
df = pd.DataFrame({
    "Caminho Antigo": caminhos_antigos,
    "Caminho Novo": caminhos_novos
})
df.to_excel("caminhos_renomeados.xlsx", index=False)

print("Renomeação concluída. Planilha salva como 'caminhos_renomeados.xlsx'.")
