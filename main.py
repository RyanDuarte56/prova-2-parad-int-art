import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ----------------- QUESTÃO 1 -----------------

# ----------------- CARREGAMENTO E PRÉ-PROCESSAMENTO -------------------

# Carregar a base
df = pd.read_csv('radiomic_data_binary.csv', header=0)

# Mapear rótulos para binário
df['class'] = df['class'].map({'BENIGN': 0, 'MALIGNANT': 1})

# Separar features e alvo
X = df.drop(columns=['class'])
y = df['class']

# ----------------- REMOVER COLUNAS NÃO NUMÉRICAS -------------------

# Remover colunas não numéricas (mantendo apenas as numéricas)
X_numeric = X.select_dtypes(include=[np.number])

# ----------------- IMPLICAÇÃO DE DADOS -------------------

# Imputação para dados numéricos (estratégia de média)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)

# Criar o DataFrame após imputação
df_imputed = pd.DataFrame(X_imputed, columns=X_numeric.columns)
df_imputed['class'] = y

# Salvar o arquivo CSV após imputação
df_imputed.to_csv('radiomic_data_imputed.csv', index=False)

# ----------------- NORMALIZAÇÃO DOS DADOS -------------------

# Normalizar os dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_imputed.drop(columns=['class']))

# Criar o DataFrame após normalização
df_scaled = pd.DataFrame(X_scaled, columns=df_imputed.drop(columns=['class']).columns)
df_scaled['class'] = y

# Salvar o arquivo CSV após normalização
df_scaled.to_csv('radiomic_data_scaled.csv', index=False)

# ----------------- REDUÇÃO DE DIMENSIONALIDADE -------------------

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=0.95, random_state=42)  # Retém 95% da variância
X_pca = pca.fit_transform(X_scaled)

# ----------------- MODELOS E AVALIAÇÃO -------------------

# Definir os modelos
modelos = {
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=1000)
}

# Configurar validação cruzada estratificada 10-fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Avaliar modelos
for nome, modelo in modelos.items():
    print(f'\nModelo: {nome}')
    
    # Avaliação com dados originais
    scores_f1 = cross_val_score(modelo, X_scaled, y, cv=cv, scoring='f1')
    print(f'F1-score (dados originais): {scores_f1.mean():.4f} ± {scores_f1.std():.4f}')
    
    # Avaliação com dados após PCA
    scores_f1_pca = cross_val_score(modelo, X_pca, y, cv=cv, scoring='f1')
    print(f'F1-score (após PCA): {scores_f1_pca.mean():.4f} ± {scores_f1_pca.std():.4f}')
    
    # Previsões para matriz de confusão e curva ROC
    y_pred = cross_val_predict(modelo, X_scaled, y, cv=cv, method='predict')
    y_proba = cross_val_predict(modelo, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    
    # Matriz de confusão
    cm = confusion_matrix(y, y_pred)
    
    # Nomes das classes para exibição na matriz
    classes = ['Benigno', 'Maligno']
    
    # Plotando a matriz de confusão com rótulos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusão - {nome}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Curva ROC - {nome}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.show()

# ----------------- QUESTÃO 2 -----------------

# Remover rótulos da classe (não supervisionado) e colunas não numéricas
X_clustering = df.drop(columns=['class']).select_dtypes(include=[np.number])

# ----------------- K-means -------------------

# Determinar o valor ideal de K usando o método do cotovelo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clustering)
    inertia.append(kmeans.inertia_)

# Plotando o gráfico do cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Inércia')
plt.show()

# Agora, vamos aplicar o K-means com o valor de K ideal (suponha K=3, por exemplo)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_clustering)

# Avaliação do K-means com Silhouette Score
sil_score_kmeans = silhouette_score(X_clustering, y_kmeans)
print(f'\nSilhouette Score (K-means): {sil_score_kmeans:.4f}')

# ----------------- Hierárquico -------------------

# Realizar o linkagamento hierárquico
Z = linkage(X_clustering, method='ward')

# Plotando o dendrograma
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrograma - Hierárquico')
plt.xlabel('Índices das amostras')
plt.ylabel('Distância')
plt.show()

# Aplicar o corte no dendrograma (supondo 3 clusters)
y_hierarchical = fcluster(Z, 3, criterion='maxclust')

# Avaliação do Hierárquico com Silhouette Score
sil_score_hierarchical = silhouette_score(X_clustering, y_hierarchical)
print(f'Silhouette Score (Hierárquico): {sil_score_hierarchical:.4f}')

# Comparando os resultados - Gráfico de clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_clustering.iloc[:, 0], X_clustering.iloc[:, 1], c=y_kmeans, cmap='viridis', label='K-means')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_clustering.iloc[:, 0], X_clustering.iloc[:, 1], c=y_hierarchical, cmap='viridis', label='Hierárquico')
plt.title('Hierárquico Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Comparação dos resultados
print(f'Número de clusters (K-means): {len(set(y_kmeans))}')
print(f'Número de clusters (Hierárquico): {len(set(y_hierarchical))}')