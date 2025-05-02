# Análise e Classificação de Dados Radiômicos
**Disciplina:** Paradigmas de Aprendizagem de Máquina  
**Aluno:** Ryan Duarte Sarmento Pedrosa

**Professoras:** Thais Gaudêncio e Daniela Coelho  

---

## Questão 1 — Classificação Supervisionada

### Objetivo
Avaliar modelos de classificação (Árvore de Decisão e MLP) com base em dados radiômicos, após aplicar pré-processamento e redução de dimensionalidade.

---

### 1. Pré-processamento
- Remoção de colunas não numéricas
- Imputação pela média
- Normalização (Min-Max)

Justificativa:  
Garante consistência nos dados e prepara o conjunto para algoritmos que dependem de escala (ex: MLP). A imputação simples preserva a média e a distribuição original.
A remoção das colunas não numéricas ocorre, pois não são importantes para os modelos de aprendizagem.

---

### 2. PCA (Análise de Componentes Principais)
- Redução da dimensionalidade mantendo 95% da variância.

Justificativa:  
Reduz a complexidade do modelo e o risco de overfitting, além de acelerar o treinamento.

---

### 3. Modelos Treinados
- Árvore de Decisão: simples, interpretável.
- MLP (Perceptron Multicamadas): poderoso para dados complexos.

Avaliação feita com:
- Validação cruzada estratificada (10 folds)
- F1-score
- Curva ROC
- Matriz de confusão

Interpretação:  
A MLP se beneficia bastante do PCA e da normalização. A árvore de decisão teve desempenho razoável, mesmo sem PCA, mas menos robusta.

---

## Questão 2 — Agrupamento Não Supervisionado

### Objetivo
Descobrir possíveis agrupamentos naturais no conjunto de dados sem usar rótulos.

---

### 1. K-means
- Número de clusters definido via método do cotovelo
- Avaliação com Silhouette Score

### 2. Hierárquico
- Método de linkage: Ward
- Avaliação visual com dendrograma
- Avaliação com Silhouette Score

Interpretação:  
Ambos os métodos sugerem 3 agrupamentos plausíveis, com resultados similares no Silhouette Score. O dendrograma reforçou a estrutura hierárquica dos dados.

---

## Conclusões

- O pipeline de pré-processamento, PCA e modelagem supervisionada foi eficaz na classificação.
- O uso do PCA melhorou o desempenho da MLP e simplificou o problema.
- Os métodos de clusterização revelaram agrupamentos coerentes mesmo sem rótulo.
- O trabalho segue boas práticas de ciência de dados, incluindo validação cruzada, métricas adequadas e análise visual.

---

## Como Executar

1. Certifique-se de ter o Python 3.8+ instalado.
2. Instale as dependências com:
   ```bash
   pip install -r requirements.txt
   ```
