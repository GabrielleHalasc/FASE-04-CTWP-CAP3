import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carregar os dados pré-processados
# Substitua o caminho pelo caminho correto do seu arquivo
scaled_df = pd.read_csv(r'C:\Users\gabi_\Downloads\seeds_scaled.csv')

# Separar as features do target
X = scaled_df.drop("Class", axis=1)
y = scaled_df["Class"]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir os parâmetros para busca em cada modelo
param_grid_knn = {'n_neighbors': [1,3], 'weights': ['uniform', 'distance']}
param_grid_svm = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
param_grid_rf = {'n_estimators': [5, 6], 'max_depth': [None, 10, 20]}
param_grid_lr = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}

# Criar instâncias do GridSearchCV para cada modelo
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy')
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')

# Treinar os modelos com GridSearchCV
grid_search_knn.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_lr.fit(X_train, y_train)


# Função para exibir as métricas
def avaliar_modelo(y_true, y_pred, nome_modelo):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    matrix = confusion_matrix(y_true, y_pred)

    print(f"Métricas para {nome_modelo}:")
    print(f"Acurácia: {accuracy:.2f}")
    print(f"Precisão: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Matriz de Confusão:")
    print(matrix)
    print("-" * 30)


# Função para avaliar e exibir os resultados
def avaliar_grid_search(grid_search, nome_modelo):
    print(f"Melhores hiperparâmetros para {nome_modelo}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    avaliar_modelo(y_test, y_pred, nome_modelo)


# Avaliar os resultados do GridSearchCV
avaliar_grid_search(grid_search_knn, "KNN")
avaliar_grid_search(grid_search_svm, "SVM")
avaliar_grid_search(grid_search_rf, "Random Forest")
avaliar_grid_search(grid_search_lr, "Regressão Logística")