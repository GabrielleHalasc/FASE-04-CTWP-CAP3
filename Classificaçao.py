import pandas as pd
from sklearn.model_selection import train_test_split
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

# Criar instâncias dos modelos
model_knn = KNeighborsClassifier()  # Incluindo o KNN
model_svm = SVC()
model_rf = RandomForestClassifier()
model_lr = LogisticRegression()

# Treinar os modelos
model_knn.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

# Fazer previsões
y_pred_knn = model_knn.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)


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


# Avaliar KNN
avaliar_modelo(y_test, y_pred_knn, "KNN")

# Avaliar SVM
avaliar_modelo(y_test, y_pred_svm, "SVM")

# Avaliar Random Forest
avaliar_modelo(y_test, y_pred_rf, "Random Forest")

# Avaliar Regressão Logística
avaliar_modelo(y_test, y_pred_lr, "Regressão Logística")