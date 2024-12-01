#4. Interpretar os resultados e extrair insights relevantes: 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Resultados obtidos
resultados = {
    "KNN": {
        "Melhores Hiperparâmetros": {'n_neighbors': 1, 'weights': 'uniform'},
        "Métricas": {
            "Acurácia": 0.89,
            "Precisão": 0.89,
            "Recall": 0.89,
            "F1-Score": 0.89,
            "Matriz de Confusão": [[17, 0, 3], [2, 19, 0], [2, 0, 20]],
        },
    },
    "SVM": {
        "Melhores Hiperparâmetros": {'C': 1, 'kernel': 'linear'},
        "Métricas": {
            "Acurácia": 0.90,
            "Precisão": 0.91,
            "Recall": 0.90,
            "F1-Score": 0.91,
            "Matriz de Confusão": [[17, 0, 3], [1, 20, 0], [2, 0, 20]],
        },
    },
    "Random Forest": {
        "Melhores Hiperparâmetros": {'max_depth': 20, 'n_estimators': 5},
        "Métricas": {
            "Acurácia": 0.87,
            "Precisão": 0.88,
            "Recall": 0.87,
            "F1-Score": 0.87,
            "Matriz de Confusão": [[17, 0, 3], [1, 20, 0], [4, 0, 18]],
        },
    },
    "Regressão Logística": {
        "Melhores Hiperparâmetros": {'C': 10, 'solver': 'lbfgs'},
        "Métricas": {
            "Acurácia": 0.92,
            "Precisão": 0.92,
            "Recall": 0.92,
            "F1-Score": 0.92,
            "Matriz de Confusão": [[17, 0, 3], [0, 21, 0], [2, 0, 20]],
        },
    },
}

# Insights
def extrair_insights(resultados):
    print("Análise dos Resultados")
    print("=" * 40)


    modelos = []
    acuracias = []
    precisaoes = []
    recalls = []
    f1_scores = []

    for modelo, detalhes in resultados.items():
        print(f"Modelo: {modelo}")
        print(f"Melhores Hiperparâmetros: {detalhes['Melhores Hiperparâmetros']}")

        # Métricas
        acuracia = detalhes['Métricas']['Acurácia']
        precisao = detalhes['Métricas']['Precisão']
        recall = detalhes['Métricas']['Recall']
        f1_score = detalhes['Métricas']['F1-Score']
        matriz_confusao = detalhes['Métricas']['Matriz de Confusão']

        
        modelos.append(modelo)
        acuracias.append(acuracia)
        precisaoes.append(precisao)
        recalls.append(recall)
        f1_scores.append(f1_score)

        # Interpretação
        print(f"Acurácia: {acuracia:.2f} - Este modelo classifica corretamente {acuracia * 100:.2f}% das amostras.")
        print(f"Precisão: {precisao:.2f} - As classificações positivas são corretas em {precisao * 100:.2f}% dos casos.")
        print(f"Recall: {recall:.2f} - O modelo identifica corretamente {recall * 100:.2f}% das amostras de cada classe.")
        print(f"F1-Score: {f1_score:.2f} - Equilíbrio entre precisão e recall.")

        # Matriz de Confusão
        print("Matriz de Confusão:")
        for linha in matriz_confusao:
            print(linha)

        
        if acuracia >= 0.90:
            print("🟢 Este modelo é altamente eficaz para o problema de classificação de grãos.")
        elif 0.85 <= acuracia < 0.90:
            print("🟡 Este modelo é aceitável, mas pode não ser ideal para a automação total.")
        else:
            print("🔴 Este modelo apresenta desempenho insuficiente para uso prático.")

        print("-" * 40)

    # Gráficos de barras para as métricas
    x = range(len(modelos))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.bar(x, acuracias, color='blue')
    plt.xticks(x, modelos)
    plt.ylabel('Acurácia')
    plt.title('Acurácia dos Modelos')

    plt.subplot(2, 2, 2)
    plt.bar(x, precisaoes, color='green')
    plt.xticks(x, modelos)
    plt.ylabel('Precisão')
    plt.title('Precisão dos Modelos')

    plt.subplot(2, 2, 3)
    plt.bar(x, recalls, color='red')
    plt.xticks(x, modelos)
    plt.ylabel('Recall')
    plt.title('Recall dos Modelos')

    plt.subplot(2, 2, 4)
    plt.bar(x, f1_scores, color='purple')
    plt.xticks(x, modelos)
    plt.ylabel('F1-Score')
    plt.title('F1-Score dos Modelos')

    plt.tight_layout()
    plt.show()


extrair_insights(resultados)