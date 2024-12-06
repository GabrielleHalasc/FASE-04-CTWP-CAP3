# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="https://upload.wikimedia.org/wikipedia/commons/d/d4/Fiap-logo-novo.jpg" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# FIAP Fase4_Cap3  Implementando algoritmos de Machine Learning com Scikit-learn


## 👨‍🎓 Integrantes: 
- <a href="https://www.linkedin.com/in/amanda-fragnan-b61537255/" target="_blank">Amanda Fragnan RM 555684 </a>
- <a href="https://www.linkedin.com/in/cunhaandre/" target="_blank">Andre Cunha RM 560648</a>
- <a href="https://www.linkedin.com/in/gabriellehalasc/" target="_blank">Gabrielle Halasc RM 560147</a>

## 👩‍🏫 Professores:
### Tutor(a)
- <a href="https://www.linkedin.com/in/lucas-gomes-moreira-15a8452a/">Lucas Gomes Moreira</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/in/profandregodoi/">André Godoi</a>

## 📜 Descrição

Em cooperativas agrícolas de pequeno porte, a classificação dos grãos é realizada manualmente por especialistas, o que pode ser demorado e sujeito a erros humanos. Com o avanço do aprendizado de máquina, é possível automatizar esse processo, aumentando a eficiência e a precisão da classificação.

O objetivo é aplicar a metodologia CRISP-DM para desenvolver um modelo de aprendizado de máquina que classifique variedades de grãos de trigo com base em suas características físicas. Para isso você deverá:

Analisar e pré-processar os dados fornecidos.
Implementar e comparar diferentes algoritmos de classificação.
Otimizar os modelos para melhorar o desempenho.
Interpretar os resultados e extrair insights relevantes.
Descrição do Conjunto de Dados:

Para esta atividade, utilizaremos o "Seeds Dataset" disponível no UCI Machine Learning Repository.

Faça o download do Dataset "Seeds" em UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/236/seeds>.

O conjunto de dados contém medições de 210 amostras de grãos de trigo pertencentes a três variedades diferentes:

Kama;
Rosa;
Canadian.
 Atributos do conjunto de dados:

Área: medida da área do grão.
Perímetro: comprimento do contorno do grão.
Compacidade: calculada como 
Comprimento do Núcleo: comprimento do eixo principal da elipse equivalente ao grão.
Largura do Núcleo: comprimento do eixo secundário da elipse.
Coeficiente de Assimetria: medida da assimetria do grão.
Comprimento do Sulco do Núcleo: comprimento do sulco central do grão.
Tarefas:

1. Analisar e pré-processar os dados fornecidos: nesta etapa, você deve se familiarizar com o conjunto de dados, entender os atributos e como eles se correlacionam. Siga os passos a seguir:

Crie um arquivo notebook (.ipynb), pode ser jupyter (localmente) ou google Colab (em nuvem);

Importe o conjunto de dados e exiba as primeiras linhas para familiarizar-se com a estrutura;

Calcule estatísticas descritivas (média, mediana, desvio padrão) para cada característica.

Visualize a distribuição das características utilizando histogramas e boxplots.

Utilize gráficos de dispersão para identificar possíveis relações entre as características.

Identifique e trate valores ausentes;

Avalie a necessidade de escalar as características e aplique normalização ou padronização se necessário.



2. Implementar e comparar diferentes algoritmos de classificação: nesta etapa, você deverá construir modelos de classificação utilizando diferentes algoritmos e comparar seus desempenhos. Siga os seguintes passos:

Separe os dados em conjuntos de treinamento e teste (por exemplo, 70% para treinamento e 30% para teste).

Escolha pelo menos três algoritmos de classificação diferentes. Por exemplo:

K-Nearest Neighbors (KNN);

Support Vector Machine (SVM);

Random Forest;

Naive Bayes;

Logistic Regression.

Treine cada modelo utilizando o conjunto de treinamento.

Avalie cada modelo no conjunto de teste. Use métricas de desempenho, como acurácia, precisão, recall, F1-score e matrizes de confusão.

Compare o desempenho dos diferentes algoritmos.



3. Otimizar os modelos para melhorar o desempenho (se necessário): após a avaliação inicial, você deve avaliar se é necessário otimizar os modelos para aprimorar o desempenho, caso seja necessário:

Utilize Grid Search ou Randomized Search para encontrar os melhores hiperparâmetros para cada modelo.

Treine novamente cada modelo utilizando os melhores hiperparâmetros encontrados.

Avalie novamente cada modelo otimizado no conjunto de teste. Use métricas de desempenho, como acurácia, precisão, recall, F1-score e matrizes de confusão.

Verifique se houve melhorias significativas no desempenho.



4. Interpretar os resultados e extrair insights relevantes: 

Por fim, você deverá analisar profundamente os resultados e extrair conclusões significativas.

Interprete o desempenho de cada modelo e relacione os resultados com o contexto do nosso problema de classificação de grãos.

## 📁 Estrutura de pastas

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:

- <b>Fase4_Cap3.ipynb</b>: aqui está o arquivo notebook (.ipynb) com os entregáveis compilados.
- <b>conjunto_de_dados_sementes.txt</b>: aqui está o arquivo em txt utilizado na entrega.
- <b>conjunto_de_dados_sementes.csv</b>: aqui está o arquivo em csv utilizado na entrega.
- <b>README.md</b>: arquivo que serve como guia e explicação geral sobre a entrega (o mesmo que você está lendo agora).


## 🔧 Como executar o código

Clonar o repositório
Primeiro, faça o clone deste repositório localmente usando o Git:

git clone https://github.com/GabrielleHalasc/FIAP4-CAP3.git

Instalar dependências
Certifique-se de ter todas as dependências instaladas. Se estiver usando Python, você pode instalar os pacotes necessários com:

pip install -r requirements.txt

Executar o código
Dependendo da linguagem e estrutura do projeto, execute o código usando o comando apropriado. Para Python, por exemplo:

python main.py

Abrir o notebook
Para visualizar e executar o notebook que está contido no repositório, abra o arquivo Fase4_Cap3.ipynb. Você pode fazer isso usando o Jupyter Notebook ou qualquer outro ambiente compatível:

jupyter notebook Fase4_Cap3.ipynb

## Historico de lançamentos

- <b> 0.1.0 - 01/12/2024<b>
  

