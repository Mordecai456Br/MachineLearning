import pandas as pd

#baixar a database no link abaixo
#https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data?select=StudentPerformanceFactors.csv
dadosEstudantes = pd.read_csv("StudentPerformanceFactors.csv")

colunasManter=["Hours_Studied","Attendance","Extracurricular_Activities","Sleep_Hours","Motivation_Level",
               "Family_Income","School_Type","Gender","Exam_Score"]
dadosEstudantes = dadosEstudantes[colunasManter]
pd.set_option('display.max_columns', None)

notaCorte = 60
dadosEstudantes["Exam_Result"] = dadosEstudantes["Exam_Score"].apply(lambda x: 1 if x >= notaCorte else 0)
print(dadosEstudantes["Exam_Result"])
#Formatação dos dados
dadosEstudantes["Extracurricular_Activities"] = dadosEstudantes["Extracurricular_Activities"].replace("No", 0)
dadosEstudantes["Extracurricular_Activities"] = dadosEstudantes["Extracurricular_Activities"].replace("Yes", 1)

dadosEstudantes["Motivation_Level"] = dadosEstudantes["Motivation_Level"].replace({"Low": 0})
dadosEstudantes["Motivation_Level"] = dadosEstudantes["Motivation_Level"].replace("Medium", 1)
dadosEstudantes["Motivation_Level"] = dadosEstudantes["Motivation_Level"].replace("High", 2)

dadosEstudantes["Family_Income"] = dadosEstudantes["Family_Income"].replace("Low", 0)
dadosEstudantes["Family_Income"] = dadosEstudantes["Family_Income"].replace("Medium", 1)
dadosEstudantes["Family_Income"] = dadosEstudantes["Family_Income"].replace("High", 2)

dadosEstudantes["School_Type"] = dadosEstudantes["School_Type"].replace("Public", 0)
dadosEstudantes["School_Type"] = dadosEstudantes["School_Type"].replace("Private", 1)

dadosEstudantes["Gender"] = dadosEstudantes["Gender"].replace("Male", 0)
dadosEstudantes["Gender"] = dadosEstudantes["Gender"].replace("Female", 1)

print(dadosEstudantes.head())

#Separacao de dados
from sklearn.model_selection import train_test_split
X = dadosEstudantes.drop(columns=["Exam_Score", "Exam_Result"])
y = dadosEstudantes["Exam_Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"tamanho do conjunto de teste: {y_train.shape[0]}")

#Treinando o modelo
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score

modelo = ExtraTreesClassifier()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("precisaoBoladao: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Importância das características
importancia = modelo.feature_importances_
caracteristicas = X.columns

# graficaodobao
plt.figure(figsize=(10,6))
plt.barh(caracteristicas, importancia)
plt.xlabel('Importância')
plt.title('Importância das Características')
plt.show()
