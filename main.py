# Для решения этой задачи я создала virtual environment в vs code и установила следующие 
# пакеты из requirements.txt: scikit-learn, pandas (pip install requirements.txt)

# В начале программы импортируем необходимые библиотеки:
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Текстовые файлы я положила в одну папку с кодом, поэтому путь - это название файла
train_url = "train.txt"
test_url = "test.txt"

train_data = pd.read_csv(train_url, sep="\t", header=None, names=["Name", "Category"])
test_data = pd.read_csv(test_url, sep="\t", header=None, names=["Name"])

# Выведем 5 первых строк каждого DataFrame, чтобы проверить корректно ли считаны данные
print(train_data.head())
print(test_data.head())

# Зададим последовательность обработки данных (pipeline). 
# CountVectorizer - преобразует текстовые данные в числовые (каждый текст в вектор)
# MultinomialNB (Multinomial Naive Bayes) - это модель машинного обучения, которая 
# будет использоваться для предсказания категории на основе преобразованных текстовых данных.
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Обучаем модель с помощью обучающих данных
model.fit(train_data["Name"], train_data["Category"])

# Делаем предсказания для тестовых данных
predictions = model.predict(test_data["Name"])

# Сохраняем предсказанные категории в файл
pd.DataFrame(predictions).to_csv("predictions.txt", index=False, header=False)