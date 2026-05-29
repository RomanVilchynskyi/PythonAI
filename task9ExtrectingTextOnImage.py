import json
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# завантаження ресурсів nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')

# відкриття JSON файлу
with open(
    "product-reviews-list_product-reviews_captured-list_2026-05-04_19-13-31_019df3c3-f87e-7d18-86b1-06a74569d16e.json",
    "r",
    encoding="utf-8"
) as file:
    reviews = json.load(file)

# українські стоп-слова
uk_stopwords = {
    "і", "й", "та", "але", "бо", "що", "це", "як", "а", "не",
    "на", "у", "в", "до", "за", "з", "із", "при", "по", "для",
    "дуже", "вже", "ще", "ж", "би", "б", "чи"
}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

result = []

# лічильники
positive_count = 0
negative_count = 0
neutral_count = 0

for item in reviews:

    text = item["Review Text"]
    author = item["Author"]

    # 1 токенізація
    tokens = word_tokenize(text)

    # 2 видалення стоп-слів
    filtered = []

    for word in tokens:
        lower_word = word.lower()

        if word.isalpha() and lower_word not in uk_stopwords:
            filtered.append(lower_word)

    # 3 стеммінг
    stemmed = []

    for word in filtered:
        stemmed.append(stemmer.stem(word))

    # 4 лемматизація
    lemmatized = []

    for word in filtered:
        lemmatized.append(lemmatizer.lemmatize(word))

    # 5 аналіз тональності
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "positive"
        positive_count += 1

    elif polarity < 0:
        sentiment = "negative"
        negative_count += 1

    else:
        sentiment = "neutral"
        neutral_count += 1

    # збереження результату
    result.append({
        "author": author,
        "original": text,
        "tokens": tokens,
        "filtered": filtered,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
        "sentiment": sentiment
    })

# збереження JSON
with open(
    "processed_reviews.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=4
    )

# збереження CSV
df = pd.DataFrame(result)

df.to_csv(
    "processed_reviews.csv",
    index=False,
    encoding="utf-8-sig"
)

# статистика
print("Позитивних відгуків:", positive_count)
print("Негативних відгуків:", negative_count)
print("Нейтральних відгуків:", neutral_count)

print("Готово")