from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = ["Win money now", "Hello friend", "Free offer", "Meeting tomorrow"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

test = vectorizer.transform(["Free money"])
print(model.predict(test))
