import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Load data
df = pd.read_csv("spam.csv", encoding="latin-1")

# 2. Keep only the useful columns and rename
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# 3. Convert labels: ham -> 0, spam -> 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 4. Features and target
X = df["message"]
y = df["label"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Convert text to numbers using CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 8. Check accuracy
score = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {score:.3f}")

# 9. Save model and vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
