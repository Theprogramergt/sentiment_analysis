
import os
import time
import joblib

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Create models folder
os.makedirs("models", exist_ok=True)

# 2. Load IMDb movie reviews dataset
print("Loading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

# 3. Use a smaller training sample for Model A
X_small, _, y_small, _ = train_test_split(
    X_train,
    y_train,
    train_size=0.20,
    random_state=42,
    stratify=y_train
)

print(f"Model A training samples: {len(X_small)}")
print(f"Model B training samples: {len(X_train)}")

# 4. Create a function to train and evaluate a model
def train_model(name, X, y):

    print(f"\nTraining {name}...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Save model
    joblib.dump(model, f"models/{name}.joblib")

    print(f"{name} accuracy: {accuracy:.4f}")
    print(f"{name} training time: {training_time:.2f} seconds")

    return {
        "model": model,
        "accuracy": accuracy,
        "training_time": training_time
    }


# 5. Train Model A and Model B
model_a_results = train_model("model_a", X_small, y_small)
model_b_results = train_model("model_b", X_train, y_train)

# 6. Display comparison
print("\n========== MODEL COMPARISON ==========")

print(f"Model A accuracy: {model_a_results['accuracy']:.4f}")
print(f"Model B accuracy: {model_b_results['accuracy']:.4f}")

print("\nTraining comparison:")
print(f"Model A: {model_a_results['training_time']:.2f} seconds")
print(f"Model B: {model_b_results['training_time']:.2f} seconds")

print("\nBoth models saved successfully!")
