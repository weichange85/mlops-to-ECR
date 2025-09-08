import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Ensure the output directory exists
    output_dir = os.path.dirname(__file__)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_model()

