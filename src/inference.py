import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_digits

def load_model(path="model_train.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def run_inference(model, X):
    return model.predict(X)

if __name__ == "__main__":
    digits = load_digits()
    X = digits.data
    model = load_model()
    predictions = run_inference(model, X)

    print("Sample Predictions:", predictions[:10])


digits = load_digits()
X, y = digits.data, digits.target
model = load_model()

predictions = run_inference(model, X)

accuracy = accuracy_score(y, predictions)
f1 = f1_score(y, predictions, average='macro')

print(" Predictions:", predictions[:10])
print(f" Accuracy: {accuracy:.4f}")
print(f" F1-Score: {f1:.4f}")
