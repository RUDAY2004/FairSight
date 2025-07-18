import math
import random
from collections import Counter
from random import shuffle
import sys
import json

def balance_data(X, y):
    counter = Counter(y)
    min_count = min(counter.values())

    combined = list(zip(X, y))
    yes_samples = [pair for pair in combined if pair[1] == 1]
    no_samples = [pair for pair in combined if pair[1] == 0]

    balanced = yes_samples[:min_count] + no_samples[:min_count]
    shuffle(balanced)

    X_bal, y_bal = zip(*balanced)
    return list(X_bal), list(y_bal)

def read_csv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    headers = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    return headers, data

def zscore_column(col):
    mean = sum(col) / len(col)
    std = (sum((x - mean) ** 2 for x in col) / len(col)) ** 0.5 + 1e-8
    return [(x - mean) / std for x in col], mean, std

def preprocess(data):
    X_raw, y = [], []

    majors = list(set(row[12] for row in data))
    major_map = {m: i for i, m in enumerate(majors)}

    for row in data:
        features = []

        gender = 1 if row[3].lower() == 'male' else 0
        features.append(gender)

        for i in [5, 6, 7, 8, 9, 10]:
            features.append(float(row[i]))

        siblings = 1 if row[11].lower() == 'yes' else 0
        features.append(siblings)

        features.append(major_map[row[12]])
        X_raw.append(features)

        label = 1 if row[13].lower() == 'yes' else 0
        y.append(label)

    transposed = list(zip(*X_raw))
    normalized, means, stds = [], [], []
    for col in transposed:
        norm_col, mean, std = zscore_column(list(col))
        normalized.append(norm_col)
        means.append(mean)
        stds.append(std)
    X = list(zip(*normalized))
    return list(X), y, major_map, means, stds

def sigmoid(z):
    if z > 100:
        return 1.0
    elif z < -100:
        return 0.0
    return 1 / (1 + math.exp(-z))

def train(X, y, lr=0.05, epochs=2000):
    n = len(X[0])
    weights = [random.uniform(-1, 1) for _ in range(n)]
    bias = 0.0

    for epoch in range(epochs):
        for xi, yi in zip(X, y):
            z = sum(w * x for w, x in zip(weights, xi)) + bias
            pred = sigmoid(z)
            error = pred - yi

            for j in range(n):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error
    return weights, bias

def predict(X, weights, bias):
    preds = []
    for xi in X:
        z = sum(w * x for w, x in zip(weights, xi)) + bias
        prob = sigmoid(z)
        pred = 1 if prob >= 0.5 else 0
        preds.append(pred)
    return preds

def accuracy(preds, y):
    correct = sum(1 for p, a in zip(preds, y) if p == a)
    return correct / len(y) * 100

def predict_with_explanation(new_row, weights, bias, means, stds, major_map):
    feature_names = ['Gender', 'GPA', 'Percentage', 'Income', 'Activities',
                     'Volunteering Hours', 'Score', 'Siblings', 'Preferred Major']

    features = []

    gender = 1 if new_row[3].lower() == 'male' else 0
    norm_gender = (gender - means[0]) / stds[0]
    features.append(norm_gender)

    for idx, i in enumerate([5, 6, 7, 8, 9, 10], start=1):
        val = float(new_row[i])
        norm_val = (val - means[idx]) / stds[idx]
        features.append(norm_val)

    siblings = 1 if new_row[11].lower() == "yes" else 0
    norm_siblings = (siblings - means[7]) / stds[7]
    features.append(norm_siblings)

    major = major_map.get(new_row[12], 0)
    norm_major = (major - means[8]) / stds[8]
    features.append(norm_major)

    contributions = [w * x for w, x in zip(weights, features)]
    total_contribution = sum(abs(c) for c in contributions)
    explanation = {
        name: f"{(abs(c) / total_contribution * 100):.1f}%"
        for name, c in zip(feature_names, contributions)
    }

    z = sum(contributions) + bias
    prob = sigmoid(z)
    pred = 1 if prob >= 0.5 else 0

    return {
        "Name": new_row[1],
        "Prediction": "Yes" if pred == 1 else "No",
        "Confidence": f"{prob * 100:.2f}%",
        "Explanation": explanation
    }

def predict_by_id_or_name(csv_data, target_id_or_name, weights, bias, means, stds, major_map):
    for row in csv_data:
        if row[0] == target_id_or_name or row[1].lower() == target_id_or_name.lower():
            return predict_with_explanation(row, weights, bias, means, stds, major_map)
    return {"error": f"Entry with ID or Name '{target_id_or_name}' not found."}

# ------------------- Main -------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python Education_model.py <csv_path> <StudentID/Name>"}))
        sys.exit(1)

    csv_path = sys.argv[1]
    target = sys.argv[2]

    headers, data = read_csv(csv_path)
    X, y, major_map, means, stds = preprocess(data)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]
    X_train, y_train = balance_data(X_train, y_train)

    weights, bias = train(X_train, y_train, lr=0.05, epochs=2000)

    result = predict_by_id_or_name(data, target, weights, bias, means, stds, major_map)
    print(json.dumps(result))
