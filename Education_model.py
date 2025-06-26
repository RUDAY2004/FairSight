import math
import random
from collections import Counter
from random import shuffle

def balance_data(X, y):
    counter = Counter(y)
    min_count = min(counter.values())

    combined = list(zip(X, y))
    yes_samples = [pair for pair in combined if pair[1] == 1]
    no_samples  = [pair for pair in combined if pair[1] == 0]

    balanced = yes_samples[:min_count] + no_samples[:min_count]
    shuffle(balanced)

    X_bal, y_bal = zip(*balanced)
    return list(X_bal), list(y_bal)

# ------------------- CSV Reader -------------------
def read_csv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    headers = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    return headers, data

# ------------------- Normalize -------------------
def zscore_column(col):
    mean = sum(col) / len(col)
    std = (sum((x - mean)**2 for x in col) / len(col))**0.5 + 1e-8
    return [(x - mean) / std for x in col]


# ------------------- Preprocessing -------------------
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

    # Transpose to normalize each column
    transposed = list(zip(*X_raw))
    normalized = [zscore_column(list(col)) for col in transposed]
    X = list(zip(*normalized))  # Transpose back

    return list(X), y, major_map

# ------------------- Sigmoid (Safe) -------------------
def sigmoid(z):
    if z > 100:
        return 1.0
    elif z < -100:
        return 0.0
    return 1 / (1 + math.exp(-z))

# ------------------- Train Logistic Regression -------------------
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

# ------------------- Predict -------------------
def predict(X, weights, bias):
    preds = []
    for xi in X:
        z = sum(w * x for w, x in zip(weights, xi)) + bias
        prob = sigmoid(z)
        pred = 1 if prob >= 0.5 else 0
        preds.append(pred)
    return preds

# ------------------- Accuracy -------------------
def accuracy(preds, y):
    correct = sum(1 for p, a in zip(preds, y) if p == a)
    return correct / len(y) * 100

# ---------------------------- Explanation -----------------------------------------
def predict_with_explanation(new_row, weights, bias, col_stats, major_map):
    feature_names = ['Gender', 'GPA', 'Percentage', 'Income', 'Activities',
                     'Volunteering Hours', 'Score', 'Siblings', 'Preferred Major']

    features = []

    # Gender
    gender = 1 if new_row[3].lower() == 'male' else 0
    features.append(gender)

    # Numeric fields: GPA, Percentage, Income, Activities, Volunteering Hours, Score
    for idx, i in enumerate([5, 6, 7, 8, 9, 10]):
        val = float(new_row[i])
        min_val, max_val = col_stats[idx]
        norm_val = (val - min_val) / (max_val - min_val + 1e-8)
        features.append(norm_val)

    # Siblings
    siblings = 1 if new_row[11].lower() == "yes" else 0
    features.append(siblings)

    # Preferred Major
    major = major_map.get(new_row[12], 0)
    major_min, major_max = col_stats[7]
    major_norm = (major - major_min) / (major_max - major_min + 1e-8)
    features.append(major_norm)

    # Contributions
    contributions = [w * x for w, x in zip(weights, features)]
    total_contribution = sum(abs(c) for c in contributions)
    explanation = {
        name: f"{(abs(c) / total_contribution * 100):.1f}%"
        for name, c in zip(feature_names, contributions)
    }

    # Prediction
    z = sum(contributions) + bias
    prob = sigmoid(z)
    pred = 1 if prob >= 0.5 else 0

    return {
        "Student Name": new_row[1],
        "Prediction": "Yes" if pred == 1 else "No",
        "Confidence": f"{prob * 100:.2f}%",
        "Explanation": explanation
    }

def predict_by_id_or_name(csv_data, target_id_or_name, weights, bias, col_stats, major_map):
    for row in csv_data:
        if row[0] == target_id_or_name or row[1].lower() == target_id_or_name.lower():
            # Found the row → Predict with explanation
            result = predict_with_explanation(row, weights, bias, col_stats, major_map)
            return result
    return {"error": f"Entry with ID or Name '{target_id_or_name}' not found."}


# ------------------- Main -------------------
if __name__ == "__main__":
    headers, data = read_csv(r"C:\Users\madir\OneDrive\Desktop\SAPHACK\csvfiles\csvfiles\Education.csv")
    X, y, major_map = preprocess(data)

    # Shuffle before split
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    # 80/20 Train/Test Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train, y_train = balance_data(X_train, y_train)
    weights, bias = train(X_train, y_train, lr=0.05, epochs=2000)

    train_preds = predict(X_train, weights, bias)
    test_preds = predict(X_test, weights, bias)

    print(f"Train Accuracy: {accuracy(train_preds, y_train):.2f}%")
    print(f"Test Accuracy:  {accuracy(test_preds, y_test):.2f}%")

    col_stats = []
    for col in zip(*X_train):
        min_val = min(col)
        max_val = max(col)
        col_stats.append((min_val, max_val))

    # ------------------ Example Input ------------------
    # Format: [STU_ID,Name,Age,Gender,Country,GPA,Percentage,Income,Activities,Volunteering Hours,Score,Siblings,Preferred_Major,Scholarship]
    #example_input = ["101", "Aditi Sharma", "19", "Female", "India",
                    # "3.70", "91.5", "9500", "3", "120", "1350", "Yes", "Management", ""]

    target = input("Enter Student ID or Name: ")  # e.g., "101" or "Aditi Sharma"
    result = predict_by_id_or_name(data, target, weights, bias, col_stats, major_map)
    import json
    print("\nFinal JSON Output:")
    print(json.dumps(result, indent=2))

