import math
import random
import csv
import json
import sys

# ------------------- Sigmoid -------------------
def sigmoid(z):
    return 1.0 if z > 100 else 0.0 if z < -100 else 1 / (1 + math.exp(-z))

# ------------------- Read CSV -------------------
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader if len(row) == 16]  # Skip malformed rows
    return headers, data

# ------------------- Normalize -------------------
def normalize_column(col):
    min_val, max_val = min(col), max(col)
    return [(x - min_val) / (max_val - min_val + 1e-8) for x in col], (min_val, max_val)

# ------------------- Preprocess -------------------
def preprocess(data):
    X_raw, y = [], []
    emps = list(set(row[6].strip() for row in data))
    purposes = list(set(row[12].strip() for row in data))
    locs = list(set(row[14].strip() for row in data))
    marital = list(set(row[13].strip() for row in data))

    emp_map = {e: i for i, e in enumerate(emps)}
    purpose_map = {p: i for i, p in enumerate(purposes)}
    loc_map = {l: i for i, l in enumerate(locs)}
    marital_map = {m: i for i, m in enumerate(marital)}

    for row in data:
        features = [
            float(row[3].strip()),  # Age
            float(row[4].strip()),  # Credit Score
            float(row[5].strip()),  # Income
            emp_map[row[6].strip()],  # Employment Type
            float(row[7].strip()),  # Loan Amount
            float(row[8].strip()),  # Loan Tenure
            float(row[9].strip()),  # Existing Debt
            int(row[10].strip()),  # Num Dependents
            1 if row[11].strip().lower() == "yes" else 0,  # Collateral
            purpose_map[row[12].strip()],  # Purpose
            marital_map[row[13].strip()],  # Marital Status
            loc_map[row[14].strip()]  # Location
        ]
        X_raw.append(features)
        y.append(1 if row[15].strip().lower() == "approved" else 0)

    transposed = list(zip(*X_raw))
    normalized, stats = [], []
    for col in transposed:
        norm_col, stat = normalize_column(list(map(float, col)))
        normalized.append(norm_col)
        stats.append(stat)

    X = list(zip(*normalized))
    return X, y, stats, emp_map, purpose_map, marital_map, loc_map

# ------------------- Train -------------------
def train(X, y, lr=0.05, epochs=1500):
    n = len(X[0])
    weights = [random.uniform(-1, 1) for _ in range(n)]
    bias = 0.0

    for _ in range(epochs):
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
        preds.append(1 if prob >= 0.5 else 0)
    return preds

def accuracy(preds, y):
    return sum(p == a for p, a in zip(preds, y)) / len(y) * 100

# ----------------------- Explaination ----------------------------------------------
def predict_with_explanation(entry, weights, bias, stats, emp_map, purpose_map, marital_map, loc_map):
    raw_features = [
        float(entry[3].strip()),
        float(entry[4].strip()),
        float(entry[5].strip()),
        emp_map.get(entry[6].strip(), 0),
        float(entry[7].strip()),
        float(entry[8].strip()),
        float(entry[9].strip()),
        int(entry[10].strip()),
        1 if entry[11].strip().lower() == "yes" else 0,
        purpose_map.get(entry[12].strip(), 0),
        marital_map.get(entry[13].strip(), 0),
        loc_map.get(entry[14].strip(), 0)
    ]
    norm_features = [
        (val - stats[i][0]) / (stats[i][1] - stats[i][0] + 1e-8)
        for i, val in enumerate(raw_features)
    ]

    z = sum(w * x for w, x in zip(weights, norm_features)) + bias
    prob = sigmoid(z)
    pred = 1 if prob >= 0.5 else 0

    # Feature Contribution
    contrib = {name: abs(w * x) for name, w, x in zip([
        "Age", "Credit Score", "Income", "Employment Type", "Loan Amount", "Tenure",
        "Existing Debt", "Dependents", "Collateral", "Purpose", "Marital Status", "Location"
    ], weights, norm_features)}

    total = sum(contrib.values()) + 1e-8
    explanation = {k: f"{(v / total) * 100:.1f}%" for k, v in contrib.items()}

    result = {
        "Applicant Name": entry[2].strip(),
        "Loan ID": entry[0].strip(),
        "Prediction": "Approved" if pred else "Rejected",
        "Confidence": f"{prob * 100:.2f}%",
        "Explanation": explanation
    }
    return result

# ------------------- Main -------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Loan_model.py <csv_file_path> <loan_id>")
        sys.exit(1)

    filename = sys.argv[1]
    loan_id = sys.argv[2].strip().upper()
    headers, data = read_csv(filename)

    X, y, stats, emp_map, purpose_map, marital_map, loc_map = preprocess(data)

    # Shuffle & Split
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    weights, bias = train(X_train, y_train)

    matched = [row for row in data if row[0].strip().upper() == loan_id]

    if not matched:
        print(json.dumps({"error": f"Loan ID '{loan_id}' not found."}))
        sys.exit(1)
    else:
        result = predict_with_explanation(
            matched[0], weights, bias, stats, emp_map, purpose_map, marital_map, loc_map
        )
        print(json.dumps(result))
