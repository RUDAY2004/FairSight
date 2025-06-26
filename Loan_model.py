import math
import random
import csv
import json
# ------------------- Sigmoid -------------------
def sigmoid(z):
    return 1.0 if z > 100 else 0.0 if z < -100 else 1 / (1 + math.exp(-z))

# ------------------- Read CSV -------------------
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)
    return headers, data

# ------------------- Normalize -------------------
def normalize_column(col):
    min_val, max_val = min(col), max(col)
    return [(x - min_val) / (max_val - min_val + 1e-8) for x in col], (min_val, max_val)

# ------------------- Preprocess -------------------
def preprocess(data):
    X_raw, y = [], []
    emps = list(set(row[6] for row in data))
    purposes = list(set(row[12] for row in data))
    locs = list(set(row[14] for row in data))
    marital = list(set(row[13] for row in data))

    emp_map = {e: i for i, e in enumerate(emps)}
    purpose_map = {p: i for i, p in enumerate(purposes)}
    loc_map = {l: i for i, l in enumerate(locs)}
    marital_map = {m: i for i, m in enumerate(marital)}

    for row in data:
        features = []
        features.append(float(row[3]))  # Age
        features.append(float(row[4]))  # Credit Score
        features.append(float(row[5]))  # Income
        features.append(emp_map[row[6]])  # Employment Type
        features.append(float(row[7]))  # Loan Amount
        features.append(float(row[8]))  # Loan Tenure
        features.append(float(row[9]))  # Existing Debt
        features.append(int(row[10]))  # Num Dependents
        features.append(1 if row[11].lower() == "yes" else 0)  # Collateral
        features.append(purpose_map[row[12]])  # Purpose
        features.append(marital_map[row[13]])  # Marital Status
        features.append(loc_map[row[14]])  # Location

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

# ------------------- Predict for New Entry -------------------
def predict_new(entry, weights, bias, stats, emp_map, purpose_map, marital_map, loc_map):
    features = [
        float(entry[3]),  # Age
        float(entry[4]),  # Credit Score
        float(entry[5]),  # Income
        emp_map.get(entry[6], 0),
        float(entry[7]),  # Loan Amt
        float(entry[8]),  # Tenure
        float(entry[9]),  # Existing debt
        int(entry[10]),
        1 if entry[11].lower() == "yes" else 0,
        purpose_map.get(entry[12], 0),
        marital_map.get(entry[13], 0),
        loc_map.get(entry[14], 0),
    ]

    # Normalize using stats
    for i in range(len(features)):
        min_val, max_val = stats[i]
        features[i] = (features[i] - min_val) / (max_val - min_val + 1e-8)

    z = sum(w * x for w, x in zip(weights, features)) + bias
    prob = sigmoid(z)
    return 1 if prob >= 0.5 else 0, prob

# ----------------------- Explaination ----------------------------------------------
def predict_with_explanation(entry, weights, bias, stats, emp_map, purpose_map, marital_map, loc_map):
    raw_features = [
        float(entry[3]), float(entry[4]), float(entry[5]), emp_map.get(entry[6], 0),
        float(entry[7]), float(entry[8]), float(entry[9]), int(entry[10]),
        1 if entry[11].lower() == "yes" else 0,
        purpose_map.get(entry[12], 0), marital_map.get(entry[13], 0), loc_map.get(entry[14], 0)
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
        "Applicant Name": entry[2],
        "Loan ID": entry[0],
        "Prediction": "Approved" if pred else "Rejected",
        "Confidence": f"{prob * 100:.2f}%",
        "Explanation": explanation
    }
    return result

# ------------------- Main -------------------
if __name__ == "__main__":
    headers, data = read_csv(r"C:\Users\madir\OneDrive\Desktop\SAPHACK\csvfiles\csvfiles\Loandata.csv")  # Update path
    X, y, stats, emp_map, purpose_map, marital_map, loc_map = preprocess(data)

    # Shuffle & Split
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    weights, bias = train(X_train, y_train)

    train_acc = accuracy(predict(X_train, weights, bias), y_train)
    test_acc = accuracy(predict(X_test, weights, bias), y_test)

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%")

    # Example new input (row format)
    #new_user = ["LN1100", "ABCXY1234Z", "Amit Rawat", "35", "720", "50000", "Salaried", "200000", "36", "10000", "2", "Yes", "Medical", "Married", "Urban", ""]
    #pred, prob = predict_new(new_user, weights, bias, stats, emp_map, purpose_map, marital_map, loc_map)
    #print("\nPrediction for new loan application:")
    #print(f"  Loan Status: {'Approved' if pred else 'Rejected'} (Confidence: {prob*100:.2f}%)")

    user_input = input("\nEnter Loan ID: ").strip().upper()
    matched = [row for row in data if row[0].strip().upper() == user_input]

    if not matched:
        print("Loan ID not found.")
    else:
        result = predict_with_explanation(
            matched[0], weights, bias, stats, emp_map, purpose_map, marital_map, loc_map
        )
        print("\nFinal JSON Output:")
        print(json.dumps(result, indent=2))
