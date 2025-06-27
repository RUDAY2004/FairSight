import csv
import math
import random
from collections import Counter
import json
import sys

# ------------------ Utility Functions ------------------

def read_csv(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0], data[1:]

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def one_hot(val, mapping):
    vec = [0.0] * len(mapping)
    vec[mapping[val]] = 1.0
    return vec

def normalize_column(col):
    min_val = min(col)
    max_val = max(col)
    return [(x - min_val) / (max_val - min_val + 1e-8) for x in col], min_val, max_val

# ------------------ Preprocessing ------------------

def preprocess(data):
    genders = list(set(r[3] for r in data))
    locations = list(set(r[4] for r in data))
    categories = list(set(r[8] for r in data))

    gender_map = {g: i for i, g in enumerate(genders)}
    location_map = {l: i for i, l in enumerate(locations)}
    category_map = {c: i for i, c in enumerate(categories)}

    def remap(a):
        return "Engagement" if a in ["Cashback", "Bundle Offers", "Loyalty Points"] else "Acquisition"

    actions = list(set(remap(r[10]) for r in data))
    action_map = {a: i for i, a in enumerate(actions)}
    inv_action_map = {i: a for a, i in action_map.items()}

    X_raw, y = [], []
    num_cols = [[] for _ in range(5)]
    for row in data:
        nums = [float(row[2]), float(row[5]), float(row[6]), float(row[7]), float(row[9])]
        for i in range(5):
            num_cols[i].append(nums[i])
        X_raw.append((row, nums))
        y.append(remap(row[10]))

    norm_cols, col_stats = [], []
    for col in num_cols:
        norm, mn, mx = normalize_column(col)
        norm_cols.append(norm)
        col_stats.append((mn, mx))

    X = []
    feature_names = []
    for g in genders:
        feature_names.append(f"Gender_{g}")
    for l in locations:
        feature_names.append(f"Location_{l}")
    for c in categories:
        feature_names.append(f"Category_{c}")
    feature_names.extend(["Age", "Days_Since_Last_Purchase", "Purchase_Frequency", 
                         "Avg_Purchase_Amount", "Total_Spend", 
                         "Freq_Avg_Spend", "Days_Freq", "Total_Avg"])

    for idx, (row, _) in enumerate(X_raw):
        f = []
        f += one_hot(row[3], gender_map)
        f += one_hot(row[4], location_map)
        f += one_hot(row[8], category_map)
        for i in range(5):
            f.append(norm_cols[i][idx])
        f.append(norm_cols[2][idx] * norm_cols[3][idx])  # freq * avg spend
        f.append(norm_cols[1][idx] * norm_cols[2][idx])  # days * freq
        f.append(norm_cols[4][idx] * norm_cols[3][idx])  # total * avg
        X.append(f)

    return X, y, gender_map, location_map, category_map, action_map, inv_action_map, col_stats, feature_names

# ------------------ Gini, Splits, Tree Building ------------------

def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini += (1 - score) * (size / n_instances)
    return gini

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if float(row[index]) < float(value):
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split_node(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split_node(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split_node(node['right'], max_depth, min_size, depth + 1)

def build_tree(train, max_depth=5, min_size=2):
    root = get_split(train)
    split_node(root, max_depth, min_size, 1)
    return root

def predict(tree, row):
    if isinstance(tree, dict):
        index, value = tree['index'], tree['value']
        if float(row[index]) < float(value):
            return predict(tree['left'], row)
        else:
            return predict(tree['right'], row)
    else:
        return tree

# ------------------ Random Forest & Evaluation ------------------

def subsample(dataset, ratio=1.0):
    return [random.choice(dataset) for _ in range(int(len(dataset) * ratio))]

def random_forest(train, test, max_depth, min_size, n_trees):
    trees = [build_tree(subsample(train), max_depth, min_size) for _ in range(n_trees)]
    predictions = []
    for row in test:
        votes = [predict(tree, row) for tree in trees]
        predictions.append(max(set(votes), key=votes.count))
    return predictions, trees

def accuracy_metric(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual) * 100

# ------------------ Explainability ------------------

def feature_importance(tree, feature_counts):
    if isinstance(tree, dict):
        feature_counts[tree['index']] += 1
        feature_importance(tree['left'], feature_counts)
        feature_importance(tree['right'], feature_counts)

def explain_forest(trees, feature_names):
    feature_counts = Counter()
    for tree in trees:
        feature_importance(tree, feature_counts)
    total = sum(feature_counts.values())
    explanation = {}
    for idx, feature_name in enumerate(feature_names):
        perc = (feature_counts[idx] / total * 100) if total else 0
        explanation[feature_name] = f"{perc:.1f}%"
    return explanation

# ------------------ Predict on New ------------------

def find_customer_row(data, full_data, customer_id):
    for i, row in enumerate(full_data):
        if row[0].strip() == customer_id.strip():
            return data[i], full_data[i][1]  # row without ID/Name, plus name
    return None, None

def predict_new(customer_id, full_data, gender_map, location_map, category_map, col_stats, feature_names, trees):
    X, _, _, _, _, _, inv_action_map, _, _ = preprocess(full_data)
    row, name = find_customer_row(X, full_data, customer_id)
    if row is None:
        return None, None, None
    votes = [predict(tree, row) for tree in trees]
    prediction = max(set(votes), key=votes.count)
    confidence = votes.count(prediction) / len(votes) * 100
    return name, prediction, confidence

# ------------------ Main ------------------

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "Retail.csv"
    customer_id = sys.argv[2] if len(sys.argv) > 2 else None

    if not customer_id:
        print("No customer ID provided.", file=sys.stderr)
        sys.exit(1)

    headers, full_data = read_csv(input_file)
    headers = [h.strip().lstrip('\ufeff') for h in headers]
    drop_columns = ["CustomerID", "Name"]
    drop_indices = [i for i, h in enumerate(headers) if h in drop_columns]
    headers = [h for i, h in enumerate(headers) if i not in drop_indices]
    data = [[cell for i, cell in enumerate(row) if i not in drop_indices] for row in full_data]

    X, y, gender_map, location_map, category_map, action_map, inv_action_map, col_stats, feature_names = preprocess(full_data)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split_idx = int(0.8 * len(X))
    train_data, test_data = X[:split_idx], X[split_idx:]
    actual = y[split_idx:]
    predicted, trees = random_forest(train_data, test_data, max_depth=6, min_size=2, n_trees=20)
    acc = accuracy_metric(actual, predicted)
    explanation_dict = explain_forest(trees, feature_names)

    name, prediction, confidence = predict_new(customer_id, full_data, gender_map, location_map, category_map, col_stats, feature_names, trees)
    if name is None:
        print("Customer ID not found.", file=sys.stderr)
        sys.exit(1)

    final_json = {
        "Name": name,
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%",
        "Explanation": explanation_dict
    }
    sys.stderr.write(f"Name: {name}, Prediction: {prediction}, Confidence: {confidence:.2f}%\n")
    sys.stderr.write(f"Explanation: {json.dumps(explanation_dict)}\n")
    print("DEBUG: Final JSON will be dumped below...", file=sys.stderr)
    print(json.dumps(final_json))
