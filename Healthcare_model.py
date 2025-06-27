import csv
import random
from collections import Counter
import json
import sys

# ------------------ Utility Functions ------------------

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0], data[1:]

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

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
        if is_float(row[index]):
            if float(row[index]) < float(value):
                left.append(row)
            else:
                right.append(row)
        else:
            if row[index] == value:
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

def filter_features(data, headers):
    indices_to_keep = [i for i, col in enumerate(headers) if col not in ("Patient_ID", "Patient_Name")]
    new_headers = [headers[i] for i in indices_to_keep]
    new_data = [[row[i] for i in indices_to_keep] for row in data]
    return new_headers, new_data

def predict(tree, row):
    if isinstance(tree, dict):
        index, value = tree['index'], tree['value']
        if is_float(row[index]):
            if float(row[index]) < float(value):
                return predict(tree['left'], row)
            else:
                return predict(tree['right'], row)
        else:
            if row[index] == value:
                return predict(tree['left'], row)
            else:
                return predict(tree['right'], row)
    else:
        return tree

# ------------------ Random Forest & Evaluation ------------------

def subsample(dataset, ratio=1.0):
    return [random.choice(dataset) for _ in range(int(len(dataset) * ratio))]

def random_forest(train, test_row, max_depth=6, min_size=2, n_trees=5):
    trees = [build_tree(subsample(train), max_depth, min_size) for _ in range(n_trees)]
    votes = [predict(tree, test_row) for tree in trees]
    prediction = max(set(votes), key=votes.count)
    confidence = votes.count(prediction) / len(votes)
    return prediction, confidence, trees


# ------------------ Explainability ------------------

def feature_importance(tree, feature_counts):
    if isinstance(tree, dict):
        feature_counts[tree['index']] += 1
        feature_importance(tree['left'], feature_counts)
        feature_importance(tree['right'], feature_counts)

def explain_forest(trees, headers):
    feature_counts = Counter()
    for tree in trees:
        feature_importance(tree, feature_counts)

    total = sum(feature_counts.values())
    explanation = {}

    used_features = set()
    for idx in range(len(headers) - 1):
        feature_name = headers[idx]
        if feature_name in used_features:
            continue
        used_features.add(feature_name)
        perc = (feature_counts[idx] / total * 100) if total else 0
        explanation[feature_name] = f"{perc:.1f}%"
    return explanation

# ------------------ ID/Name Lookup ------------------

def find_row_by_id_or_name(data, id_or_name):
    for row in data:
        if row[0] == id_or_name or row[1].lower() == id_or_name.lower():
            return row
    return None

# ------------------ Main ------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python Healthcare_model.py <csv_file> <Patient_ID>"}))
        sys.exit(1)

    csv_path = sys.argv[1]
    patient_id = sys.argv[2]

    headers, data = read_csv(csv_path)
    raw_data = data.copy()
    headers, filtered_data = filter_features(data, headers)

    row = find_row_by_id_or_name(raw_data, patient_id)
    if not row:
        print(json.dumps({"error": "Patient not found."}))
        sys.exit(1)

    # Prepare data for training
    random.shuffle(raw_data)
    train_data = raw_data[:int(0.8 * len(raw_data))]

    prediction, confidence, trees = random_forest(train_data, row, max_depth=6, min_size=2, n_trees=5)
    explanation = explain_forest(trees, headers)

    output = {
        "Patient Name": row[1],
        "Prediction": prediction,
        "Confidence": f"{confidence * 100:.2f}%",
        "Explanation": explanation
    }

    print(json.dumps(output))
