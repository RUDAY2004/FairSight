import csv
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

def explain_forest(trees, headers):
    feature_counts = Counter()
    for tree in trees:
        feature_importance(tree, feature_counts)

    total = sum(feature_counts.values())
    explanation = {}

    for idx, feature_name in enumerate(headers):  # ✅ no -1 here
        perc = (feature_counts[idx] / total * 100) if total else 0
        explanation[feature_name] = f"{perc:.1f}%"

    return explanation

def find_candidate_row(data, full_data, candidate_id):
    for i, row in enumerate(full_data):
        if row[0].strip() == candidate_id.strip():
            return data[i], full_data[i][1]  # row without ID/Name, plus name
    return None, None

# ------------------ Main ------------------

if __name__ == "__main__":
    # 🔽 Get CSV and Candidate ID
    input_file = sys.argv[1] if len(sys.argv) > 1 else "HR_Resume.csv"
    candidate_id = sys.argv[2] if len(sys.argv) > 2 else None

    if not candidate_id:
        print("No candidate ID provided.")

        sys.exit(1)

    # 🔽 Read and Clean Data
    headers, full_data = read_csv(input_file)
    headers = [h.strip().lstrip('\ufeff') for h in headers]

    # Remove ID and Name columns
    drop_columns = ["CandidateID", "Name"]
    drop_indices = [i for i, h in enumerate(headers) if h in drop_columns]
    headers = [h for i, h in enumerate(headers) if i not in drop_indices]
    data = [[cell for i, cell in enumerate(row) if i not in drop_indices] for row in full_data]

    # Shuffle rows together
    indices = list(range(len(data)))
    random.shuffle(indices)
    data = [data[i] for i in indices]
    full_data = [full_data[i] for i in indices]

    # 🧠 Train Forest
    split_idx = int(0.8 * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]
    actual = [row[-1] for row in test_data]
    predicted, trees = random_forest(train_data, test_data, max_depth=6, min_size=2, n_trees=20)
    acc = accuracy_metric(actual, predicted)

    # 📊 Feature Importance
    explanation_dict = explain_forest(trees, headers)

    # 🔍 Predict for candidate
    row, name = find_candidate_row(data, full_data, candidate_id)
    if row is None:
        print("No candidate ID provided.")

    else:
        votes = [predict(tree, row) for tree in trees]
        prediction = max(set(votes), key=votes.count)
        confidence = votes.count(prediction) / len(votes) * 100

        final_json = {
            "Student Name": name,
            "Prediction": prediction,
            "Confidence": f"{confidence:.2f}%",
            "Explanation": explanation_dict
        }
        sys.stderr.write(f"Name: {name}, Prediction: {prediction}, Confidence: {confidence:.2f}%\n")
        sys.stderr.write(f"Explanation: {json.dumps(explanation_dict)}\n")


        print("DEBUG: Final JSON will be dumped below...", file=sys.stderr)
        print(json.dumps(final_json))


