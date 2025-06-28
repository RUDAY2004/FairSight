import csv
import random
from collections import Counter
import json
import sys
import os

# ------------------ Utility Functions ------------------

def read_csv(filename):
    """
    Reads data from a CSV file.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing headers (list of str) and data (list of lists of str).
    """
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0], data[1:]

def is_float(s):
    """
    Checks if a string can be converted to a float.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is a float, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

# ------------------ Gini, Splits, Tree Building ------------------

def gini_index(groups, classes):
    """
    Calculates the Gini impurity for a set of groups.

    Args:
        groups (list): A list of groups, where each group is a list of data rows.
        classes (list): A list of all possible class labels.

    Returns:
        float: The calculated Gini impurity.
    """
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        # Get the class labels (last element of each row) for the current group
        labels = [row[-1] for row in group]
        for class_val in classes:
            # Proportion of instances with this class value in the group
            p = labels.count(class_val) / size
            score += p * p
        # Weighted average of (1 - sum(p^2)) for each group
        gini += (1 - score) * (size / n_instances)
    return gini

def test_split(index, value, dataset):
    """
    Splits a dataset into two groups based on a feature and its value.
    Handles both numerical (less than/greater than or equal to)
    and categorical (equal to/not equal to) splits.

    Args:
        index (int): The index of the feature to split on.
        value (str): The value to use for splitting.
        dataset (list): The dataset to split.

    Returns:
        tuple: A tuple containing two lists: left group and right group.
    """
    left, right = [], []
    for row in dataset:
        if is_float(row[index]):
            # Numerical split
            if float(row[index]) < float(value):
                left.append(row)
            else:
                right.append(row)
        else:
            # Categorical split
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
    return left, right

def get_split(dataset):
    """
    Finds the best split point (feature and value) for a dataset
    based on Gini impurity.

    Args:
        dataset (list): The dataset to find the best split for.

    Returns:
        dict: A dictionary containing the best split's index, value, and groups.
    """
    class_values = list(set(row[-1] for row in dataset)) # Get unique class labels
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    # Iterate through each feature (column)
    for index in range(len(dataset[0]) - 1): # Exclude the last column (target)
        # Iterate through each unique value in that feature as a potential split point
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def to_terminal(group):
    """
    Determines the most common class label in a group, used as the prediction
    for a leaf node.

    Args:
        group (list): A list of data rows in a leaf node.

    Returns:
        str: The most frequent class label in the group.
    """
    outcomes = [row[-1] for row in group]
    # Handle empty outcomes list to prevent ValueError for max()
    if not outcomes:
        return None
    return max(set(outcomes), key=outcomes.count)

def split_node(node, max_depth, min_size, depth):
    """
    Recursively splits a node in the decision tree.

    Args:
        node (dict): The current node to split.
        max_depth (int): The maximum allowed depth of the tree.
        min_size (int): The minimum number of samples required to split a node.
        depth (int): The current depth of the node.
    """
    left, right = node['groups']
    del(node['groups']) # No longer needed after splitting

    # Check for no split (one or both groups are empty)
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # Check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # Process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split_node(node['left'], max_depth, min_size, depth + 1)

    # Process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split_node(node['right'], max_depth, min_size, depth + 1)

def build_tree(train, max_depth=5, min_size=2):
    """
    Builds a decision tree from the training data.

    Args:
        train (list): The training dataset.
        max_depth (int): The maximum depth of the tree.
        min_size (int): The minimum size of a node to split.

    Returns:
        dict: The root node of the constructed decision tree.
    """
    root = get_split(train)
    split_node(root, max_depth, min_size, 1)
    return root

def predict(tree, row):
    """
    Makes a prediction for a single data row using a trained decision tree.

    Args:
        tree (dict or str): The decision tree (root node) or a terminal value.
        row (list): The data row to predict.

    Returns:
        str: The predicted class label.
    """
    if isinstance(tree, dict):
        index, value = tree['index'], tree['value']
        if is_float(row[index]):
            # Numerical comparison
            if float(row[index]) < float(value):
                return predict(tree['left'], row)
            else:
                return predict(tree['right'], row)
        else:
            # Categorical comparison
            if row[index] == value:
                return predict(tree['left'], row)
            else:
                return predict(tree['right'], row)
    else:
        # If it's a terminal node (leaf), return its value
        return tree

# ------------------ Random Forest & Evaluation ------------------

def subsample(dataset, ratio=1.0):
    """
    Creates a bootstrap sample (with replacement) from a dataset.

    Args:
        dataset (list): The original dataset.
        ratio (float): The ratio of the original dataset size to sample.

    Returns:
        list: The subsampled dataset.
    """
    # Ensure ratio is within valid range [0, 1]
    ratio = max(0.0, min(1.0, ratio))
    sample_size = int(len(dataset) * ratio)
    if sample_size == 0 and len(dataset) > 0: # Ensure at least one sample if dataset is not empty
        sample_size = 1
    return [random.choice(dataset) for _ in range(sample_size)]

def random_forest(train, test, max_depth, min_size, n_trees):
    """
    Trains and makes predictions using a Random Forest model.

    Args:
        train (list): The training dataset.
        test (list): The test dataset.
        max_depth (int): Max depth for individual trees.
        min_size (int): Min size for nodes for individual trees.
        n_trees (int): Number of trees in the forest.

    Returns:
        tuple: A tuple containing predicted labels (list of str) and
               the list of trained decision trees.
    """
    trees = []
    # Build each tree in the forest
    for _ in range(n_trees):
        # Create a subsample for each tree
        sample = subsample(train)
        if not sample: # Skip if subsample is empty
            continue
        try:
            tree = build_tree(sample, max_depth, min_size)
            trees.append(tree)
        except Exception as e:
            # sys.stderr.write(f"Error building tree: {e}\n") # Commented out
            continue # Skip this tree if building fails

    predictions = []
    # Make predictions for each row in the test set
    for row in test:
        votes = []
        # Get predictions from all trees
        for tree in trees:
            pred = predict(tree, row)
            if pred is not None: # Ensure a valid prediction is returned
                votes.append(pred)
        
        # If no votes, return a default or error
        if not votes:
            predictions.append("Unknown") # Or handle this as an error
        else:
            # Predict the class with the majority vote
            predictions.append(max(set(votes), key=votes.count))
    return predictions, trees

def accuracy_metric(actual, predicted):
    """
    Calculates the accuracy of predictions.

    Args:
        actual (list): The true class labels.
        predicted (list): The predicted class labels.

    Returns:
        float: The accuracy as a percentage.
    """
    if not actual:
        return 0.0
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual) * 100

# ------------------ Explainability ------------------

def feature_importance(tree, feature_counts):
    """
    Recursively counts how many times each feature is used for splitting
    across a decision tree.

    Args:
        tree (dict or str): The current node or terminal value of the tree.
        feature_counts (Counter): A Counter object to store feature usage.
    """
    if isinstance(tree, dict) and 'index' in tree:
        feature_counts[tree['index']] += 1 # Increment count for the feature used at this split
        if 'left' in tree:
            feature_importance(tree['left'], feature_counts)
        if 'right' in tree:
            feature_importance(tree['right'], feature_counts)

def explain_forest(trees, headers):
    """
    Calculates feature importance across all trees in the forest.

    Args:
        trees (list): A list of trained decision trees.
        headers (list): The list of feature names (headers).

    Returns:
        dict: A dictionary mapping feature names to their importance percentages.
    """
    feature_counts = Counter()
    for tree in trees:
        feature_importance(tree, feature_counts)

    total = sum(feature_counts.values())
    explanation = {}

    for idx, feature_name in enumerate(headers):
        # Calculate percentage of total splits attributed to this feature
        perc = (feature_counts[idx] / total * 100) if total else 0
        explanation[feature_name] = f"{perc:.1f}%"

    return explanation

def find_customer_row(full_data_rows, customer_id, id_col_idx=0, name_col_idx=1):
    """
    Finds a customer's original data row and name from the full dataset
    based on their ID.

    Args:
        full_data_rows (list): The complete raw dataset including ID and Name.
        customer_id (str): The ID of the customer to find.
        id_col_idx (int): The column index for the customer ID.
        name_col_idx (int): The column index for the customer Name.

    Returns:
        tuple: A tuple containing the found row (list) and customer name (str),
               or (None, None) if not found.
    """
    for row in full_data_rows:
        if len(row) > id_col_idx and row[id_col_idx].strip() == customer_id.strip():
            customer_name = row[name_col_idx] if len(row) > name_col_idx else "N/A"
            return row, customer_name
    return None, None

# ------------------ Main ------------------

if __name__ == "__main__":
    # Get CSV file and Customer ID from command-line arguments, or exit if not provided
    if len(sys.argv) < 3:
        # Print a JSON error to stdout to be consistent with success output
        error_output = {"error": "Usage: python retail_model.py <csv_file_path> <customer_id>"}
        print(json.dumps(error_output))
        sys.exit(1)

    input_file = sys.argv[1]
    customer_id = sys.argv[2]

    # Ensure the input CSV file exists
    if not os.path.exists(input_file):
        error_output = {"error": f"Input file '{input_file}' not found."}
        print(json.dumps(error_output))
        sys.exit(1)

    # Read the full dataset including headers
    original_headers, full_data = read_csv(input_file)
    original_headers = [h.strip().lstrip('\ufeff') for h in original_headers] # Clean headers

    # Dynamically find the indices for 'CustomerID' and 'Name'
    customer_id_col_name = "CustomerID"
    customer_name_col_name = "Name"
    
    id_col_idx = -1
    name_col_idx = -1
    
    # Identify indices of columns to keep for the model training
    model_column_indices = []
    processed_headers = [] # Headers for the features used in the model

    for i, h in enumerate(original_headers):
        if h == customer_id_col_name:
            id_col_idx = i
        elif h == customer_name_col_name:
            name_col_idx = i
        else:
            processed_headers.append(h)
            model_column_indices.append(i)

    # Handle cases where ID or Name columns might be missing or not at expected positions
    if id_col_idx == -1:
        # Fallback: Assume first column is CustomerID if not explicitly named
        # sys.stderr.write(f"Warning: '{customer_id_col_name}' column not found. Assuming index 0 is CustomerID.\n") # Commented out
        id_col_idx = 0
    if name_col_idx == -1:
        # Fallback: Assume second column is Name if not explicitly named
        # sys.stderr.write(f"Warning: '{customer_name_col_name}' column not found. Assuming index 1 is Name.\n") # Commented out
        name_col_idx = 1

    # Create a new 'data' list where ID and Name columns are removed from each row
    data = [[row[i] for i in model_column_indices] for row in full_data]

    # Check if data is empty after processing
    if not data:
        error_output = {"error": "No valid data rows found after preprocessing. Check CSV content."}
        print(json.dumps(error_output))
        sys.exit(1)

    # Shuffle the processed 'data'
    random.shuffle(data)

    # Split the 'data' into training and testing sets (80/20 split)
    split_idx = int(0.8 * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]

    if not train_data:
        error_output = {"error": "Training data is empty. Cannot build model. Ensure sufficient data."}
        print(json.dumps(error_output))
        sys.exit(1)
    
    actual = []
    if test_data:
        actual = [row[-1] for row in test_data]
    # else:
        # sys.stderr.write("Warning: Test data is empty. Accuracy cannot be calculated.\n") # Commented out

    # Train the Random Forest model and get predictions for the test set
    predicted, trees = random_forest(train_data, test_data, max_depth=6, min_size=2, n_trees=20)

    # Calculate the model's accuracy on the test set
    acc = accuracy_metric(actual, predicted)
    # sys.stderr.write(f"Model Accuracy: {acc:.2f}%\n") # Commented out

    # Get feature importance explanation from the trained forest
    explanation_dict = explain_forest(trees, processed_headers)

    # Find the specific customer's original data row and name
    customer_raw_row, customer_name = find_customer_row(read_csv(input_file)[1], customer_id, id_col_idx, name_col_idx)

    if customer_raw_row is None:
        error_output = {"error": f"Customer ID '{customer_id}' not found in the dataset. Please check the ID."}
        print(json.dumps(error_output))
        sys.exit(1)
    else:
        # Prepare the customer's row for prediction
        customer_prediction_row = [customer_raw_row[i] for i in model_column_indices]

        # Get predictions from all trees for the specific customer
        votes = [predict(tree, customer_prediction_row) for tree in trees if tree is not None]
        
        prediction = "Unknown"
        confidence = 0.0

        if votes:
            prediction = max(set(votes), key=votes.count)
            confidence = votes.count(prediction) / len(votes) * 100

        # Construct the final JSON output
        final_json = {
            "Name": customer_name,
            "Prediction": prediction,
            "Confidence": f"{confidence:.2f}%",
            "Explanation": explanation_dict
        }
        
        # All debug prints to stderr have been removed to ensure clean JSON output

        # Print the final JSON to stdout
        print(json.dumps(final_json, indent=2))
