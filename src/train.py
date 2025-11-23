from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import pickle
from preprocess import load_and_preprocess
import joblib
import os

# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data", "german_credit_data.csv")
models_dir = os.path.join(project_root, "models")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

print(f"Loading data from: {data_path}")
X, y = load_and_preprocess(data_path)

print(f"\nDataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n🌳 Training Decision Tree with Gini criterion...")
tree_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_gini.fit(X_train, y_train)
gini_score = tree_gini.score(X_test, y_test)
print(f"✓ Gini Model Accuracy: {gini_score:.4f}")

print("\n🌳 Training Decision Tree with Entropy criterion...")
tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_entropy.fit(X_train, y_train)
entropy_score = tree_entropy.score(X_test, y_test)
print(f"✓ Entropy Model Accuracy: {entropy_score:.4f}")

# Save models
print("\n💾 Saving models...")
pickle.dump(tree_gini, open(os.path.join(models_dir, "tree_gini.pkl"), "wb"))
pickle.dump(tree_entropy, open(os.path.join(models_dir, "tree_entropy.pkl"), "wb"))
joblib.dump(tree_gini, os.path.join(models_dir, "decision_tree.pkl"))

print("✓ Models saved successfully!")

# Display tree structure (limited depth for readability)
print("\n" + "="*60)
print("DECISION TREE USING GINI (first 3 levels)")
print("="*60)
print(export_text(tree_gini, feature_names=list(X.columns), max_depth=3))

print("\n" + "="*60)
print("DECISION TREE USING ENTROPY (first 3 levels)")
print("="*60)
print(export_text(tree_entropy, feature_names=list(X.columns), max_depth=3))

print("\n✅ Training complete!")
print(f"\nModel Summary:")
print(f"  - Gini Depth: {tree_gini.get_depth()}")
print(f"  - Gini Leaves: {tree_gini.get_n_leaves()}")
print(f"  - Entropy Depth: {tree_entropy.get_depth()}")
print(f"  - Entropy Leaves: {tree_entropy.get_n_leaves()}")