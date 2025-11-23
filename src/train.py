from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import pickle
from preprocess import load_and_preprocess
import joblib


X, y = load_and_preprocess("data/german_credit_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_gini.fit(X_train, y_train)

tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_entropy.fit(X_train, y_train)

# Save with pickle
pickle.dump(tree_gini, open("models/tree_gini.pkl", "wb"))
pickle.dump(tree_entropy, open("models/tree_entropy.pkl", "wb"))

# Save with joblib
joblib.dump(tree_gini, "models/decision_tree.pkl")  


print("\n========== DECISION TREE USING GINI ==========\n")
print(export_text(tree_gini, feature_names=list(X.columns)))

print("\n========== DECISION TREE USING ENTROPY ==========\n")
print(export_text(tree_entropy, feature_names=list(X.columns)))
print(export_text(tree_entropy, feature_names=list(X.columns)))
print(export_text(tree_entropy, feature_names=list(X.columns), max_depth=3))