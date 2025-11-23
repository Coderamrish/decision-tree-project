import matplotlib.pyplot as plt
from sklearn import tree
import pickle

# Load both valid models
model_gini = pickle.load(open("../models/tree_gini.pkl", "rb"))
model_entropy = pickle.load(open("../models/tree_entropy.pkl", "rb"))

# Plot Gini tree
plt.figure(figsize=(25, 12))
tree.plot_tree(model_gini, filled=True, fontsize=7)
plt.title("Decision Tree (Gini)")
plt.show()

# Plot Entropy tree
plt.figure(figsize=(25, 12))
tree.plot_tree(model_entropy, filled=True, fontsize=7)
plt.title("Decision Tree (Entropy)")
plt.show()
