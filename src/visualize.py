import matplotlib.pyplot as plt
from sklearn import tree
import pickle

model = pickle.load(open("models/tree_gini.pkl", "rb"))
model = pickle.load(open("models/decision_tree.pkl", "rb"))
model = pickle.load(open("models/tree_entropy.pkl", "rb"))
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, fontsize=8)
plt.show()
