from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib

# Simple training data
X = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])
y = np.array([1, 0, 0, 0])

# Train decision tree
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

print(f"Train accuracy: {model.score(X, y):.4f}")

# Save model
joblib.dump(model, 'tree_model.joblib')
print("Model saved to tree_model.joblib")