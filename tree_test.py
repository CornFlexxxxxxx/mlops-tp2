import joblib
import subprocess

# Load Python model
model = joblib.load('tree_model.joblib')

# Test cases
test_cases = [
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
]

print("Python vs C predictions:\n")

# Python predictions
py_preds = []
for features in test_cases:
    pred = model.predict([features])[0]
    py_preds.append(pred)

# Create C test program
c_code = """#include <stdio.h>
#include "tree_model.c"

int main() {
    float tests[][2] = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
    for(int i=0; i<4; i++) {
        printf("%d\\n", predict(tests[i]));
    }
    return 0;
}
"""

with open('test_tree.c', 'w') as f:
    f.write(c_code)

# Compile and run C
subprocess.run(['gcc', 'test_tree.c', '-o', 'test_tree'], check=True)
result = subprocess.run(['./test_tree'], capture_output=True, text=True, check=True)
c_preds = [int(x) for x in result.stdout.strip().split('\n')]

# Compare
for i, features in enumerate(test_cases):
    match = "✓" if py_preds[i] == c_preds[i] else "✗"
    print(f"{features} → Python: {py_preds[i]}, C: {c_preds[i]} {match}")

if py_preds == c_preds:
    print("\n✓ All predictions match!")
else:
    print("\n✗ Predictions differ!")