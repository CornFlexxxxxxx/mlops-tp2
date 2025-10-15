import joblib
import numpy as np
import subprocess
import os

# Load the Python model
model = joblib.load('linear_model.joblib')

# Create test data
test_cases = [
    [100.0, 3.0, 1.0],   # size=100, nb_rooms=3, garden=1
    [150.0, 4.0, 0.0],   # size=150, nb_rooms=4, garden=0
    [200.0, 5.0, 1.0],   # size=200, nb_rooms=5, garden=1
    [80.0, 2.0, 0.0],    # size=80, nb_rooms=2, garden=0
    [175.5, 3.5, 1.0],   # size=175.5, nb_rooms=3.5, garden=1
]

print("=" * 70)
print("LINEAR REGRESSION - Python vs C Comparison")
print("=" * 70)

# Get Python predictions
python_predictions = []
for i, features in enumerate(test_cases):
    pred = model.predict([features])[0]
    python_predictions.append(pred)
    print(f"\nTest case {i+1}: size={features[0]}, nb_rooms={features[1]}, garden={features[2]}")
    print(f"  Python prediction: {pred:.6f}")

# Create C test program
c_test_code = """#include <stdio.h>
#include "linear_model.c"

int main() {
    float test_cases[][3] = {
        {100.0f, 3.0f, 1.0f},
        {150.0f, 4.0f, 0.0f},
        {200.0f, 5.0f, 1.0f},
        {80.0f, 2.0f, 0.0f},
        {175.5f, 3.5f, 1.0f}
    };
    
    int n_cases = 5;
    
    for (int i = 0; i < n_cases; i++) {
        float pred = predict(test_cases[i]);
        printf("%.6f\\n", pred);
    }
    
    return 0;
}
"""

with open('test_linear.c', 'w') as f:
    f.write(c_test_code)

# Compile and run C code
try:
    subprocess.run(['gcc', 'test_linear.c', '-o', 'test_linear', '-lm'], 
                   check=True, capture_output=True)
    result = subprocess.run(['./test_linear'], capture_output=True, text=True, check=True)
    c_predictions = [float(x) for x in result.stdout.strip().split('\n')]
    
    # Compare predictions
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    max_diff = 0
    for i, (py_pred, c_pred) in enumerate(zip(python_predictions, c_predictions)):
        diff = abs(py_pred - c_pred)
        max_diff = max(max_diff, diff)
        print(f"\nTest case {i+1}:")
        print(f"  Python: {py_pred:.6f}")
        print(f"  C:      {c_pred:.6f}")
        print(f"  Diff:   {diff:.6e}")
    
    print("\n" + "=" * 70)
    print(f"Maximum difference: {max_diff:.6e}")
    if max_diff < 1e-5:
        print("✓ Models match closely (difference < 1e-5)")
    elif max_diff < 1e-3:
        print("⚠ Small differences detected (likely due to float vs double precision)")
    else:
        print("✗ Significant differences detected - check implementation")
    print("=" * 70)
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error compiling/running C code: {e}")
    print(f"  stderr: {e.stderr.decode() if e.stderr else 'None'}")
except FileNotFoundError:
    print("\n✗ gcc compiler not found. Please install gcc to run C tests.")
