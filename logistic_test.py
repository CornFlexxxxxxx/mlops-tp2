import joblib
import numpy as np
import subprocess
import os

# Load the Python model
model = joblib.load('logistic_model.joblib')

# Create test data (size, nb_rooms, price)
test_cases = [
    [100.0, 3.0, 250000.0],   # size=100, nb_rooms=3, price=250k
    [150.0, 4.0, 400000.0],   # size=150, nb_rooms=4, price=400k
    [200.0, 5.0, 500000.0],   # size=200, nb_rooms=5, price=500k
    [80.0, 2.0, 180000.0],    # size=80, nb_rooms=2, price=180k
    [175.5, 3.5, 350000.0],   # size=175.5, nb_rooms=3.5, price=350k
]

print("=" * 70)
print("LOGISTIC REGRESSION - Python vs C Comparison")
print("=" * 70)

# Get Python predictions (both probability and class)
python_predictions = []
python_probabilities = []
for i, features in enumerate(test_cases):
    prob = model.predict_proba([features])[0]
    pred_class = model.predict([features])[0]
    python_predictions.append(pred_class)
    python_probabilities.append(prob[1])  # probability of class 1
    print(f"\nTest case {i+1}: size={features[0]}, nb_rooms={features[1]}, price={features[2]}")
    print(f"  Python probability (has garden): {prob[1]:.6f}")
    print(f"  Python prediction: {pred_class}")

# Create C test program
c_test_code = """#include <stdio.h>
#include "logistic_model.c"

int main() {
    float test_cases[][3] = {
        {100.0f, 3.0f, 250000.0f},
        {150.0f, 4.0f, 400000.0f},
        {200.0f, 5.0f, 500000.0f},
        {80.0f, 2.0f, 180000.0f},
        {175.5f, 3.5f, 350000.0f}
    };
    
    int n_cases = 5;
    
    for (int i = 0; i < n_cases; i++) {
        // Calculate probability manually for comparison
        float z = """ + str(model.intercept_[0]) + """f;
        float coefs[] = {""" + ', '.join([f'{c}f' for c in model.coef_[0]]) + """};
        for (int j = 0; j < 3; j++) z += coefs[j] * test_cases[i][j];
        float prob = sigmoid(z);
        
        int pred_class = predict(test_cases[i]);
        printf("%.6f %d\\n", prob, pred_class);
    }
    
    return 0;
}
"""

with open('test_logistic.c', 'w') as f:
    f.write(c_test_code)

# Compile and run C code
try:
    subprocess.run(['gcc', 'test_logistic.c', '-o', 'test_logistic', '-lm'], 
                   check=True, capture_output=True)
    result = subprocess.run(['./test_logistic'], capture_output=True, text=True, check=True)
    
    c_results = []
    for line in result.stdout.strip().split('\n'):
        prob, pred = line.split()
        c_results.append((float(prob), int(pred)))
    
    # Compare predictions
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    max_prob_diff = 0
    all_classes_match = True
    
    for i, ((c_prob, c_class), py_class, py_prob) in enumerate(zip(c_results, python_predictions, python_probabilities)):
        prob_diff = abs(py_prob - c_prob)
        max_prob_diff = max(max_prob_diff, prob_diff)
        class_match = (c_class == py_class)
        all_classes_match = all_classes_match and class_match
        
        print(f"\nTest case {i+1}:")
        print(f"  Python probability: {py_prob:.6f}, class: {py_class}")
        print(f"  C probability:      {c_prob:.6f}, class: {c_class}")
        print(f"  Prob diff:          {prob_diff:.6e}")
        print(f"  Classes match:      {'✓' if class_match else '✗'}")
    
    print("\n" + "=" * 70)
    print(f"Maximum probability difference: {max_prob_diff:.6e}")
    print(f"All classes match: {'✓ Yes' if all_classes_match else '✗ No'}")
    
    if max_prob_diff < 1e-5 and all_classes_match:
        print("✓ Models match closely (difference < 1e-5)")
    elif max_prob_diff < 1e-3 and all_classes_match:
        print("⚠ Small differences detected (likely due to float vs double precision)")
    else:
        print("✗ Significant differences detected - check implementation")
    print("=" * 70)
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error compiling/running C code: {e}")
    print(f"  stderr: {e.stderr.decode() if e.stderr else 'None'}")
except FileNotFoundError:
    print("\n✗ gcc compiler not found. Please install gcc to run C tests.")
