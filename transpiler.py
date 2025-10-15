import joblib
import sys

def transpile(model_path):
    model = joblib.load(model_path)
    model_type = type(model).__name__
    output = ""
    
    if model_type == 'LinearRegression':
        output = 'linear_model.c'
        intercept = model.intercept_
        coefs = model.coef_
        n_features = len(coefs)
        
        code = f"""#include <stdio.h>

float predict(float *x) {{
    float result = {intercept}f;
    float coefs[] = {{{', '.join([f'{c}f' for c in coefs])}}};
    for (int i = 0; i < {n_features}; i++) result += coefs[i] * x[i];
    return result;
}}
"""
        
    elif model_type == 'LogisticRegression':
        output = 'logistic_model.c'
        intercept = model.intercept_[0]
        coefs = model.coef_[0]
        n_features = len(coefs)
        
        code = f"""#include <stdio.h>
#include <math.h>

float sigmoid(float x) {{
    return 1.0f / (1.0f + exp(-x));
}}

int predict(float *x) {{
    float z = {intercept}f;
    float coefs[] = {{{', '.join([f'{c}f' for c in coefs])}}};
    for (int i = 0; i < {n_features}; i++) z += coefs[i] * x[i];
    return sigmoid(z) >= 0.5f ? 1 : 0;
}}
"""
    
    with open(output, 'w') as f:
        f.write(code)
    
    print(f"Transpiled {model_type} to {output}")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model.joblib'
    transpile(model_path)
