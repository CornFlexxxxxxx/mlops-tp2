import joblib
import sys

def transpile_node(tree, node_id):
    if tree.feature[node_id] == -2:  # Leaf
        return f"return {int(tree.value[node_id][0].argmax())};"
    
    feat = tree.feature[node_id]
    thresh = tree.threshold[node_id]
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    
    return f"if(features[{feat}]<={thresh}){{{transpile_node(tree, left)}}}else{{{transpile_node(tree, right)}}}"

def transpile(model_path, output='tree_model.c'):
    model = joblib.load(model_path)
    tree = model.tree_
    
    code = f"""#include <stdio.h>
int predict(float *features){{
{transpile_node(tree, 0)}
}}
"""
    
    with open(output, 'w') as f:
        f.write(code)
    
    print(f"Transpiled to {output}")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'tree_model.joblib'
    output = sys.argv[2] if len(sys.argv) > 2 else 'tree_model.c'
    transpile(model_path, output)