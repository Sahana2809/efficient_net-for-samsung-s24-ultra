import onnxruntime as ort
import sys

# default model
model_path = "efficientnet_v2_s_int8_static.onnx"

# allow optional command line input
if len(sys.argv) > 1:
    model_path = sys.argv[1]

print(f"\nChecking model: {model_path}\n")

session = ort.InferenceSession(model_path)

print("=== Model Inputs ===")
for i, input_node in enumerate(session.get_inputs()):
    print(f"Input {i}:")
    print(f"  name  = {input_node.name}")
    print(f"  shape = {input_node.shape}")
    print(f"  type  = {input_node.type}")

print("\n=== Model Outputs ===")
for i, output_node in enumerate(session.get_outputs()):
    print(f"Output {i}:")
    print(f"  name  = {output_node.name}")
    print(f"  shape = {output_node.shape}")
    print(f"  type  = {output_node.type}")

print("\nModel loaded successfully.\n")