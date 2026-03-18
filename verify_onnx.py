import onnx

def check_model(path):
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"Model {path} is valid.")
        
        # Check inputs
        for input in model.graph.input:
            print(f"Input: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
            
        # Check outputs
        for output in model.graph.output:
            print(f"Output: {output.name}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
            
    except Exception as e:
        print(f"Error checking model: {e}")

if __name__ == "__main__":
    check_model("/Users/sahanaagadi/edge_ai_project/efficient_net/efficientnet_v2_s_int8.onnx")
