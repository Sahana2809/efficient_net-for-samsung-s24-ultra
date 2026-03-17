import onnxruntime as ort
import numpy as np
import os
import onnx

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType
)

model_fp32 = "efficientnet_v2_s_fp32.onnx"
model_simplified = "efficientnet_v2_s_simplified.onnx"
model_int8 = "efficientnet_v2_s_int8.onnx"
model_int8_static = "efficientnet_v2_s_int8_static.onnx"


# -------------------------------------------------
# Step 1: Test FP32 inference
# -------------------------------------------------

def run_fp32_inference():
    print("\n=== Step 1: FP32 Inference ===")

    session = ort.InferenceSession(model_fp32)
    input_name = session.get_inputs()[0].name

    dummy = np.random.rand(1,3,224,224).astype(np.float32)

    outputs = session.run(None, {input_name: dummy})

    print("FP32 Output shape:", outputs[0].shape)
    print("FP32 Top score:", outputs[0].max())


# -------------------------------------------------
# Step 2: Simplify ONNX model
# -------------------------------------------------

def simplify_model():

    print("\n=== Step 2: Simplifying ONNX model ===")

    cmd = f"python -m onnxsim {model_fp32} {model_simplified}"

    if os.system(cmd) != 0:
        raise RuntimeError("ONNX simplification failed!")

    print("Simplified model saved:", model_simplified)


# -------------------------------------------------
# Step 3: INT8 Quantization
# -------------------------------------------------

class DummyCalibrationDataReader(CalibrationDataReader):

    def __init__(self, model_path, size=10):

        self.size = size
        self.count = 0

        session = ort.InferenceSession(model_path)
        self.input_name = session.get_inputs()[0].name


    def get_next(self):

        if self.count < self.size:
            self.count += 1

            data = np.random.rand(1,3,224,224).astype(np.float32)

            return {self.input_name: data}

        return None


def quantize_model():

    print("\n=== Step 3: Quantizing model to INT8 ===")

    data_reader = DummyCalibrationDataReader(model_simplified, size=10)

    quantize_static(
        model_input=model_simplified,
        model_output=model_int8,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )

    print("INT8 model saved:", model_int8)


# -------------------------------------------------
# Step 4: Fix dynamic batch size
# -------------------------------------------------

def fix_batch_dimension():

    print("\n=== Step 4: Fixing batch size to 1 ===")

    model = onnx.load(model_int8)

    for input_tensor in model.graph.input:
        input_tensor.type.tensor_type.shape.dim[0].dim_value = 1

    for output_tensor in model.graph.output:
        output_tensor.type.tensor_type.shape.dim[0].dim_value = 1

    onnx.save(model, model_int8_static)

    print("Static batch model saved:", model_int8_static)


# -------------------------------------------------
# Step 5: Validate INT8 inference
# -------------------------------------------------

def run_int8_inference():

    print("\n=== Step 5: Testing INT8 inference ===")

    session = ort.InferenceSession(model_int8_static)

    input_name = session.get_inputs()[0].name

    dummy = np.random.rand(1,3,224,224).astype(np.float32)

    outputs = session.run(None, {input_name: dummy})

    print("INT8 Output shape:", outputs[0].shape)
    print("INT8 Top score:", outputs[0].max())


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

if __name__ == "__main__":

    run_fp32_inference()

    simplify_model()

    quantize_model()

    fix_batch_dimension()

    run_int8_inference()

    print("\nPipeline completed successfully!")