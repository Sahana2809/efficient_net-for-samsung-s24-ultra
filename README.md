# EfficientNetV2-S Edge AI Benchmark: Samsung S24 Ultra

This repository provides a high-performance **Edge AI Benchmarking Application** optimized for the **Samsung Galaxy S24 Ultra** (Snapdragon 8 Gen 3). It demonstrates the deployment and performance measurement of an **INT8 Quantized EfficientNet-V2-S** model across different hardware accelerators.

## 🚀 Key Features

*   **Multi-Backend Support**: Toggle between **CPU (XNNPACK)**, **NNAPI (GPU)**, and **QNN (NPU HTP v75)** in real-time.
*   **Dual Model Modes**: Support for **Live ONNX** inference and **Compiled QNN Context Binary (.bin)** for maximum NPU performance.
*   **Snapdragon 8 Gen 3 Optimized**: Explicitly targets **HTP Architecture v75** with `high_performance` clock profiles.
*   **Professional Metrics Suite**:
    *   **Latency Breakdown**: Stage-wise tracking for Pre-processing, Inference, and Post-processing.
    *   **Throughput**: Real-time **TOPS** (Tera Operations Per Second) and **FPS** calculation.
    *   **Accuracy Tracking**: Dataset-wide Top-1 and Top-5 accuracy benchmarking on a Tiny ImageNet subset.
    *   **Reliability**: Min/Max latency tracking to measure performance jitter and stability.
*   **Robust Architecture**: Background session management and atomic locking to prevent crashes during hardware switching.

## 📱 Hardware Configuration

*   **Device**: Samsung Galaxy S24 Ultra (SM-S928B)
*   **Chipset**: Snapdragon 8 Gen 3 for Galaxy
*   **TPU/NPU**: Hexagon HTP v75
*   **Quantization**: INT8 QDQ (Symmetric Weight, Asymmetric Activation)

## 📁 Project Structure

*   **/android_app**: Full Android Studio project (Kotlin 1.9, Gradle 8.9).
    *   `ImageClassifier.kt`: Core ORT integration with QNN/NNAPI/CPU support.
    *   `MainActivity.kt`: Real-time benchmarking UI and dataset evaluation logic.
    *   `src/main/jniLibs`: Optimized Qualcomm QNN HTP backend libraries.
*   **/pipeline.py**: Python pipeline for model simplification and INT8 quantization.
*   **/images**: Evaluation dataset (30 images subset of Tiny ImageNet validation).

## 🛠️ Getting Started

### Prerequisites
*   Android Studio Ladybug (or newer)
*   Gradle 8.9+
*   Physical Samsung S24 Ultra (Emulator does not support NPU/QNN)

### Setup & Run
1.  Clone the repository and open `/android_app` in Android Studio.
2.  Ensure your device is in **Developer Mode** with USB Debugging enabled.
3.  Click **Run** to deploy the `onnxcamera` app.
4.  **Accept Camera Permissions** when prompted.

## 📊 Benchmarking Workflow

1.  **Select Backend**: Use the radio buttons at the bottom to switch hardware (CPU → NNAPI → NPU).
2.  **Toggle Model Source**: Use the ONNX/QNN BIN switch to compare live model execution vs. pre-compiled context binaries.
3.  **Real-Time Analysis**: Click "Next Image" to see single-frame latency breakdown and TOPS.
4.  **Full Data Report**: Click **"Benchmark All"** to run the model against the entire 30-image dataset and generate a final report including:
    *   **Average Latency & FPS**
    *   **Top-1 & Top-5 Accuracy**
    *   **Throughput (TOPS)**
    *   **Jitter (Min/Max Latency)**

## 🛡️ License & Acknowledgements
Built on **ONNX Runtime 1.24.3** with **Qualcomm QNN SDK**. Evaluation uses images from the **Tiny ImageNet** dataset.
