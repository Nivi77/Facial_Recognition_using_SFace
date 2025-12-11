This is a GPU-powered face-recognition system that:
-Loads face embeddings using SFace models (ONNX, CUDA accelerated).
-Processes a video file instead of webcam input.
-Detects faces, extracts embeddings, and compares them to a SQLite face database.
-Prints results to console every time a face is identified: name, confidence score, and distance.

dependecies
pip install opencv-python deepface numpy

✔️ If you have NVIDIA GPU
pip install onnxruntime-gpu

❌ If you do not have GPU
pip install onnxruntime
