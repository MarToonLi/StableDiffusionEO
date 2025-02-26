import onnxruntime as ort

sess = ort.InferenceSession("./onnx/CLIP.onnx")
for input in sess.get_inputs():
    print(f"Input: {input.name}, Shape: {input.shape}")

for output in sess.get_outputs():
    print(f"Output: {output.name}, Shape: {output.shape}")