import tflite2onnx

tflite_path = '/home/berlin/Desktop/mediaPipe_hand_models/palm_detection.tflite'
onnx_path = '/home/berlin/Desktop/mediaPipe_hand_models/palm_detection.onnx'

tflite2onnx.convert(tflite_path, onnx_path)