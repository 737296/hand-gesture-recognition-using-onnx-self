import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
model_path = "../model/palm_detection/palm_detection_full_inf_post_192x192_dynamic.onnx"
session = ort.InferenceSession(model_path,
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

# 检查输入
input_details = session.get_inputs()
input_name = input_details[0].name
print(f"Input Name: {input_name}, Shape: {input_details[0].shape}")

# 测试动态输入
try:
    # 创建符合输入形状的数据
    input_data_1 = np.random.randn(1, 3, 192, 192).astype(np.float32)  # 默认大小
    input_data_2 = np.random.randn(8, 3, 192, 192).astype(np.float32)  # 不同大小
    output_1 = session.run(None, {input_name: input_data_1})
    output_2 = session.run(None, {input_name: input_data_2})
    print("Model supports dynamic axes.")
except Exception as e:
    print(f"Model does not support dynamic axes: {e}")
