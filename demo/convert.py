# import onnx
# from onnx import helper
#
# # 加载模型
# model_path = "../model/palm_detection/palm_detection_full_inf_post_192x192.onnx"
# model = onnx.load(model_path)
#
# # 获取模型图
# graph = model.graph
#
# # 修改输入维度
# for input_tensor in graph.input:
#     # 获取输入的维度
#     shape = input_tensor.type.tensor_type.shape
#     if shape.dim[0].dim_value is not None:  # 如果 batch 维是固定的
#         shape.dim[0].dim_param = "batch_size"  # 将第 0 维设为动态，命名为 batch_size
#
# # 保存修改后的模型
# output_model_path = "../model/palm_detection/palm_detection_full_inf_post_192x192_dynamic.onnx"
# onnx.save(model, output_model_path)
#
# print(f"Saved the model with dynamic batch size to {output_model_path}")


import onnx
from onnx import helper, TensorProto

def modify_dynamic_batch(onnx_model_path, output_model_path):
    # 加载 ONNX 模型
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # 修改输入和输出的 Batch 维度为动态
    for input_tensor in graph.input:
        dim_proto = input_tensor.type.tensor_type.shape.dim
        if len(dim_proto) > 0:  # 确保有维度信息
            dim_proto[0].dim_param = 'batch_size'  # 将第一个维度设为动态，命名为 'batch_size'

    for output_tensor in graph.output:
        dim_proto = output_tensor.type.tensor_type.shape.dim
        if len(dim_proto) > 0:
            dim_proto[0].dim_param = 'batch_size'

    # 遍历图中的所有节点和中间值（通常这些已经被 ONNX 自动推导，通常无需手动修改）
    for value_info in graph.value_info:
        dim_proto = value_info.type.tensor_type.shape.dim
        if len(dim_proto) > 0:
            dim_proto[0].dim_param = 'batch_size'

    # 保存修改后的模型
    onnx.save(model, output_model_path)
    print(f"Model saved to: {output_model_path}")

# 示例使用
modify_dynamic_batch("../model/palm_detection/palm_detection_full_inf_post_192x192.onnx",
                   "../model/palm_detection/palm_detection_full_inf_post_192x192_dynamic.onnx")
