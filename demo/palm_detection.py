import copy
from typing import (
    Tuple,
    Optional,
    List,
)
from math import (
    sin,
    cos,
    atan2,
    pi,
)

import cv2
import onnxruntime
import numpy as np

from utils.utils import (
    normalize_radians,
    keep_aspect_resize_and_pad,
)


class PalmDetection(object):
    def __init__(
            self,
            model_path: Optional[str] = '../model/palm_detection/palm_detection_full_inf_post_192x192_dynamic.onnx',
            score_threshold: Optional[float] = 0.60,
            providers: Optional[List] = [
                # (
                #     'TensorrtExecutionProvider', {
                #         'trt_engine_cache_enable': True,
                #         'trt_engine_cache_path': '.',
                #         'trt_fp16_enable': True,
                #     }
                # ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ],
    ):
        # Threshold
        self.score_threshold = score_threshold

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]
        self.square_standard_size = 0

    def __call__(
            self,
            image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # print(self.input_shapes)
        # print(image.shape)
        temp_image = copy.deepcopy(image)
        # PreProcess
        inference_image = self.__preprocess(
            temp_image,
        )
        # Inference
        inferece_image = np.asarray([inference_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )
        # boxes [n,8] 不过滤阈值手掌的检测结果非常多
        # print(boxes[0][0])
        # PostProcess
        hands = self.__postprocess(
            image=temp_image,
            boxes=boxes[0],
        )

        return hands

    def __preprocess(
            self,
            image: np.ndarray,
            swap: Optional[Tuple[int, int, int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

                Parameters
                ----------
                image: np.ndarray
                    Entire image

                swap: tuple
                    HWC to CHW: (2,0,1)
                    CHW to HWC: (1,2,0)
                    HWC to HWC: (0,1,2)
                    CHW to CHW: (0,1,2)

                Returns
                -------
                padded_image: np.ndarray
                    Resized and Padding and normalized image.
                """
        # Resize + Padding + Normalization + BGR->RGB

        # Resize + Padding + Normalization + BGR->RGB
        input_h = self.input_shapes[0][2]
        input_w = self.input_shapes[0][3]
        image_height, image_width = image.shape[:2]

        self.square_standard_size = max(image_height, image_width)  # 确定一个基准尺寸，使得图像可以通过填充成为正方形，且正方形边长等于较长边。
        self.square_padding_half_size = abs(image_height - image_width) // 2  # 计算短边需要填充的值 letter box

        # letter box
        padded_image, resized_image = keep_aspect_resize_and_pad(
            image=image,
            resize_width=input_w,
            resize_height=input_h,
        )

        # cv2.imshow('Image', image)
        # cv2.imshow('padded_image', padded_image)
        # cv2.imshow('resized_image', resized_image)
        # print('Image ', image.shape)
        # print('padded_image ', padded_image.shape)
        # print('resized_image ', resized_image.shape)

        # 没用到
        pad_size_half_h = max(0, (input_h - resized_image.shape[0]) // 2)
        pad_size_half_w = max(0, (input_w - resized_image.shape[1]) // 2)
        self.pad_size_scale_h = pad_size_half_h / input_h
        self.pad_size_scale_w = pad_size_half_w / input_w
        # 归一化
        padded_image = np.divide(padded_image, 255.0)
        # BGR 到 RGB
        padded_image = padded_image[..., ::-1]
        # HWC 到 CHW
        padded_image = padded_image.transpose(swap)
        # 保证图像内存连续性
        padded_image = np.ascontiguousarray(
            padded_image,
            dtype=np.float32,
        )

        # 处理过的图像已经不能显示了
        return padded_image

    def __postprocess(
            self,
            image: np.ndarray,
            boxes: np.ndarray,
    ) -> np.ndarray:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 8]
            pd_score(检测得分), box_x(框的中心 X 坐标), box_y(框的中心 Y 坐标), box_size(框的大小), kp0_x, kp0_y, kp2_x, kp2_y

            kp0_x（第一个关键点的 X 坐标）：这是手掌上某个关键点（通常是手指或掌心等部位）的 X 坐标。该关键点的位置通常是由模型预测的。
            kp0_y（第一个关键点的 Y 坐标）：这是手掌上某个关键点的 Y 坐标，表示该关键点在图像中的垂直位置。
            kp2_x（第二个关键点的 X 坐标）：这是手掌上另一个关键点（通常是手指或掌心等部位）的 X 坐标。这个点常常用于与第一个关键点进行比较，计算旋转角度等信息。
            kp2_y（第二个关键点的 Y 坐标）：这是手掌上另一个关键点的 Y 坐标，表示该关键点在图像中的垂直位置。

        Returns
        -------
        hands: np.ndarray
            float32[N, 4]
            sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        hands = []
        # 这里过滤出是bool值，大于阈值的框
        keep = boxes[:, 0] > self.score_threshold  # pd_score > self.score_threshold
        # boxes = boxes[keep, :] 是 NumPy 中的高级索引操作，用于根据布尔数组 keep 对 boxes 进行筛选。
        boxes = boxes[keep, :]
        # print(boxes)

        # 处理每一个box
        for box in boxes:
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
            if box_size > 0:
                # 两个计算角度的关键点相减
                kp02_x = kp2_x - kp0_x
                kp02_y = kp2_y - kp0_y
                # 放大box的size,这个size应该是矩形最长边
                sqn_rr_size = 2.9 * box_size
                # 这段代码的作用是计算一个点的旋转角度
                rotation = 0.5 * pi - atan2(-kp02_y, kp02_x)
                # 将角度值规范化到 [−π,π] 的范围
                rotation = normalize_radians(rotation)
                # 中心点根据旋转角度修改一下
                sqn_rr_center_x = box_x + 0.5 * box_size * sin(rotation)
                sqn_rr_center_y = box_y - 0.5 * box_size * cos(rotation)
                sqn_rr_center_y = (sqn_rr_center_y * self.square_standard_size - self.square_padding_half_size) / image_height
                hands.append(
                    [
                        sqn_rr_size,
                        rotation,
                        sqn_rr_center_x,
                        sqn_rr_center_y,
                    ]
                )
                # print("hands:", hands)

        return np.asarray(hands)